# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from typing import Any
import os
import json
import re
import tempfile
import copy
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import qwix
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    split_state_dict_into_shards_factory,
)
from safetensors.flax import save_file

from taktiny.nn import Module, Rngs
from taktiny.nn.module import iter_children
from taktiny.utils.typing import AxisNames


class PretrainedModel(Module):
    """
    Base class for models that load and save pretrained checkpoints.

    Full models are serialized as Safetensors together with a weight index.
    Qwix arrays retain their quantized components and reconstruction metadata.
    LoRA-transformed models instead save adapter tensors and reconstruction
    metadata. Loading first constructs an abstract parameter tree with
    ``jax.eval_shape``, then maps checkpoint names to module paths, applies any
    requested quantization, and places arrays using parameter sharding metadata.

    Subclasses are expected to accept ``config`` and ``rngs`` in their
    constructor. They may provide module-mapping rules to translate external
    checkpoint names and may expose default logical sharding rules.
    """

    def _config_dict(self) -> Any:
        config = getattr(self, 'config', None)
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        to_dict = getattr(config, 'to_dict', None)
        if callable(to_dict):
            return to_dict()
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith('_')
        }

    def _save_config(self, path: str) -> Any:
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as config_file:
            json.dump(
                self._config_dict(),
                config_file,
                indent=2,
                default=str,
            )
        return config_path

    @staticmethod
    def _qtype_name(qtype: Any) -> Any:
        if isinstance(qtype, str):
            return qtype
        return jnp.dtype(qtype).name

    @staticmethod
    def _safetensors_qvalue(array: Any) -> Any:
        dtype = array.dtype
        if jnp.issubdtype(dtype, jnp.signedinteger):
            storage_dtype = np.int8
        elif jnp.issubdtype(dtype, jnp.unsignedinteger):
            storage_dtype = np.uint8
        elif jnp.issubdtype(dtype, jnp.floating):
            storage_dtype = np.float16
        else:
            raise TypeError(
                f'Unsupported Qwix qvalue dtype for serialization: {dtype}'
            )
        return np.asarray(jax.device_get(array), dtype=storage_dtype)

    @classmethod
    def _encode_qwix_state(cls, state: Any) -> tuple[Any, ...]:
        encoded = {}
        parameters = {}

        for name, value in state.items():
            if not isinstance(value, qwix.QArray):
                encoded[name] = value
                continue

            component_prefix = f'{name}.__qwix__'
            qvalue_name = f'{component_prefix}.qvalue'
            scale_name = f'{component_prefix}.scale'
            zero_point_name = (
                f'{component_prefix}.zero_point'
                if value.zero_point is not None
                else None
            )
            encoded[qvalue_name] = cls._safetensors_qvalue(value.qvalue)
            encoded[scale_name] = value.scale
            if zero_point_name is not None:
                encoded[zero_point_name] = cls._safetensors_qvalue(
                    value.zero_point
                )
            parameters[name] = {
                'qtype': cls._qtype_name(value.qtype),
                'qvalue_dtype': value.qvalue.dtype.name,
                'qvalue': qvalue_name,
                'scale': scale_name,
                'zero_point': zero_point_name,
            }

        metadata = None
        if parameters:
            metadata = {
                'format': 'taktiny-qwix',
                'version': 1,
                'parameters': parameters,
            }
        return encoded, metadata

    @staticmethod
    def _decode_qwix_state(state: Any, metadata: Any) -> Any:
        if metadata is None:
            return state
        if (
            metadata.get('format') != 'taktiny-qwix'
            or metadata.get('version') != 1
        ):
            raise ValueError('Unsupported Qwix checkpoint metadata format')
        parameters = metadata.get('parameters')
        if not isinstance(parameters, dict):
            raise ValueError(
                'Qwix checkpoint metadata has no parameter mapping'
            )

        decoded = dict(state)
        for name, specification in parameters.items():
            if not isinstance(specification, dict):
                raise ValueError(
                    f'Invalid Qwix metadata for parameter {name!r}'
                )
            component_names = (
                specification.get('qvalue'),
                specification.get('scale'),
                specification.get('zero_point'),
            )
            required = component_names[:2]
            missing = [
                component
                for component in required
                if component not in decoded
            ]
            if missing:
                raise ValueError(
                    f'Qwix parameter {name!r} is missing components: '
                    f'{", ".join(missing)}'
                )

            qvalue = decoded.pop(component_names[0])
            scale = decoded.pop(component_names[1])
            zero_point = None
            if component_names[2] is not None:
                if component_names[2] not in decoded:
                    raise ValueError(
                        f'Qwix parameter {name!r} is missing component '
                        f'{component_names[2]!r}'
                    )
                zero_point = decoded.pop(component_names[2])

            qvalue_dtype = specification.get('qvalue_dtype')
            if not isinstance(qvalue_dtype, str):
                raise ValueError(
                    f'Qwix parameter {name!r} has no qvalue dtype'
                )
            try:
                qvalue_dtype = jnp.dtype(qvalue_dtype)
            except TypeError as error:
                raise ValueError(
                    f'Qwix parameter {name!r} has unsupported qvalue dtype '
                    f'{qvalue_dtype!r}'
                ) from error
            qvalue = jnp.asarray(qvalue).astype(qvalue_dtype)
            if zero_point is not None:
                zero_point = jnp.asarray(zero_point).astype(qvalue_dtype)
            decoded[name] = qwix.QArray(
                qvalue=qvalue,
                scale=jnp.asarray(scale),
                zero_point=zero_point,
                qtype=specification.get('qtype'),
            )
        return decoded

    def _lora_state_dict(self) -> Any:
        from taktiny.nn.lora import LoRALinear

        state = {}

        def collect(module: Any, prefix: str='') -> None:
            for name, child in iter_children(module):
                full_name = f'{prefix}.{name}' if prefix else name
                if isinstance(child, LoRALinear):
                    state[f'{full_name}.lora_A'] = child.lora_A.value
                    state[f'{full_name}.lora_B'] = child.lora_B.value
                elif isinstance(child, Module):
                    collect(child, full_name)

        collect(self)
        return state

    @staticmethod
    def _host_state_snapshot(state: Any) -> Any:
        def copy_leaf(value: Any) -> Any:
            value = jax.device_get(value)
            if isinstance(value, np.ndarray):
                return np.array(value, copy=True)
            return value

        return jax.tree.map(copy_leaf, state)

    def _checkpoint_snapshot(self) -> dict[Any, Any]:
        """Capture stable host state for background checkpoint writing."""
        adapter_state = self._expand_stacked_state_dict(
            self._lora_state_dict()
        )
        if adapter_state:
            peft_config = getattr(self, 'peft_config', None)
            if peft_config is None:
                raise ValueError(
                    'LoRA modules were found but PEFT configuration metadata '
                    'is missing; apply LoRA through Takt.apply_peft'
                )
            return {
                'kind': 'adapter',
                'config': copy.deepcopy(self._config_dict()),
                'peft_config': copy.deepcopy(peft_config),
                'state': self._host_state_snapshot(adapter_state),
            }

        state = self._expand_stacked_state_dict(self.flat_state_dict())
        return {
            'kind': 'model',
            'config': copy.deepcopy(self._config_dict()),
            'state': self._host_state_snapshot(state),
        }

    @classmethod
    def _save_pretrained_snapshot(
        cls,
        snapshot: Any,
        path: str,
        *,
        max_shard_size: str='5GB',
    ) -> tuple[Any, ...]:
        os.makedirs(path, exist_ok=True)
        model_config_path = os.path.join(path, 'config.json')
        with open(model_config_path, 'w') as config_file:
            json.dump(
                snapshot['config'],
                config_file,
                indent=2,
                default=str,
            )

        if snapshot['kind'] == 'adapter':
            config_path = os.path.join(path, 'adapter_config.json')
            with open(config_path, 'w') as config_file:
                json.dump(snapshot['peft_config'], config_file, indent=2)
            adapter_paths = cls._save_safetensors(
                snapshot['state'],
                path,
                'adapter_model.safetensors',
                max_shard_size=max_shard_size,
            )
            return (
                model_config_path,
                config_path,
                *adapter_paths,
            )

        state_dict, quantization_metadata = cls._encode_qwix_state(
            snapshot['state']
        )
        quantization_path = os.path.join(
            path,
            'quantization_config.json',
        )
        if quantization_metadata is not None:
            with open(quantization_path, 'w') as quantization_file:
                json.dump(
                    quantization_metadata,
                    quantization_file,
                    indent=2,
                )
        elif os.path.isfile(quantization_path):
            os.remove(quantization_path)
            quantization_path = None
        else:
            quantization_path = None
        checkpoint_paths = cls._save_safetensors(
            state_dict,
            path,
            'model.safetensors',
            max_shard_size=max_shard_size,
            always_write_index=True,
        )
        return (
            model_config_path,
            *((quantization_path,) if quantization_path else ()),
            *checkpoint_paths,
        )

    @staticmethod
    def _expand_stacked_state_dict(state: Any) -> Any:
        layout = []
        stacked_groups = {}

        for name, value in state.items():
            parts = name.split('.')
            if 'stacked' not in parts:
                layout.append(('parameter', name, value))
                continue

            stacked_index = parts.index('stacked')
            group_key = (tuple(parts[:stacked_index]), stacked_index)
            if group_key not in stacked_groups:
                stacked_groups[group_key] = []
                layout.append(('stack', group_key))
            stacked_groups[group_key].append((parts, value))

        expanded = {}
        for entry in layout:
            if entry[0] == 'parameter':
                _, name, value = entry
                expanded[name] = value
                continue

            _, group_key = entry
            group = stacked_groups[group_key]
            stacked_index = group_key[1]
            num_layers = None
            for parts, value in group:
                name = '.'.join(parts)
                if not getattr(value, 'shape', ()):
                    raise ValueError(
                        f'Stacked parameter {name!r} has no leading layer axis'
                    )
                if num_layers is None:
                    num_layers = value.shape[0]
                elif value.shape[0] != num_layers:
                    raise ValueError(
                        'Parameters in the same stack have inconsistent '
                        f'layer counts: expected {num_layers}, found '
                        f'{value.shape[0]} for {name!r}'
                    )

            for layer_index in range(num_layers):
                for parts, value in group:
                    layer_parts = list(parts)
                    layer_parts[stacked_index] = str(layer_index)
                    expanded['.'.join(layer_parts)] = value[layer_index]

        return expanded

    @staticmethod
    def _save_safetensors(
        state: Any,
        path: str,
        filename: Any,
        *,
        max_shard_size: int,
        always_write_index: bool=False,
    ) -> Any:
        stem, extension = os.path.splitext(filename)
        split = split_state_dict_into_shards_factory(
            state,
            get_storage_size=lambda value: int(value.nbytes),
            filename_pattern=f'{stem}{{suffix}}{extension}',
            max_shard_size=max_shard_size,
        )

        shard_pattern = re.compile(
            rf'{re.escape(stem)}-\d{{5}}-of-\d{{5}}'
            rf'{re.escape(extension)}'
        )
        for existing_filename in os.listdir(path):
            if (
                existing_filename == filename
                or shard_pattern.fullmatch(existing_filename)
                or existing_filename == f'{filename}.index.json'
            ):
                os.remove(os.path.join(path, existing_filename))

        saved_paths = []
        for shard_filename, tensor_names in (
            split.filename_to_tensors.items()
        ):
            shard_path = os.path.join(path, shard_filename)
            save_file(
                {name: state[name] for name in tensor_names},
                shard_path,
            )
            saved_paths.append(shard_path)

        if split.is_sharded or always_write_index:
            index_path = os.path.join(
                path,
                f'{filename}.index.json',
            )
            with open(index_path, 'w') as index_file:
                json.dump(
                    {
                        'metadata': split.metadata,
                        'weight_map': split.tensor_to_filename,
                    },
                    index_file,
                    indent=2,
                )
            saved_paths.append(index_path)

        return tuple(saved_paths)

    def save_pretrained(self, path: str, max_shard_size: str='5GB') -> Any:
        """Save a full model checkpoint or the model's LoRA adapters.

        Models containing ``LoRALinear`` modules save only adapter tensors and
        their reconstruction metadata. Models without LoRA save their complete
        parameter state and a Safetensors index. Parameters held by a
        ``SeqStack`` are expanded into conventional numbered layer keys.

        Args:
            path: Directory in which to write the checkpoint.
            max_shard_size: Maximum tensor data size per Safetensors file,
                expressed as an integer byte count or a string using ``KB``,
                ``MB``, ``GB``, or ``TB``, such as ``"5GB"``. A tensor larger
                than the limit is saved alone without being split.

        Returns:
            A tuple containing the paths written by this invocation, with
            configuration files first, followed by weight files and their
            index when present.
        """
        return self._save_pretrained_snapshot(
            self._checkpoint_snapshot(),
            path,
            max_shard_size=max_shard_size,
        )

    def load_pretrained(self, path: str) -> Any:
        """Load a Taktiny-native full checkpoint into this model in place.

        This is the inverse of ``save_pretrained`` for full-model checkpoints.
        Numbered checkpoint layers are reconstructed into ``SeqStack``
        parameters without applying external checkpoint name mappings or
        matrix transpositions.

        Args:
            path: Local directory containing model Safetensors.

        Returns:
            This model instance.
        """
        from safetensors import safe_open

        path = os.fspath(path)
        quantization_path = os.path.join(
            path,
            'quantization_config.json',
        )
        quantization_metadata = None
        if os.path.isfile(quantization_path):
            with open(quantization_path) as quantization_file:
                quantization_metadata = json.load(quantization_file)
        index_path = os.path.join(
            path,
            'model.safetensors.index.json',
        )
        if os.path.isfile(index_path):
            with open(index_path) as index_file:
                index = json.load(index_file)
            weight_map = index.get('weight_map')
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    'Model Safetensors index has no weight_map'
                )
            filenames = dict.fromkeys(weight_map.values())
        else:
            filenames = {'model.safetensors': None}

        parameters = self.flat_parameter_dict()
        checkpoint_state = {}
        loaded = {}
        stacked_parameters = {}
        unexpected = []

        for filename in filenames:
            checkpoint_path = os.path.join(path, filename)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f'Model checkpoint file was not found: {checkpoint_path}'
                )
            with safe_open(
                checkpoint_path,
                framework='np',
                device='cpu',
            ) as checkpoint:
                for name in checkpoint.keys():
                    if name in checkpoint_state:
                        raise ValueError(
                            f'Duplicate model tensor in checkpoint: {name}'
                        )
                    checkpoint_state[name] = checkpoint.get_tensor(name)

        checkpoint_state = self._decode_qwix_state(
            checkpoint_state,
            quantization_metadata,
        )
        for name, value in checkpoint_state.items():
            if name in parameters:
                parameter = parameters[name]
                if value.shape != parameter.shape:
                    raise ValueError(
                        f'Model tensor {name!r} has shape '
                        f'{value.shape}, expected {parameter.shape}'
                    )
                loaded[name] = value
                continue

            matched_stack = False
            parts = name.split('.')
            for position, part in enumerate(parts):
                if not part.isdigit():
                    continue
                stacked_parts = list(parts)
                stacked_parts[position] = 'stacked'
                stacked_name = '.'.join(stacked_parts)
                if stacked_name not in parameters:
                    continue

                parameter = parameters[stacked_name]
                layer_index = int(part)
                if layer_index >= parameter.shape[0]:
                    raise ValueError(
                        f'Model layer index {layer_index} is out of '
                        f'range for {stacked_name!r}'
                    )
                expected_shape = parameter.shape[1:]
                if value.shape != expected_shape:
                    raise ValueError(
                        f'Model tensor {name!r} has shape '
                        f'{value.shape}, expected {expected_shape}'
                    )
                entry = stacked_parameters.setdefault(
                    stacked_name,
                    {'values': {}, 'indices': set()},
                )
                if layer_index in entry['indices']:
                    raise ValueError(
                        f'Duplicate model layer tensor: {name}'
                    )
                entry['values'][layer_index] = value
                entry['indices'].add(layer_index)
                matched_stack = True
                break

            if not matched_stack:
                unexpected.append(name)

        if unexpected:
            preview = ', '.join(sorted(unexpected)[:8])
            raise ValueError(
                f'Model checkpoint contains unexpected tensors: {preview}'
            )

        for name, entry in stacked_parameters.items():
            parameter = parameters[name]
            expected_indices = set(range(parameter.shape[0]))
            missing_indices = expected_indices - entry['indices']
            if missing_indices:
                missing = ', '.join(map(str, sorted(missing_indices)))
                raise ValueError(
                    f'Model checkpoint is missing layers {missing} '
                    f'for {name!r}'
                )
            ordered = [
                entry['values'][index]
                for index in range(parameter.shape[0])
            ]
            if isinstance(ordered[0], qwix.QArray):
                loaded[name] = jax.tree.map(
                    lambda *values: jnp.stack(values),
                    *ordered,
                )
            else:
                loaded[name] = np.stack(ordered)

        missing = sorted(set(parameters) - set(loaded))
        if missing:
            preview = ', '.join(missing[:8])
            raise ValueError(
                f'Model checkpoint is missing tensors: {preview}'
            )

        for name, value in loaded.items():
            parameter = parameters[name]
            if (
                isinstance(parameter.value, qwix.QArray)
                and not isinstance(value, qwix.QArray)
            ):
                raise TypeError(
                    'Loading a dense native checkpoint into an existing '
                    f'quantized parameter is unsupported: {name}'
                )
            if isinstance(value, qwix.QArray):
                target = parameter.value

                def place(component: Any, target_component: Any=None) -> Any:
                    component = jnp.asarray(component)
                    sharding = getattr(target_component, 'sharding', None)
                    if sharding is not None:
                        component = jax.device_put(component, sharding)
                    return component

                if isinstance(target, qwix.QArray):
                    value = qwix.QArray(
                        qvalue=place(value.qvalue, target.qvalue),
                        scale=place(value.scale, target.scale),
                        zero_point=(
                            place(value.zero_point, target.zero_point)
                            if value.zero_point is not None
                            else None
                        ),
                        qtype=value.qtype,
                    )
                else:
                    value = jax.tree.map(place, value)
                parameter.value = value
                continue
            array = jnp.asarray(value, dtype=parameter.dtype)
            sharding = getattr(parameter.value, 'sharding', None)
            if sharding is not None:
                array = jax.device_put(array, sharding)
            parameter.value = array

        return self

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: Any=None,
        commit_description: Any=None,
        private: Any=None,
        token: Any=None,
        revision: Any=None,
        create_pr: bool=False,
        max_shard_size: str='5GB',
    ) -> str:
        """Save and upload this model or adapter to the Hugging Face Hub.

        The checkpoint is staged in a temporary directory and removed after
        the upload completes. Existing unrelated repository files are
        preserved, while obsolete shards belonging to the uploaded checkpoint
        family are deleted in the same commit.

        Args:
            repo_id: Hub repository identifier, optionally including an
                organization or username.
            commit_message: Optional Hub commit title.
            commit_description: Optional longer commit description.
            private: Visibility used when creating a new repository.
            token: Hugging Face authentication token or token-selection flag.
            revision: Branch or revision to receive the commit.
            create_pr: Whether to create a pull request instead of committing
                directly to the target revision.
            max_shard_size: Maximum size passed to ``save_pretrained``.

        Returns:
            The URL of the created Hub commit or pull request.
        """
        api = HfApi(
            token=token,
            library_name='taktiny',
        )
        repo = api.create_repo(
            repo_id=repo_id,
            private=private,
            token=token,
            repo_type='model',
            exist_ok=True,
        )
        resolved_repo_id = getattr(repo, 'repo_id', repo_id)

        if revision is not None and not revision.startswith('refs/pr'):
            api.create_branch(
                repo_id=resolved_repo_id,
                branch=revision,
                token=token,
                exist_ok=True,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            saved_paths = self.save_pretrained(
                temporary_directory,
                max_shard_size=max_shard_size,
            )
            filenames = {
                os.path.basename(path)
                for path in saved_paths
            }
            is_adapter = any(
                filename.startswith('adapter_model')
                for filename in filenames
            )
            stem = 'adapter_model' if is_adapter else 'model'
            delete_patterns = [
                f'{stem}.safetensors',
                f'{stem}-*-of-*.safetensors',
                f'{stem}.safetensors.index.json',
            ]
            if not is_adapter:
                delete_patterns.append('quantization_config.json')

            commit = api.upload_folder(
                repo_id=resolved_repo_id,
                folder_path=temporary_directory,
                commit_message=commit_message or 'Upload model',
                commit_description=commit_description,
                token=token,
                repo_type='model',
                revision=revision,
                create_pr=create_pr,
                delete_patterns=delete_patterns,
            )

        commit_url = getattr(commit, 'commit_url', commit)
        return str(commit_url)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: Any,
        config: Any,
        module_map: Any=None,
        local: bool=False,
        dtype: Any=None,
        quant: Any=None,
        subfolder: Any=None,
        mesh: Any=None,
        sharding_rules: Any=None,
        **kwargs: Any
    ) -> Any:
        """
        Loads safetensors weights into a newly instantiated model.
        Supports both single-file (model.safetensors) and sharded models.
        """
        uniform_quant = None
        if dtype is not None:
            dtype_name = dtype.lower() if isinstance(dtype, str) else None
            quantized_dtypes = {'fp8', 'int8', 'int4', 'nf4'}
            if dtype_name in quantized_dtypes:
                compute_dtype = (
                    getattr(config, 'torch_dtype', None)
                    or getattr(config, 'dtype', None)
                )
                if (
                    compute_dtype is None
                    or (
                        isinstance(compute_dtype, str)
                        and compute_dtype.lower() in quantized_dtypes
                    )
                ):
                    compute_dtype = 'bfloat16'

                uniform_quant = dtype_name
                setattr(config, 'dtype', compute_dtype)
                setattr(config, 'torch_dtype', compute_dtype)
            else:
                setattr(config, 'dtype', dtype)
                setattr(config, 'torch_dtype', dtype)
        if quant is not None and uniform_quant is not None:
            from ..utils.quantization import merge_quantization

            setattr(
                config,
                'quant',
                merge_quantization(quant, uniform_quant),
            )
        elif quant is not None:
            setattr(config, 'quant', quant)
        elif uniform_quant is not None:
            setattr(config, 'quant', uniform_quant)

        path_or_repo_str = str(path_or_repo)
        module_map = module_map or []
        if isinstance(module_map, dict):
            module_map = list(module_map.items())
        native_qwix_directory = None
        if local:
            candidate = os.path.join(
                path_or_repo_str,
                subfolder if subfolder else '',
                'quantization_config.json',
            )
            if os.path.isfile(candidate):
                native_qwix_directory = os.path.dirname(candidate)

        # 1. Determine if model is sharded or single file
        is_sharded = False
        if local:
            index_path = os.path.join(path_or_repo_str, subfolder if subfolder else "", "model.safetensors.index.json")
            if os.path.exists(index_path):
                is_sharded = True
        else:
            from huggingface_hub import repo_info
            try:
                info = repo_info(repo_id=path_or_repo_str)
                files = [f.rfilename for f in info.siblings]
                target_index = f"{subfolder}/model.safetensors.index.json" if subfolder else "model.safetensors.index.json"
                if target_index in files:
                    is_sharded = True
                    index_path = hf_hub_download(repo_id=path_or_repo_str, subfolder=subfolder, filename="model.safetensors.index.json")
                target_quantization = (
                    f'{subfolder}/quantization_config.json'
                    if subfolder
                    else 'quantization_config.json'
                )
                if target_quantization in files:
                    quantization_path = hf_hub_download(
                        repo_id=path_or_repo_str,
                        subfolder=subfolder,
                        filename='quantization_config.json',
                    )
                    native_qwix_directory = os.path.dirname(
                        quantization_path
                    )
            except Exception as e:
                print(f"Failed to fetch repo info: {e}")
                is_sharded = False

        # 2. Build files_to_load mapping: file_name -> list of keys (or None for all)
        files_to_load = {}
        if is_sharded:
            with open(index_path, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            for k_str, file_name in weight_map.items():
                if file_name not in files_to_load:
                    files_to_load[file_name] = []
                files_to_load[file_name].append(k_str)
        else:
            files_to_load["model.safetensors"] = None

        # 3. Resolve every checkpoint file before materializing any parameters.
        resolved_files = {}
        for file_name in files_to_load:
            if local:
                resolved_files[file_name] = os.path.join(
                    path_or_repo_str,
                    subfolder if subfolder else "",
                    file_name,
                )
            else:
                resolved_files[file_name] = hf_hub_download(
                    repo_id=path_or_repo_str,
                    subfolder=subfolder,
                    filename=file_name,
                )

        # 4. Instantiate model skeleton using eval_shape (no memory allocation)
        rngs = kwargs.pop('rngs', Rngs(0))
        state = jax.eval_shape(
            lambda: cls(
                config,
                rngs=rngs,
                mesh=mesh,
                sharding_rules=sharding_rules,
                **kwargs,
            )
        )
        if native_qwix_directory is not None:
            state.load_pretrained(native_qwix_directory)
            state.base_model_name_or_path = path_or_repo_str
            return state

        current_state_dict = state.flat_parameter_dict()
        new_state = {}
        not_found_some = False

        # 5. Load weights
        import numpy as np
        from safetensors import safe_open
        from ..utils.quantization import (
            quantize_embedding_weight,
            quantize_linear_weight,
            resolve_quantization_rule,
        )
        from ..utils.weights import map_state_dict

        cpu_device = jax.devices('cpu')[0]
        default_device = jax.devices()[0]

        def parameter_sharding(
            parameter: Any,
            axis_names: AxisNames | None=None,
            *,
            use_explicit: bool=True,
        ) -> Any:
            sharding = (
                getattr(parameter, 'sharding', None)
                if use_explicit
                else None
            )
            if (
                sharding is None
                and axis_names is not None
                and mesh is not None
            ):
                from ..utils.sharding import create_sharding

                sharding = create_sharding(
                    mesh,
                    axis_names,
                    rules=sharding_rules,
                )
            if sharding is None and mesh is None:
                sharding = default_device
            return sharding

        def place_qarray(value: Any, parameter: Any) -> Any:
            axis_names = getattr(parameter, 'axis_names', None)
            qvalue_sharding = parameter_sharding(parameter, axis_names)

            scale_axis_names = None
            if axis_names is not None:
                scale_axis_names = tuple(
                    axis_name if size != 1 else None
                    for axis_name, size in zip(
                        axis_names,
                        value.scale.shape,
                    )
                )
            scale_sharding = parameter_sharding(
                parameter,
                scale_axis_names,
                use_explicit=False,
            )

            zero_point = value.zero_point
            if zero_point is not None:
                zero_axis_names = None
                if axis_names is not None:
                    zero_axis_names = tuple(
                        axis_name if size != 1 else None
                        for axis_name, size in zip(
                            axis_names,
                            zero_point.shape,
                        )
                    )
                zero_point = jax.device_put(
                    zero_point,
                    parameter_sharding(
                        parameter,
                        zero_axis_names,
                        use_explicit=False,
                    ),
                )

            return value.replace(
                qvalue=jax.device_put(value.qvalue, qvalue_sharding),
                scale=jax.device_put(value.scale, scale_sharding),
                zero_point=zero_point,
            )

        def parameter_quantization_rule(key: Any, parameter: Any) -> tuple[Any, ...]:
            quantization = getattr(parameter, 'quantization', None)
            quantization_kind = getattr(
                parameter,
                'quantization_kind',
                'linear',
            )
            rule = resolve_quantization_rule(
                quantization,
                key.rsplit('.', 1)[0],
                op_name=quantization_kind,
            )
            return rule, quantization_kind

        def materialize_parameter(key: Any, value: Any, parameter: Any) -> Any:
            rule, quantization_kind = parameter_quantization_rule(
                key,
                parameter,
            )
            if rule is not None:
                parameter.trainable = False
                with jax.default_device(cpu_device):
                    if quantization_kind == 'embedding':
                        quantized = quantize_embedding_weight(
                            value,
                            parameter,
                            rule,
                        )
                    else:
                        quantized = quantize_linear_weight(
                            value,
                            parameter,
                            rule,
                        )
                return place_qarray(quantized, parameter)

            target_dtype = np.dtype(parameter.dtype)
            if value.dtype != target_dtype:
                value = value.astype(target_dtype, copy=False)
            return jax.device_put(
                value,
                parameter_sharding(
                    parameter,
                    getattr(parameter, 'axis_names', None),
                ),
            )

        def initialize_stacked_parameter(parameter: Any) -> Any:
            sharding = parameter_sharding(
                parameter,
                getattr(parameter, 'axis_names', None),
            )
            shape = tuple(parameter.shape)
            dtype = jnp.dtype(parameter.dtype)
            if isinstance(sharding, jax.sharding.Sharding):
                return jax.jit(
                    lambda: jnp.zeros(shape, dtype=dtype),
                    out_shardings=sharding,
                )()
            return jax.device_put(jnp.zeros(shape, dtype=dtype), sharding)

        def update_stacked_parameter(stacked: Any, layer: Any, layer_index: int) -> Any:
            return jax.lax.dynamic_update_index_in_dim(
                stacked,
                layer,
                layer_index,
                axis=0,
            )

        update_stacked_parameter = jax.jit(
            update_stacked_parameter,
            donate_argnums=(0,),
        )

        stacked_states = {}
        grouped_mapping = any(
            len(rule) == 3
            and isinstance(rule[0], (list, tuple))
            and len(rule[0]) > 1
            for rule in module_map
        )

        for file_name, keys_in_file in files_to_load.items():
            shard_path = resolved_files[file_name]
            with safe_open(shard_path, framework="np", device="cpu") as f:
                keys_to_process = keys_in_file if keys_in_file is not None else f.keys()

                if grouped_mapping:
                    shard = {key: f.get_tensor(key) for key in keys_to_process}
                    mapped_items = map_state_dict(shard, module_map).items()
                else:
                    mapped_items = (
                        item
                        for key in keys_to_process
                        for item in map_state_dict(
                            {key: f.get_tensor(key)},
                            module_map,
                        ).items()
                    )

                for k_mapped, value in mapped_items:
                    if k_mapped in current_state_dict:
                        target_var = current_state_dict[k_mapped]

                        if value.ndim == 2:
                            if k_mapped.endswith(".weight") or ".lora_" in k_mapped:
                                value = value.T
                        if value.shape != target_var.shape:
                            value = value.reshape(target_var.shape)
                        new_state[k_mapped] = materialize_parameter(
                            k_mapped,
                            value,
                            target_var,
                        )

                    else:
                        # Check if it belongs to a SeqStack
                        match = re.search(r'\.(\d+)\.', k_mapped)
                        if match:
                            idx = int(match.group(1))
                            k_stacked = k_mapped[:match.start()] + '.stacked.' + k_mapped[match.end():]

                            if k_stacked in current_state_dict:
                                target_var = current_state_dict[k_stacked]

                                layer_shape = target_var.shape[1:]
                                if value.ndim == 2:
                                    if k_mapped.endswith(".weight") or ".lora_" in k_mapped:
                                        value = value.T
                                if value.shape != layer_shape:
                                    value = value.reshape(layer_shape)

                                stacked_state = stacked_states.get(k_stacked)
                                if stacked_state is None:
                                    rule, quantization_kind = (
                                        parameter_quantization_rule(
                                            k_stacked,
                                            target_var,
                                        )
                                    )
                                    if rule is None:
                                        stacked_state = {
                                            'kind': 'dense',
                                            'value': initialize_stacked_parameter(
                                                target_var
                                            ),
                                            'indices': set(),
                                        }
                                    else:
                                        target_var.trainable = False
                                        stacked_state = {
                                            'kind': 'quantized',
                                            'layers': {},
                                            'rule': rule,
                                            'quantization_kind': (
                                                quantization_kind
                                            ),
                                        }
                                    stacked_states[k_stacked] = stacked_state

                                loaded_indices = (
                                    stacked_state['indices']
                                    if stacked_state['kind'] == 'dense'
                                    else stacked_state['layers']
                                )
                                if idx in loaded_indices:
                                    raise ValueError(
                                        'Checkpoint contains duplicate layer '
                                        f'{idx} for {k_stacked!r}'
                                    )

                                if stacked_state['kind'] == 'quantized':
                                    layer_parameter = SimpleNamespace(
                                        dtype=target_var.dtype,
                                        input_axis_count=getattr(
                                            target_var,
                                            'input_axis_count',
                                            None,
                                        ),
                                        quantization_batch_axis_count=max(
                                            0,
                                            getattr(
                                                target_var,
                                                'quantization_batch_axis_count',
                                                1,
                                            )
                                            - 1,
                                        ),
                                    )
                                    with jax.default_device(cpu_device):
                                        if (
                                            stacked_state[
                                                'quantization_kind'
                                            ]
                                            == 'embedding'
                                        ):
                                            layer_value = quantize_embedding_weight(
                                                value,
                                                layer_parameter,
                                                stacked_state['rule'],
                                            )
                                        else:
                                            layer_value = quantize_linear_weight(
                                                value,
                                                layer_parameter,
                                                stacked_state['rule'],
                                            )
                                    stacked_state['layers'][idx] = layer_value
                                else:
                                    target_dtype = np.dtype(target_var.dtype)
                                    if value.dtype != target_dtype:
                                        value = value.astype(
                                            target_dtype,
                                            copy=False,
                                        )
                                    axis_names = getattr(
                                        target_var,
                                        'axis_names',
                                        None,
                                    )
                                    layer_axis_names = (
                                        tuple(axis_names[1:])
                                        if axis_names is not None
                                        else None
                                    )
                                    layer_value = jax.device_put(
                                        value,
                                        parameter_sharding(
                                            target_var,
                                            layer_axis_names,
                                            use_explicit=False,
                                        ),
                                    )
                                    stacked_state['value'] = (
                                        update_stacked_parameter(
                                            stacked_state['value'],
                                            layer_value,
                                            jnp.asarray(idx, dtype=jnp.int32),
                                        )
                                    )
                                    stacked_state['indices'].add(idx)
                                continue

                        not_found_some = True
                        print(f"Warning: mapped key {k_mapped} found in checkpoint but not in model.")

        # Move accumulated SeqStack weights to JAX
        for k_stacked, stacked_state in stacked_states.items():
            target_var = current_state_dict[k_stacked]
            expected_indices = set(range(target_var.shape[0]))
            loaded_indices = (
                stacked_state['indices']
                if stacked_state['kind'] == 'dense'
                else set(stacked_state['layers'])
            )
            if loaded_indices != expected_indices:
                missing = sorted(expected_indices - loaded_indices)
                raise ValueError(
                    f'Checkpoint is missing layers {missing} for '
                    f'{k_stacked!r}'
                )

            if stacked_state['kind'] == 'dense':
                new_state[k_stacked] = stacked_state['value']
                continue

            ordered_layers = [
                stacked_state['layers'][index]
                for index in range(target_var.shape[0])
            ]
            stacked_array = jax.tree.map(
                lambda *values: jnp.stack(values, axis=0),
                *ordered_layers,
            )
            new_state[k_stacked] = place_qarray(
                stacked_array,
                target_var,
            )

        if not_found_some:
            print("\nSome modules from the checkpoint were not found in this model.")
            print("You can try to map module names using module_map.")
            print("e.g. module_map = {'target_module': 'name_to_change'}")

        missing_parameters = sorted(set(current_state_dict) - set(new_state))
        if missing_parameters:
            preview = ', '.join(missing_parameters[:8])
            if len(missing_parameters) > 8:
                preview += f', ... ({len(missing_parameters)} total)'
            raise ValueError(
                'Checkpoint did not provide values for model parameters: '
                f'{preview}'
            )

        # 6. Inject actual arrays into the PyTree skeleton
        state.load_flat_state_dict(new_state)
        state.base_model_name_or_path = path_or_repo_str
        return state

__all__ = ['PretrainedModel']
