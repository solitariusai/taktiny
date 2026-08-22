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
"""Public model-transformation API."""
from __future__ import annotations
from collections.abc import Callable
import json
import os
from typing import Any, TypeVar
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open

from taktiny.nn import Rngs
from taktiny.nn.module import Module, iter_children
from taktiny.utils.typing import DType, PathLike

M = TypeVar('M', bound=Module)


def _replace_child(parent: Module, name: str, child: Module) -> None:
    if name.isdigit() and hasattr(parent, 'layers'):
        position = int(name)
        if isinstance(parent.layers, tuple):
            updated = list(parent.layers)
            updated[position] = child
            parent.layers = tuple(updated)
        else:
            parent.layers[position] = child
        return

    if '.' in name:
        attribute, index = name.rsplit('.', 1)
        sequence = getattr(parent, attribute)
        position = int(index)
        if isinstance(sequence, tuple):
            updated = list(sequence)
            updated[position] = child
            setattr(parent, attribute, tuple(updated))
        else:
            sequence[position] = child
        return

    setattr(parent, name, child)


class Takt:
    """
    Apply registered transformations to existing model instances.

    ``Takt`` complements ``Maestro``: Maestro constructs and loads a model,
    while Takt transforms a model that already exists. PEFT implementations
    are selected by configuration type so new methods can be added without
    changing the public ``apply_peft`` signature.
    """

    _peft_methods: dict[type, Callable[[Any, Any], Any]] = {}
    _peft_mergers: dict[type, Callable[..., Any]] = {}
    _peft_loaders: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register_peft(cls, config_type: type) -> Any:
        """
        Register an implementation for a PEFT configuration type.

        Args:
            config_type: Configuration class used to select the implementation.

        Returns:
            A decorator that registers a ``(model, config)`` callable.

        Raises:
            ValueError: If the configuration type already has a different
                registered implementation.
        """

        def decorator(implementation: Any) -> Any:
            registered = cls._peft_methods.get(config_type)
            if registered is not None and registered is not implementation:
                raise ValueError(
                    f'{config_type.__name__} already has a registered '
                    'PEFT implementation'
                )
            cls._peft_methods[config_type] = implementation
            return implementation

        return decorator

    @classmethod
    def register_peft_merger(cls, module_type: type) -> Any:
        """Register the merge implementation for a PEFT wrapper module."""

        def decorator(implementation: Any) -> Any:
            registered = cls._peft_mergers.get(module_type)
            if registered is not None and registered is not implementation:
                raise ValueError(
                    f'{module_type.__name__} already has a registered '
                    'PEFT merger'
                )
            cls._peft_mergers[module_type] = implementation
            return implementation

        return decorator

    @classmethod
    def register_peft_loader(cls, peft_type: str) -> Any:
        """Register a checkpoint loader for a serialized PEFT type."""
        normalized_type = peft_type.upper()

        def decorator(implementation: Any) -> Any:
            registered = cls._peft_loaders.get(normalized_type)
            if registered is not None and registered is not implementation:
                raise ValueError(
                    f'{normalized_type} already has a registered PEFT loader'
                )
            cls._peft_loaders[normalized_type] = implementation
            return implementation

        return decorator

    @classmethod
    def apply_peft(cls, model: M, config: Any) -> M:
        """
        Apply a PEFT configuration to a model in place.

        The registered implementation may replace modules inside ``model``.
        The same model instance is returned for convenient assignment.

        Args:
            model: Existing model instance to transform.
            config: Registered PEFT configuration instance.

        Returns:
            The transformed model.

        Raises:
            NotImplementedError: If no implementation is registered for the
                supplied configuration type.
        """
        implementation = cls._peft_methods.get(type(config))
        if implementation is None:
            implementation = next(
                (
                    candidate
                    for config_type, candidate in cls._peft_methods.items()
                    if isinstance(config, config_type)
                ),
                None,
            )
        if implementation is None:
            raise NotImplementedError(
                'Unsupported PEFT configuration: '
                f'{type(config).__name__}'
            )
        return implementation(model, config)

    @classmethod
    def load_peft(
        cls,
        model: M,
        path_or_repo: PathLike,
        *,
        local: bool | None = None,
        token: str | bool | None = None,
        revision: str | None = None,
        subfolder: PathLike | None = None,
        rngs: Rngs | None = None,
    ) -> M:
        """
        Load a saved PEFT adapter into an existing base model.

        Local directories are detected automatically. Hub repositories support
        revisions, private-repository tokens, subfolders, and sharded adapter
        Safetensors indexes.

        Args:
            model: Existing base model or a model with matching PEFT wrappers.
            path_or_repo: Local adapter directory or Hub repository ID.
            local: Override automatic local-directory detection.
            token: Hugging Face authentication token or token-selection flag.
            revision: Optional Hub branch, tag, or commit.
            subfolder: Optional adapter subdirectory.
            rngs: Optional RNG collection used when adapter wrappers must be
                created before loading their saved values.

        Returns:
            The same model instance with loaded adapter parameters.
        """
        if not isinstance(model, Module):
            raise TypeError(
                'PEFT loading currently requires a Taktiny nn.Module model'
            )

        source = os.fspath(path_or_repo)
        if local is None:
            local = os.path.isdir(source)

        def local_path(filename: str) -> str:
            parts = [source]
            if subfolder:
                parts.append(os.fspath(subfolder))
            parts.append(filename)
            return os.path.join(*parts)

        def resolve(filename: str) -> str:
            if local:
                resolved = local_path(filename)
                if not os.path.isfile(resolved):
                    raise FileNotFoundError(
                        f'Adapter file was not found: {resolved}'
                    )
                return resolved
            return hf_hub_download(
                repo_id=source,
                filename=filename,
                subfolder=subfolder,
                revision=revision,
                token=token,
            )

        config_path = resolve('adapter_config.json')
        with open(config_path) as config_file:
            adapter_config = json.load(config_file)

        peft_type = str(adapter_config.get('peft_type', '')).upper()
        implementation = cls._peft_loaders.get(peft_type)
        if implementation is None:
            raise NotImplementedError(
                f'Unsupported saved PEFT type: {peft_type or "<missing>"}'
            )

        index_filename = 'adapter_model.safetensors.index.json'
        index_path = None
        if local:
            candidate = local_path(index_filename)
            if os.path.isfile(candidate):
                index_path = candidate
        else:
            try:
                index_path = resolve(index_filename)
            except EntryNotFoundError:
                pass

        if index_path is None:
            adapter_files = [resolve('adapter_model.safetensors')]
        else:
            with open(index_path) as index_file:
                index = json.load(index_file)
            weight_map = index.get('weight_map')
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    'Adapter Safetensors index has no weight_map'
                )
            filenames = dict.fromkeys(weight_map.values())
            adapter_files = [
                resolve(filename)
                for filename in filenames
            ]

        adapter_state = {}
        for adapter_file in adapter_files:
            with safe_open(
                adapter_file,
                framework='np',
                device='cpu',
            ) as checkpoint:
                for name in checkpoint.keys():
                    if name in adapter_state:
                        raise ValueError(
                            f'Duplicate adapter tensor in checkpoint: {name}'
                        )
                    adapter_state[name] = checkpoint.get_tensor(name)

        return implementation(
            model,
            adapter_config,
            adapter_state,
            rngs=rngs,
        )

    @classmethod
    def merge_peft(
        cls,
        model: M,
        *,
        dtype: DType | str | None = None,
        quant: Any = None,
    ) -> M:
        """
        Merge PEFT adapter weights into their base modules in place.

        Adapter calculations are performed in float32. ``dtype`` controls the
        merged dense-weight dtype, while ``quant`` optionally requantizes the
        merged weights using Taktiny's Qwix quantization rules.

        Args:
            model: Existing Taktiny model containing PEFT wrapper modules.
            dtype: Optional floating-point dtype for merged dense weights.
                Dense base weights retain their dtype when omitted. Quantized
                weights use their dequantized dtype.
            quant: Optional Qwix quantization rule, provider, or qtype string
                applied after merging.

        Returns:
            The same model instance with adapters merged and removed.

        Raises:
            TypeError: If ``model`` is not a Taktiny module.
            ValueError: If no registered mergeable PEFT modules are found.
        """
        if not isinstance(model, Module):
            raise TypeError(
                'PEFT merging currently requires a Taktiny nn.Module model'
            )

        merged = []

        def merger_for(module: Module) -> Callable[..., Module] | None:
            implementation = cls._peft_mergers.get(type(module))
            if implementation is not None:
                return implementation
            return next(
                (
                    candidate
                    for module_type, candidate in cls._peft_mergers.items()
                    if isinstance(module, module_type)
                ),
                None,
            )

        def transform(module: Module, prefix: str = '') -> None:
            for name, child in list(iter_children(module)):
                full_name = f'{prefix}.{name}' if prefix else name
                implementation = merger_for(child)
                if implementation is not None:
                    replacement = implementation(
                        child,
                        dtype=dtype,
                        quant=quant,
                        module_path=full_name,
                    )
                    _replace_child(module, name, replacement)
                    merged.append(full_name)
                elif isinstance(child, Module):
                    transform(child, full_name)

        transform(model)
        if not merged:
            raise ValueError('No mergeable PEFT modules were found in model')

        trainable_state = getattr(
            model,
            '_peft_trainable_state',
            None,
        )
        if trainable_state is not None:
            for name, parameter in model.flat_parameter_dict().items():
                if name in trainable_state:
                    parameter.trainable = trainable_state[name]
            delattr(model, '_peft_trainable_state')

        if hasattr(model, 'peft_config'):
            delattr(model, 'peft_config')
        return model


__all__ = ['Takt']
