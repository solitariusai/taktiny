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
"""
Public API for retrieving registered architectures from the Hugging Face Hub
if implemented in this library.
"""

from __future__ import annotations
from collections.abc import Mapping
import typing as tp
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from taktiny.maestro._livret import repertoire
from taktiny.maestro.config import ModelConfig
from taktiny.nn import Rngs
from taktiny.nn.module import Module
from taktiny.utils.typing import DType, LogicalRules, PathLike


class Maestro:
    """
    Registry-backed entry point for pretrained Taktiny models.

    Maestro reads the architecture declared by a Hugging Face configuration,
    resolves its registered Taktiny implementation, and delegates model
    construction and checkpoint loading to that class. It also provides
    helpers for inspecting the architecture registry and constructing abstract
    models without allocating parameter buffers.

    ``from_pretrained`` materializes checkpoint weights and accepts model
    loading options such as ``dtype``, Qwix ``quant`` rules, device meshes, and
    logical sharding rules. ``eval_shape`` only constructs the model through
    ``jax.eval_shape``; it does not download or materialize checkpoint weights.

    A mesh may be supplied directly as ``jax.sharding.Mesh`` or as a mapping
    from mesh-axis names to sizes. When no sharding rules are provided, the
    selected architecture's defaults are used.

    Example:
        >>> model = Maestro.from_pretrained(
        ...     "Qwen/Qwen2.5-0.5B",
        ...     dtype="bfloat16",
        ... )
        >>> abstract_model = Maestro.eval_shape(
        ...     "Qwen/Qwen2.5-0.5B"
        ... )
    """

    @classmethod
    def list(cls) -> set[type[Module]]:
        """
        Return the distinct model implementation classes in the registry.

        Multiple architecture names may resolve to the same implementation, so
        each class appears only once.

        Returns:
            A set of registered Taktiny model classes.
        """
        return repertoire.available_classes()

    @classmethod
    def available(cls) -> list[str]:
        """
        Return all registered Hugging Face architecture names.

        The returned strings are the values expected in the ``architectures``
        field of a Hugging Face model configuration.

        Returns:
            A list of registered architecture names.
        """
        return repertoire.available()

    @classmethod
    def is_supported(cls, model_class: str) -> bool:
        """
        Check whether an architecture name is registered.

        Args:
            model_class: Hugging Face architecture name, such as
                ``"LlamaForCausalLM"``.

        Returns:
            ``True`` when the architecture can be resolved by Maestro.
        """
        return True if model_class in repertoire.available() else False

    @classmethod
    def _get_architecture_class(
        cls,
        repo_or_path: PathLike,
        config: ModelConfig | tp.Dict | None,
        config_filename: str,
        local: bool,
        subfolder: str | None = None,
    ):
        if isinstance(config, dict):
            config = ModelConfig(**config)
        if config is None:
            config = ModelConfig.load_config(
                repo_or_path,
                config_filename,
                subfolder=subfolder,
                local=local,
            )
        if config is None:
            raise ValueError(f'Unable to load config from {repo_or_path}')

        architectures = getattr(config, 'architectures', None) or []
        if not architectures:
            class_name = getattr(config, '_class_name', None)
            architectures = [class_name] if class_name else []
        if len(architectures) != 1:
            raise ValueError(
                'Expected config.architectures to contain exactly one architecture'
            )

        architecture = architectures[0]
        if not repertoire.get(architecture):
            raise NotImplementedError(
                f'Unsupported architecture: {architecture}'
            )
        model_cls = repertoire.get(architecture)
        return model_cls, config

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: PathLike,
        mesh: Mesh | Mapping[str, int] | None = None,
        sharding_rules: LogicalRules | None = None,
        local: bool = False,
        dtype: DType | str | None = None,
        quant: tp.Any = None,
        use_list: bool = False,
        **kwargs: tp.Any
    ) -> Module:
        """
        Load a registered model and materialize its checkpoint weights.

        The architecture is selected from the model configuration's
        ``architectures`` field. Checkpoint loading, dtype conversion, Qwix
        quantization, and parameter placement are delegated to the selected
        Taktiny model class.

        Args:
            repo_or_path: Hugging Face repository identifier or local
                checkpoint directory.
            mesh: Optional ``jax.sharding.Mesh`` or mapping from mesh-axis names
                to device counts.
            sharding_rules: Optional logical-to-mesh axis mapping rules. The
                architecture defaults are used when omitted.
            local: Whether ``repo_or_path`` refers to a local checkpoint.
            dtype: Model dtype or uniform Qwix PTQ shortcut. Floating-point
                values select the model parameter dtype; ``"fp8"``,
                ``"int8"``, ``"int4"``, and ``"nf4"`` quantize supported
                parameters while loading.
            quant: Optional Qwix qtype string, quantization rule, PTQ provider,
                or sequence of rules for selective weight quantization. When
                combined with a quantized ``dtype``, explicit rules take
                precedence and the dtype becomes the fallback for unmatched
                modules.
            **kwargs: Additional loading options forwarded to the selected
                model, including ``subfolder`` and ``rngs``.

        Returns:
            A materialized instance of the registered Taktiny model.

        Raises:
            AssertionError: If the configuration does not declare exactly one
                architecture.
            NotImplementedError: If the declared architecture is not
                registered.
        """
        kwargs = dict(kwargs)
        config_filename = kwargs.pop('config_filename', 'config.json')
        config = kwargs.pop('config', None)
        model_cls, config = Maestro._get_architecture_class(
            repo_or_path,
            config,
            config_filename,
            local,
            kwargs.get('subfolder'),
        )

        # Parse Mesh if provided as a dict (e.g. {'data': 4, 'model': 2})
        if isinstance(mesh, dict):
            axis_names = tuple(mesh.keys())
            shape = tuple(mesh.values())
            devices = mesh_utils.create_device_mesh(shape)
            mesh = Mesh(devices, axis_names)

        if sharding_rules is None and hasattr(model_cls, 'default_sharding_rules'):
            sharding_rules = model_cls.default_sharding_rules

        return model_cls.from_pretrained(
            repo_or_path,
            mesh=mesh,
            sharding_rules=sharding_rules,
            local=local,
            dtype=dtype,
            quant=quant,
            use_list=use_list,
            config=config,
            **kwargs
        )

    @classmethod
    def eval_shape(
        cls,
        repo_or_path: PathLike,
        *,
        mesh: Mesh | Mapping[str, int] | None = None,
        sharding_rules: LogicalRules | None = None,
        local: bool = False,
        use_list: bool = False,
        **kwargs: tp.Any,
    ) -> Module:
        """
        Construct an abstract registered model without loading its weights.

        This resolves the architecture in the same way as
        ``from_pretrained`` but invokes its constructor under
        ``jax.eval_shape``. Configuration data may be read from disk or
        downloaded, while checkpoint tensors are neither downloaded nor
        materialized.

        Args:
            repo_or_path: Hugging Face repository identifier or local model
                directory containing ``config.json``.
            mesh: Optional ``jax.sharding.Mesh`` or mapping from mesh-axis names
                to device counts.
            sharding_rules: Optional logical-to-mesh axis mapping rules. The
                architecture defaults are used when omitted.
            local: Whether ``repo_or_path`` refers to a local model directory.
            **kwargs: Additional constructor options. ``config`` may supply an
                existing ``ModelConfig`` and ``rngs`` may supply the abstract
                initialization RNG.

        Returns:
            An abstract model whose array leaves are
            ``jax.ShapeDtypeStruct`` values.

        Raises:
            ValueError: If configuration loading fails or it does not declare
                exactly one architecture.
            NotImplementedError: If the declared architecture is not
                registered.
        """
        kwargs = dict(kwargs)
        config_filename = kwargs.pop('config_filename', 'config.json')
        config = kwargs.pop('config', None)
        model_cls, config = Maestro._get_architecture_class(
            repo_or_path,
            config,
            config_filename,
            local,
            kwargs.get('subfolder'),
        )

        if isinstance(mesh, dict):
            axis_names = tuple(mesh)
            shape = tuple(mesh.values())
            devices = mesh_utils.create_device_mesh(shape)
            mesh = Mesh(devices, axis_names)

        if sharding_rules is None and hasattr(model_cls, 'default_sharding_rules'):
            sharding_rules = model_cls.default_sharding_rules

        rngs = kwargs.pop('rngs', None)
        if rngs is None:
            rngs = Rngs(0)

        return jax.eval_shape(
            lambda: model_cls(
                config,
                rngs=rngs,
                mesh=mesh,
                sharding_rules=sharding_rules,
                use_list=use_list,
                **kwargs,
            )
        )


__all__ = ['Maestro']
