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
"""Common model contracts for composed architectures."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import typing as tp

from taktiny import nn
from taktiny.maestro.config import ModelConfig
from taktiny.utils.typing import PathLike


ComponentSpec: tp.TypeAlias = (
    type[nn.Module] | tuple[type[nn.Module], str | None]
)


class DiffusionDenoiser(nn.Module):
    """Base class for composed, array-to-array diffusion models.

    Concrete classes declare ``component_map`` to describe which TakTiny model
    class owns each repository subfolder. :meth:`from_pretrained` then loads
    every component independently and assembles the complete trainable module.
    Forward methods accept prepared arrays; string and media preprocessing are
    outside this model boundary.

    Example::

        class Sana(DiffusionDenoiser):
            component_map = {
                'text_encoder': (Gemma2, 'text_encoder'),
                'transformer': (SanaTransformer, 'transformer'),
                'vae': (AutoencoderDC, 'vae'),
            }

    A component declared as ``{'vae': AutoencoderDC}`` uses its component name
    as the subfolder automatically.
    """

    component_map: tp.ClassVar[Mapping[str, ComponentSpec]] = {}

    def __init__(self, **components: nn.Module) -> None:
        self.bind(**components)

    @classmethod
    def _component_specs(
        cls,
    ) -> dict[str, tuple[type[nn.Module], str | None]]:
        specs: dict[str, tuple[type[nn.Module], str | None]] = {}
        for name, spec in cls.component_map.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    'component_map keys must be non-empty strings'
                )
            if isinstance(spec, tuple):
                if len(spec) != 2:
                    raise ValueError(
                        f'component_map[{name!r}] must contain '
                        '(module_type, subfolder)'
                    )
                module_type, subfolder = spec
            else:
                module_type, subfolder = spec, name
            if (
                not isinstance(module_type, type)
                or not issubclass(module_type, nn.Module)
            ):
                raise TypeError(
                    f'component_map[{name!r}] must contain an nn.Module class'
                )
            if subfolder is not None and (
                not isinstance(subfolder, str) or not subfolder
            ):
                raise ValueError(
                    f'the {name!r} subfolder must be a non-empty string or None'
                )
            specs[name] = (module_type, subfolder)
        return specs

    def bind(self, **modules: nn.Module) -> tp.Self:
        """Attach the initialized components declared by ``component_map``."""
        specs = self._component_specs()
        if specs:
            missing = specs.keys() - modules.keys()
            unexpected = modules.keys() - specs.keys()
            if missing or unexpected:
                details = []
                if missing:
                    details.append(f'missing: {", ".join(sorted(missing))}')
                if unexpected:
                    details.append(
                        f'unexpected: {", ".join(sorted(unexpected))}'
                    )
                raise ValueError(
                    'components do not match component_map ('
                    + '; '.join(details)
                    + ')'
                )
        for name, module in modules.items():
            if not isinstance(name, str) or not name:
                raise ValueError('module names must be non-empty strings')
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f'{name} must be an initialized nn.Module, got '
                    f'{type(module).__name__}'
                )
            setattr(self, name, module)
        self.component_names = tuple(modules)
        return self

    @staticmethod
    def _accepts_init_keyword(
        module_type: type[nn.Module],
        keyword: str,
    ) -> bool:
        parameters = inspect.signature(module_type.__init__).parameters
        return keyword in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        *,
        local: bool = False,
        components: Mapping[str, nn.Module] | None = None,
        component_kwargs: Mapping[str, Mapping[str, tp.Any]] | None = None,
        **load_kwargs: tp.Any,
    ) -> tp.Self:
        """Load registered components from their repository subfolders.

        ``load_kwargs`` are shared by all component loaders. Per-component
        options in ``component_kwargs`` take precedence. ``components`` can
        supply already initialized modules, which are not loaded again.
        """
        specs = cls._component_specs()
        if not specs:
            raise ValueError(
                f'{cls.__name__}.component_map must declare at least one '
                'loadable component'
            )

        supplied = dict(components or {})
        options_by_name = dict(component_kwargs or {})
        unknown = (supplied.keys() | options_by_name.keys()) - specs.keys()
        if unknown:
            raise ValueError(
                'unknown diffusion components: '
                + ', '.join(sorted(unknown))
            )

        loaded: dict[str, nn.Module] = {}
        for name, (module_type, subfolder) in specs.items():
            if name in supplied:
                module = supplied[name]
                if not isinstance(module, module_type):
                    raise TypeError(
                        f'components[{name!r}] must be a '
                        f'{module_type.__name__}, got {type(module).__name__}'
                    )
                loaded[name] = module
                continue

            options = dict(load_kwargs)
            if (
                'use_list' in options
                and not cls._accepts_init_keyword(module_type, 'use_list')
            ):
                options.pop('use_list')
            per_component = options_by_name.get(name, {})
            if not isinstance(per_component, Mapping):
                raise TypeError(
                    f'component_kwargs[{name!r}] must be a mapping'
                )
            options.update(per_component)
            component_subfolder = options.pop('subfolder', subfolder)
            config = options.pop('config', None)
            if config is None:
                config = ModelConfig.load_config(
                    path_or_repo,
                    subfolder=component_subfolder,
                    local=local,
                )
            if config is None:
                raise ValueError(
                    f'unable to load config for component {name!r} from '
                    f'{path_or_repo!s}'
                )

            loader = getattr(module_type, 'from_pretrained', None)
            if not callable(loader):
                raise TypeError(
                    f'{module_type.__name__} does not implement '
                    'from_pretrained'
                )
            module = loader(
                path_or_repo,
                config=config,
                local=local,
                subfolder=component_subfolder,
                **options,
            )
            if not isinstance(module, module_type):
                raise TypeError(
                    f'{module_type.__name__}.from_pretrained returned '
                    f'{type(module).__name__}'
                )
            loaded[name] = module

        return cls(**loaded)


__all__ = ['ComponentSpec', 'DiffusionDenoiser']
