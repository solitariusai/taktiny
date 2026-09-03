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

from typing import Any, TypeVar

from taktiny.nn.base import Module, iter_children
from taktiny.takt.adapter.base import BaseAdapter

M = TypeVar('M', bound=Module)


def _replace_child(parent: Module, name: str, child: Module) -> None:
    """Replace a direct child, including list and tuple-backed children."""
    if name.isdigit() and hasattr(parent, 'layers'):
        position = int(name)
        if isinstance(parent.layers, tuple):
            layers = list(parent.layers)
            layers[position] = child
            parent.layers = tuple(layers)
        else:
            parent.layers[position] = child
        return

    if '.' in name:
        attribute, index = name.rsplit('.', 1)
        sequence = getattr(parent, attribute)
        position = int(index)
        if isinstance(sequence, tuple):
            items = list(sequence)
            items[position] = child
            setattr(parent, attribute, tuple(items))
        else:
            sequence[position] = child
        return

    setattr(parent, name, child)


class Takt:
    """Apply adapter objects to existing Taktiny models."""

    @classmethod
    def apply_adapter(cls, model: M, adapter: BaseAdapter) -> M:
        """Inject ``adapter`` into all of its matching modules in ``model``.

        Existing model parameters are frozen. Parameters created by adapter
        modules remain trainable, so the trainer only needs to honor the
        normal ``Parameter.trainable`` flag. Adapter objects are retained as
        static model metadata for optional post-step updates.
        """
        if not isinstance(model, Module):
            raise TypeError('Adapters require a Taktiny nn.Module model')
        if not isinstance(adapter, BaseAdapter):
            raise TypeError('adapter must be a BaseAdapter instance')

        targets: list[tuple[Module, str, Module, str]] = []

        def collect(module: Module, prefix: str = '') -> None:
            for name, child in iter_children(module):
                if not isinstance(child, Module):
                    continue
                module_path = f'{prefix}.{name}' if prefix else name
                if adapter.matches(module_path):
                    if isinstance(child, adapter._adapter):
                        raise ValueError(
                            f'{type(adapter).__name__} is already applied '
                            f'to {module_path}'
                        )
                    targets.append((module, name, child, module_path))
                else:
                    collect(child, module_path)

        collect(model)
        if not targets:
            patterns = ', '.join(adapter.targets)
            raise ValueError(
                f'No modules matched the adapter target patterns: {patterns}'
            )

        existing_parameters = {
            id(parameter)
            for parameter in model.flat_parameter_dict().values()
        }
        adapter.prepare([
            (module_path, target)
            for _, _, target, module_path in targets
        ])
        replacements = [
            (
                parent,
                name,
                adapter.build(target, module_path=module_path),
            )
            for parent, name, target, module_path in targets
        ]
        for parent, name, replacement in replacements:
            _replace_child(parent, name, replacement)

        for parameter in model.flat_parameter_dict().values():
            if id(parameter) in existing_parameters:
                parameter.trainable = False

        adapters = tuple(getattr(model, '_takt_adapters', ()))
        model._takt_adapters = adapters + (adapter,)
        return model

    @classmethod
    def update_adapters(cls, model: Module, **kwargs: Any) -> tuple[Any, ...]:
        """Run each applied adapter's optional post-step update hook.

        This is independent of any trainer. A trainer may call this method
        after its normal optimizer update and provide the context required by
        an adapter such as AdaLoRA.
        """
        del cls
        return tuple(
            adapter(**kwargs)
            for adapter in getattr(model, '_takt_adapters', ())
        )
