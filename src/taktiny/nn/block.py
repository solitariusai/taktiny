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
"""Utilities modules for stack/group other modules"""
from __future__ import annotations
from collections.abc import (
    Callable,
    Iterable,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from typing import Any, overload
import jax
import jax.numpy as jnp
from taktiny import transforms as tt
from taktiny.nn.module import Module
from taktiny.utils.typing import PyTree

def _stack_modules(modules: Iterable[Module]) -> tuple[Module, int]:
    modules = tuple(modules)
    if not modules:
        raise ValueError('modules must contain at least one Module')

    for index, module in enumerate(modules):
        if not isinstance(module, Module):
            raise TypeError(
                f'modules[{index}] must be a Module, got '
                f'{type(module).__name__}'
            )

    reference = modules[0]
    structure = jax.tree_util.tree_structure(reference)
    reference_leaves = jax.tree_util.tree_leaves(reference)
    for index, module in enumerate(modules[1:], start=1):
        if type(module) is not type(reference):
            raise ValueError(
                'all modules must have the same type; '
                f'modules[0] is {type(reference).__name__} and '
                f'modules[{index}] is {type(module).__name__}'
            )
        if jax.tree_util.tree_structure(module) != structure:
            raise ValueError(
                'all modules must have the same PyTree structure and static '
                'configuration; '
                f'modules[0] and modules[{index}] differ'
            )
        for leaf_index, (reference_leaf, leaf) in enumerate(
            zip(reference_leaves, jax.tree_util.tree_leaves(module))
        ):
            reference_shape = getattr(reference_leaf, 'shape', None)
            shape = getattr(leaf, 'shape', None)
            if shape != reference_shape:
                raise ValueError(
                    'all corresponding module leaves must have the same '
                    f'shape; leaf {leaf_index} in modules[0] has shape '
                    f'{reference_shape} and modules[{index}] has shape {shape}'
                )
            reference_dtype = getattr(reference_leaf, 'dtype', None)
            dtype = getattr(leaf, 'dtype', None)
            if dtype != reference_dtype:
                raise ValueError(
                    'all corresponding module leaves must have the same '
                    f'dtype; leaf {leaf_index} in modules[0] has dtype '
                    f'{reference_dtype} and modules[{index}] has dtype {dtype}'
                )

    stacked = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        *modules,
    )

    for parameter in stacked.flat_parameter_dict().values():
        if hasattr(parameter, 'axis_names') and parameter.axis_names is not None:
            parameter.axis_names = (None,) + tuple(parameter.axis_names)
        if hasattr(parameter, 'quantization_batch_axis_count'):
            parameter.quantization_batch_axis_count += 1

    return stacked, len(modules)


def _stack_compatible(reference: Module, module: Module) -> bool:
    if type(module) is not type(reference):
        return False
    if jax.tree_util.tree_structure(module) != jax.tree_util.tree_structure(
        reference
    ):
        return False

    reference_leaves = jax.tree_util.tree_leaves(reference)
    leaves = jax.tree_util.tree_leaves(module)
    if len(reference_leaves) != len(leaves):
        return False
    for reference_leaf, leaf in zip(reference_leaves, leaves):
        if getattr(reference_leaf, 'shape', None) != getattr(
            leaf,
            'shape',
            None,
        ):
            return False
        if getattr(reference_leaf, 'dtype', None) != getattr(
            leaf,
            'dtype',
            None,
        ):
            return False
    return True


def _group_stack_compatible(
    modules: Sequence[Module],
) -> tuple[tuple[Module, ...], ...]:
    groups: list[list[Module]] = []
    for module in modules:
        if not groups or not _stack_compatible(groups[-1][0], module):
            groups.append([module])
        else:
            groups[-1].append(module)
    return tuple(tuple(group) for group in groups)


def _validate_module_sequence(modules: Sequence[Module]) -> None:
    if not isinstance(modules, Sequence):
        raise TypeError(
            f'modules must be a sequence, got {type(modules).__name__}'
        )
    for index, module in enumerate(modules):
        if not isinstance(module, Module):
            raise TypeError(
                f'modules[{index}] must be a Module, got '
                f'{type(module).__name__}'
            )

class List(Module):
    def __init__(self, modules: Sequence[Module]) -> None:
        _validate_module_sequence(modules)
        self.layers = list(modules)

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @overload
    def __getitem__(self, idx: slice) -> List: ...

    def __getitem__(self, idx: int | slice) -> Module | List:
        if isinstance(idx, slice):
            return List(self.layers[idx])
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def extra_repr(self) -> str:
        return f"{len(self.layers)}"

class Dict(Module):
    """A module container indexed by stable string keys."""

    def __init__(self, modules: Mapping[str, Module]) -> None:
        if not isinstance(modules, Mapping):
            raise TypeError(
                f'modules must be a mapping, got {type(modules).__name__}'
            )
        for key, module in modules.items():
            if not isinstance(key, str):
                raise TypeError(
                    f'module keys must be strings, got {type(key).__name__}'
                )
            if not key:
                raise ValueError('module keys must not be empty')
            if '.' in key:
                raise ValueError("module keys must not contain '.'")
            if not isinstance(module, Module):
                raise TypeError(
                    f'modules[{key!r}] must be a Module, got '
                    f'{type(module).__name__}'
                )
        self.layers = dict(modules)

    def __getitem__(self, key: str) -> Module:
        return self.layers[key]

    def __contains__(self, key: object) -> bool:
        return key in self.layers

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[str]:
        return iter(self.layers)

    def keys(self) -> KeysView[str]:
        return self.layers.keys()

    def values(self) -> ValuesView[Module]:
        return self.layers.values()

    def items(self) -> ItemsView[str, Module]:
        return self.layers.items()

    def extra_repr(self) -> str:
        return f'{len(self.layers)}'

class Sequential(Module):
    def __init__(self, modules: Sequence[Module]) -> None:
        _validate_module_sequence(modules)
        self.layers = tuple(modules)

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequential: ...

    def __getitem__(self, idx: int | slice) -> Module | Sequential:
        if isinstance(idx, slice):
            return Sequential(self.layers[idx])
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def __call__(self, x: PyTree, *args: Any, **kwargs: Any) -> PyTree:
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

    def extra_repr(self) -> str:
        return f"{len(self.layers)}"

class SeqStack(Module):
    def __init__(
        self,
        modules: Iterable[Module],
        *,
        reverse: bool = False,
        unroll: int | bool = 1,
        split_transpose: bool = False,
    ) -> None:
        if not isinstance(reverse, bool):
            raise TypeError('reverse must be a boolean')
        if not isinstance(unroll, (int, bool)) or (
            isinstance(unroll, int)
            and not isinstance(unroll, bool)
            and unroll <= 0
        ):
            raise ValueError('unroll must be a positive integer or boolean')
        if not isinstance(split_transpose, bool):
            raise TypeError('split_transpose must be a boolean')
        modules = tuple(modules)
        _validate_module_sequence(modules)
        if not modules:
            raise ValueError('modules must contain at least one Module')

        groups = _group_stack_compatible(modules)
        self.num_stack = len(modules)
        self.reverse = reverse
        self.unroll = unroll
        self.split_transpose = split_transpose
        if len(groups) == 1:
            self.stacked, _ = _stack_modules(groups[0])
            self.group_sizes = (self.num_stack,)
        else:
            self.groups = [
                SeqStack(
                    group,
                    reverse=reverse,
                    unroll=unroll,
                    split_transpose=split_transpose,
                )
                for group in groups
            ]
            self.group_sizes = tuple(len(group) for group in groups)

    def __len__(self) -> int:
        return self.num_stack

    def __call__(
        self,
        f: Callable[..., Any],
        carry: PyTree,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[PyTree, PyTree]:
        if hasattr(self, 'groups'):
            groups = reversed(self.groups) if self.reverse else self.groups
            group_outputs = []
            for group in groups:
                carry, outputs = group(f, carry, *args, **kwargs)
                group_outputs.append(outputs)

            if all(outputs is None for outputs in group_outputs):
                return carry, None
            if any(outputs is None for outputs in group_outputs):
                raise ValueError(
                    'all SeqStack groups must return compatible outputs'
                )
            reference_structure = jax.tree_util.tree_structure(
                group_outputs[0]
            )
            if any(
                jax.tree_util.tree_structure(outputs) != reference_structure
                for outputs in group_outputs[1:]
            ):
                raise ValueError(
                    'all SeqStack groups must return compatible outputs'
                )
            if self.reverse:
                group_outputs.reverse()
            return carry, jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0),
                *group_outputs,
            )

        @tt.scan(
            length=self.num_stack,
            reverse=self.reverse,
            unroll=self.unroll,
            _split_transpose=self.split_transpose,
        )
        def apply_fn(carry: Any, layer: Any, *broadcast_args: Any) -> Any:
            return f(layer, carry, *broadcast_args, **kwargs)

        return apply_fn(carry, self.stacked, *args)

    def extra_repr(self) -> str:
        if hasattr(self, 'groups'):
            groups = ', '.join(map(str, self.group_sizes))
            return f'{self.num_stack}, groups=({groups})'
        return f'{self.num_stack}'

class Stack(Module):
    def __init__(
        self,
        modules: Iterable[Module],
        *,
        axis_name: Any | None = None,
        spmd_axis_name: Any | tuple[Any, ...] | None = None,
    ) -> None:
        self.stacked, self.num_stack = _stack_modules(modules)
        self.axis_name = axis_name
        self.spmd_axis_name = spmd_axis_name

    def __len__(self) -> int:
        return self.num_stack

    def __call__(
        self,
        *args: Any,
        in_axes: int | None | tuple[int | None, ...] = 0,
        out_axes: Any = 0,
        **kwargs: Any,
    ) -> PyTree:
        if isinstance(in_axes, tuple):
            if len(in_axes) != len(args):
                raise ValueError(
                    'tuple in_axes must have one entry per positional argument'
                )
            vmap_in_axes = (0,) + in_axes
        else:
            vmap_in_axes = (0,) + (in_axes,) * len(args)

        @tt.vmap(
            in_axes=vmap_in_axes,
            out_axes=out_axes,
            axis_name=self.axis_name,
            axis_size=self.num_stack,
            spmd_axis_name=self.spmd_axis_name,
        )
        def apply_fn(layer: Any, *positional_args: Any) -> Any:
            return layer(*positional_args, **kwargs)

        return apply_fn(self.stacked, *args)

    def extra_repr(self) -> str:
        return f"{self.num_stack}"

__all__ = [
    'List',
    'Dict',
    'Sequential',
    'SeqStack',
    'Stack',
]
