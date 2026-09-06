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
"""Module containers and state-preserving scanned/vectorized layer stacks."""
from __future__ import annotations

import operator
from collections.abc import (
    Callable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from typing import Any, overload

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

import taktiny.utils.transforms as tt
from taktiny.nn.base import Module, Parameter
from taktiny.utils.typing import PyTree


def _update_output_axes(output: PyTree, out_axes: PyTree) -> None:
    """Keep module-output metadata handling local to module containers."""
    seen: set[int] = set()

    def update(axes: Any, subtree: Any) -> None:
        if axes is None:
            return
        axis = operator.index(axes)

        def visit(value: Any) -> Any:
            if isinstance(value, Module):
                parameters = ([value] if isinstance(value, Parameter)
                              else value.flat_parameter_dict().values())
                for parameter in parameters:
                    if id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    ndim = parameter.ndim
                    mapped_axis = axis if axis >= 0 else axis + ndim
                    names = parameter.axis_names
                    if names is not None and len(names) == ndim - 1:
                        names = tuple(names)
                        parameter.axis_names = (
                            names[:mapped_axis] + (None,) + names[mapped_axis:]
                        )
                    spec = parameter.partition_spec
                    if spec is not None and len(spec) < ndim:
                        padded = tuple(spec) + (None,) * (ndim - 1 - len(spec))
                        parameter.partition_spec = PartitionSpec(*(
                            padded[:mapped_axis] + (None,) + padded[mapped_axis:]
                        ))
            return value

        jax.tree.map(visit, subtree, is_leaf=lambda value: isinstance(value, Module))

    jax.tree.map(update, out_axes, output, is_leaf=lambda value: value is None)


def _stack_mismatch(reference: Module, module: Module) -> str | None:
    """Describe incompatibility once for both grouping and validation."""
    if type(module) is not type(reference):
        return 'all modules must have the same type'
    reference_leaves, structure = jax.tree.flatten(reference)
    leaves, other_structure = jax.tree.flatten(module)
    if structure != other_structure:
        return 'all modules must have the same PyTree structure and static configuration'
    for index, (left, right) in enumerate(zip(reference_leaves, leaves)):
        for attribute in ('shape', 'dtype'):
            expected = getattr(left, attribute, None)
            actual = getattr(right, attribute, None)
            if expected != actual:
                return (f'all corresponding module leaves must have the same {attribute}; '
                        f'leaf {index} has {expected} versus {actual}')
    return None


def _shift_stack_axes(module: Module, *, add: bool) -> None:
    """Align parameter metadata with insertion/removal of the layer axis."""
    seen: set[int] = set()
    for parameter in module.flat_parameter_dict().values():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.axis_names is not None:
            names = tuple(parameter.axis_names)
            parameter.axis_names = (None,) + names if add else names[1:]
        if parameter.partition_spec is not None:
            spec = tuple(parameter.partition_spec)
            parameter.partition_spec = PartitionSpec(
                *((None,) + spec if add else spec[1:])
            )


def _call_stacked(
    layer: Module, function: Callable[..., Any], *args: Any, **kwargs: Any,
) -> tuple[Any, Module, bool]:
    """Call a sliced layer and detect dynamic updates without retaining tracers."""
    before, structure = jax.tree.flatten(layer)
    _shift_stack_axes(layer, add=False)
    output = function(layer, *args, **kwargs)
    # Keep output references to the sliced layer separate from state metadata.
    layer = jax.tree.map(lambda value: value, layer)
    _shift_stack_axes(layer, add=True)
    after, updated_structure = jax.tree.flatten(layer)
    if structure != updated_structure:
        raise ValueError('stacked calls must preserve module structure and static configuration')
    if any(left.shape != right.shape or left.dtype != right.dtype
           for left, right in zip(before, after)):
        raise ValueError('stacked calls must preserve state leaf shapes and dtypes')
    changed = any(left is not right for left, right in zip(before, after))
    return output, layer, changed


def _validate_state_commit(previous: Module, updated: Module) -> None:
    """Reject storing traced updates in an eagerly captured container."""
    if (any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(updated))
            and not any(isinstance(leaf, jax.core.Tracer)
                        for leaf in jax.tree.leaves(previous))):
        raise RuntimeError(
            'Stateful stacks cannot be captured by a JAX transformation. '
            'Pass the container into the transformed function and return '
            'the updated container.'
        )


def _stack_modules(modules: Iterable[Module]) -> tuple[Module, int]:
    """Stacks a sequence of compatible modules into a single module.

    Args:
        modules (Iterable[Module]): The sequence of modules to stack.

    Returns:
        tuple[Module, int]: A tuple containing the stacked module and the number of stacked modules.
    """
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
    for index, module in enumerate(modules[1:], start=1):
        mismatch = _stack_mismatch(reference, module)
        if mismatch is not None:
            raise ValueError(f'modules[0] and modules[{index}]: {mismatch}')

    stacked = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        *modules,
    )

    _shift_stack_axes(stacked, add=True)
    return stacked, len(modules)


def _stack_compatible(reference: Module, module: Module) -> bool:
    """Checks if a module is compatible for stacking with a reference module.

    Args:
        reference (Module): The reference module to check against.
        module (Module): The module to check for compatibility.

    Returns:
        bool: True if the modules are compatible, False otherwise.
    """
    return _stack_mismatch(reference, module) is None


def _group_stack_compatible(
    modules: Sequence[Module],
) -> tuple[tuple[Module, ...], ...]:
    """Groups consecutive modules in a sequence that are compatible for stacking.

    Args:
        modules (Sequence[Module]): The sequence of modules to group.

    Returns:
        tuple[tuple[Module, ...], ...]: A tuple of module groups, where each group contains stack-compatible modules.
    """
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
    """A list-like module container.

    Stores the supplied module objects without copying them. Supports iteration,
    integer indexing and slicing; slices share child modules with the original.
    Parameters participate in PyTree traversal and recursive train/eval calls.
    This is a storage container, not a callable pipeline.

    Args:
        modules (Sequence[Module]): The sequence of modules to store.

    Example:
        >>> from taktiny import nn
        >>> layers = nn.List([nn.Dropout(0.1), nn.Dropout(0.2)])
        >>> len(layers[:1])
        1
    """
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
    """A dictionary-like module container indexed by stable string keys.

    Keys must be nonempty strings without dots, preserving unambiguous parameter
    paths. Iteration follows insertion order. Children are shared, not copied;
    the container supports recursive train/eval and parameter traversal but
    does not define a forward call.

    Args:
        modules (Mapping[str, Module]): A mapping of string keys to modules.

    Example:
        >>> from taktiny import nn
        >>> layers = nn.Dict({'dropout': nn.Dropout(0.1)})
        >>> list(layers)
        ['dropout']
    """

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
        """Returns a view of the module keys.

        Returns:
            KeysView[str]: A view of the module keys.
        """
        return self.layers.keys()

    def values(self) -> ValuesView[Module]:
        """Returns a view of the module values.

        Returns:
            ValuesView[Module]: A view of the module values.
        """
        return self.layers.values()

    def items(self) -> ItemsView[str, Module]:
        """Returns a view of the module items (key-value pairs).

        Returns:
            ItemsView[str, Module]: A view of the module items.
        """
        return self.layers.items()

    def extra_repr(self) -> str:
        return f'{len(self.layers)}'

class Sequential(Module):
    """Apply modules in order, feeding each output into the next module.

    Extra positional and keyword arguments are broadcast to every layer; they
    are not filtered by signature. Tuple/dict outputs remain a single input
    PyTree and are not unpacked. An empty sequence is the identity operation.
    Slices retain references to the original child modules.

    Args:
        modules (Sequence[Module]): The sequence of modules to chain.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> model = nn.Sequential([nn.Dropout(0.1), nn.Dropout(0.2)])
        >>> with nn.set_context_rng(nn.Rngs(0)):
        ...     y = model(jnp.ones((8, 4)))
        >>> y.shape
        (8, 4)
    """

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
        """Applies each module in sequence to the input.

        Args:
            x (PyTree): The input data.

        Returns:
            PyTree: The output after applying all modules sequentially.
        """
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

    def extra_repr(self) -> str:
        return f"{len(self.layers)}"


class SeqStack(Module):
    """A sequential module stack that uses scan to apply stacked modules.

    Consecutive layers with identical types, static configuration, leaf shapes
    and dtypes are grouped into separate scans. New stacked arrays are created;
    the original modules are not updated. The callback receives an individual
    layer and carry, and must return (next_carry, output). Carry structure,
    shapes and dtypes must be constant within each scan. Group outputs must
    have matching structures, trailing shapes and dtypes, or all be None.
    Outputs retain original layer order even when execution is reversed.

    Array and owned-Rngs updates are stored back into the stack. Calls must not
    change module structure, static configuration, or state shapes/dtypes.
    For stateful JIT execution, pass this container into the compiled function
    and return it alongside the result. Context RNGs must be threaded through
    carry and installed inside the callback, not captured from outside scan.
    Logical names and partition specs gain a replicated leading layer axis;
    the callback sees the original per-layer metadata.

    Individual layer restrictions still apply: BatchNorm's current running
    statistics update is eager-only; use eval mode or track_running_stats=False.

    Args:
        modules (Iterable[Module]): The sequence of modules to stack and scan over.
        reverse (bool, optional): Whether to scan in reverse. Defaults to False.
        unroll (int | bool, optional): Loop unrolling factor. Defaults to 1.
        split_transpose (bool, optional): Whether to split transpose. Defaults to False.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> layers = nn.SeqStack([nn.Linear(4, 4, rngs=nn.Rngs(i)) for i in range(2)])
        >>> def apply(layer, carry):
        ...     return layer(carry), None
        >>> result, _ = layers(apply, jnp.ones((3, 4)))
        >>> result.shape
        (3, 4)
    """

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
        self.stacked: Module | None = None
        self.groups: list[SeqStack] = []
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
        """Applies a scanned function over the stacked modules.

        Args:
            f (Callable[..., Any]): The function to scan over the modules.
            carry (PyTree): The initial carry value for the scan operation.

        Returns:
            tuple[PyTree, PyTree]: A tuple of the final carry and the stacked outputs.
        """
        if self.groups:
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
            reference_leaves = jax.tree.leaves(group_outputs[0])
            for outputs in group_outputs[1:]:
                if any(left.shape[1:] != right.shape[1:] or left.dtype != right.dtype
                       for left, right in zip(reference_leaves, jax.tree.leaves(outputs))):
                    raise ValueError(
                        'all SeqStack groups must return compatible output shapes and dtypes'
                    )
            if self.reverse:
                group_outputs.reverse()
            return carry, jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0),
                *group_outputs,
            )

        state_changed = False

        @tt.scan(
            length=self.num_stack,
            reverse=self.reverse,
            unroll=self.unroll,
            _split_transpose=self.split_transpose,
        )
        def apply_fn(carry: Any, layer: Any, *broadcast_args: Any) -> Any:
            nonlocal state_changed
            (carry, output), layer, changed = _call_stacked(
                layer, f, carry, *broadcast_args, **kwargs,
            )
            state_changed = state_changed or changed
            return carry, (output, layer)

        assert self.stacked is not None
        carry, (outputs, updated) = apply_fn(carry, self.stacked, *args)
        _update_output_axes(outputs, 0)
        if state_changed:
            _validate_state_commit(self.stacked, updated)
            self.stacked = updated
        return carry, outputs

    def extra_repr(self) -> str:
        if self.groups:
            groups = ', '.join(map(str, self.group_sizes))
            return f'{self.num_stack}, groups=({groups})'
        return f'{self.num_stack}'


class Stack(Module):
    """A module stack that uses vmap to apply stacked modules.

    All layers must share type, static configuration, leaf shapes and dtypes.
    Creates stacked arrays without changing the original modules. in_axes
    maps positional inputs, with None broadcasting an input to every layer.
    A tuple provides one axis/PyTree specification per positional argument.
    Keyword arguments are always broadcast unchanged. out_axes controls only
    result axes; stored module state always uses leading axis zero.

    Array and owned-Rngs updates are retained. Static configuration and state
    shapes/dtypes must not change during a call. Under jit, pass a stateful
    container in and return it to carry updates between compiled calls. An
    outer context RNG cannot be captured by vmap; use independent owned RNGs
    or map independent keys to a callback that establishes its own context.
    Parameter logical names and partition specs acquire a replicated layer
    axis; each mapped call sees the original per-layer metadata.

    Individual layer restrictions still apply: BatchNorm's current running
    statistics update is eager-only; use eval mode or track_running_stats=False.

    Args:
        modules (Iterable[Module]): The modules to stack and vectorize.
        axis_name (Any | None, optional): The name of the mapped axis. Defaults to None.
        spmd_axis_name (Any | tuple[Any, ...] | None, optional): The name of the SPMD mapped axis. Defaults to None.

    Example:
        >>> from taktiny import nn
        >>> import jax, jax.numpy as jnp
        >>> layers = nn.Stack([nn.Dropout(0.5, rngs=nn.Rngs(i)) for i in range(2)])
        >>> @jax.jit
        ... def step(model, x):
        ...     y = model(x, in_axes=None)
        ...     return y, model
        >>> y, layers = step(layers, jnp.ones((8, 4)))
        >>> y.shape
        (2, 8, 4)
    """

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
        in_axes: int | None | tuple[PyTree, ...] = 0,
        out_axes: Any = 0,
        **kwargs: Any,
    ) -> PyTree:
        """Applies a vectorized function over the stacked modules.

        Args:
            in_axes (int | None | tuple[int  |  None, ...], optional): The axes to map over for arguments. Defaults to 0.
            out_axes (Any, optional): The mapped output axes. Defaults to 0.

        Returns:
            PyTree: The outputs of the vectorized application.
        """
        if isinstance(in_axes, tuple):
            if len(in_axes) != len(args):
                raise ValueError(
                    'tuple in_axes must have one entry per positional argument'
                )
            vmap_in_axes = (0,) + in_axes
        else:
            vmap_in_axes = (0,) + (in_axes,) * len(args)

        state_changed = False

        @tt.vmap(
            in_axes=vmap_in_axes,
            out_axes=(out_axes, 0),
            axis_name=self.axis_name,
            axis_size=self.num_stack,
            spmd_axis_name=self.spmd_axis_name,
        )
        def apply_fn(layer: Any, *positional_args: Any) -> Any:
            nonlocal state_changed
            output, layer, changed = _call_stacked(
                layer, lambda module, *xs: module(*xs, **kwargs), *positional_args,
            )
            state_changed = state_changed or changed
            return output, layer

        output, updated = apply_fn(self.stacked, *args)
        _update_output_axes(output, out_axes)
        if state_changed:
            _validate_state_commit(self.stacked, updated)
            self.stacked = updated
        return output

    def extra_repr(self) -> str:
        return f"{self.num_stack}"


__all__ = [
    'Dict',
    'List',
    'SeqStack',
    'Sequential',
    'Stack',
]
