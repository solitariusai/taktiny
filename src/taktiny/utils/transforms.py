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
"""Generic JAX transforms. State and RNGs are explicit; metadata is untouched."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

import jax
import jax.numpy as jnp


def _axis(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f'{name} leaves must be integers or None, not booleans')
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f'{name} leaves must be integers or None') from error


def _axes_for_tree(axes: Any, tree: Any, name: str) -> list[int | None]:
    """Expand a PyTree-prefix specification to one axis per data leaf."""
    result: list[int | None] = []
    def expand(axis: Any, subtree: Any) -> None:
        result.extend([_axis(axis, name)] * len(jax.tree.leaves(subtree)))
    try:
        jax.tree.map(expand, axes, tree, is_leaf=lambda value: value is None)
    except ValueError as error:
        raise ValueError(f'{name} must be a PyTree prefix of its data') from error
    return result


def _drop_scan_outputs(outputs: Any, axes: Any) -> Any:
    """Drop requested subtrees before scan allocates their history."""
    def keep(axis: Any, subtree: Any) -> Any:
        return None if _axis(axis, 'out_axes') is None else subtree
    try:
        return jax.tree.map(keep, axes, outputs, is_leaf=lambda value: value is None)
    except ValueError as error:
        raise ValueError('out_axes must be a PyTree prefix of the body output') from error


def vmap[F: Callable[..., Any]](
    fun: F | None = None,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: Any | None = None,
    axis_size: int | None = None,
    spmd_axis_name: Any | tuple[Any, ...] | None = None,
    sum_match: bool = False,
) -> Any:
    """Vectorize arbitrary PyTrees with JAX's vmap semantics.

    Supports vmap(function), @vmap and @vmap(...). Positional inputs follow
    in_axes; keyword-argument array leaves map along axis zero, as in JAX.
    out_axes=None retains JAX's unmapped-output meaning, not scan's discard
    meaning. No module metadata is changed and RNGs are not split implicitly.
    Return updated state explicitly; use Stack for module-state conveniences.

    Examples:
        >>> import jax.numpy as jnp
        >>> @vmap(in_axes=(0, None))
        ... def scale(x, factor):
        ...     return x * factor
        >>> scale(jnp.arange(3), 2).tolist()
        [0, 2, 4]
    """

    def transform(function: Any) -> Any:
        if not callable(function):
            raise TypeError(f'fun must be callable, got {type(function).__name__}')
        mapped = jax.vmap(
            function,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            sum_match=sum_match,
        )

        return wraps(function)(mapped)

    if fun is None:
        return transform
    if not callable(fun):
        raise TypeError(f'fun must be callable, got {type(fun).__name__}')
    return transform(fun)


def _is_carry(axis: Any) -> bool:
    return isinstance(axis, str) and axis == 'carry'


def _positional_scan(function: Any, in_axes: Any, out_axes: Any, **options: Any) -> Any:
    """Adapt positional carry slots to the legacy single-carry scan engine."""
    if not isinstance(in_axes, (tuple, list)) or not isinstance(out_axes, (tuple, list)):
        raise ValueError("positional scan requires sequences for both in_axes and out_axes")
    in_axes, out_axes = tuple(in_axes), tuple(out_axes)
    input_carries = tuple(i for i, axis in enumerate(in_axes) if _is_carry(axis))
    output_carries = tuple(i for i, axis in enumerate(out_axes) if _is_carry(axis))
    if not input_carries or len(input_carries) != len(output_carries):
        raise ValueError("in_axes and out_axes must have the same nonzero number of 'carry' entries")
    scanned_axes = tuple(axis for axis in in_axes if not _is_carry(axis))
    stacked_axes = tuple(axis for axis in out_axes if not _is_carry(axis))

    @wraps(function)
    def scanned(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if len(args) != len(in_axes):
            raise ValueError(f'in_axes specifies {len(in_axes)} positional arguments, got {len(args)}')
        initial = tuple(args[i] for i in input_carries)
        inputs = tuple(arg for arg, axis in zip(args, in_axes) if not _is_carry(axis))

        def body(carry: Any, slices: Any) -> Any:
            carried, sliced = iter(carry), iter(slices)
            arguments = tuple(next(carried) if _is_carry(axis) else next(sliced) for axis in in_axes)
            outputs = function(*arguments, **kwargs)
            if not isinstance(outputs, tuple) or len(outputs) != len(out_axes):
                raise ValueError(f'positional scan body must return a tuple of {len(out_axes)} outputs')
            return (
                tuple(outputs[i] for i in output_carries),
                tuple(output for output, axis in zip(outputs, out_axes) if not _is_carry(axis)),
            )

        final, history = scan(body, in_axes=scanned_axes, out_axes=stacked_axes, **options)(initial, inputs)
        carried, stacked = iter(final), iter(history)
        return tuple(next(carried) if _is_carry(axis) else next(stacked) for axis in out_axes)

    return scanned


def scan[F: Callable[..., Any]](
    fun: F | None = None,
    *,
    in_axes: Any = 0,
    out_axes: Any = 0,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> Any:
    """Build a generic scan callable with configurable input and output axes.

    Positional mode is selected by a top-level ``'carry'`` entry in either
    axis specification. In this mode, ``in_axes`` has one entry per positional
    argument and ``out_axes`` one entry per element of the body's return tuple.
    A ``'carry'`` entry carries an entire argument PyTree between iterations.
    Carry outputs feed carry inputs in declaration order, regardless of their
    positions; their counts must match. The returned tuple contains final
    carries and stacked outputs in the body's original output order.
    Integer input axes are scanned; ``None`` inputs are broadcast unchanged.
    Integer output axes place the stacked dimension; ``None`` outputs are
    discarded and returned as ``None``. Non-carry entries may also be PyTree
    prefixes. Carry markers must be top-level entries, not nested in a PyTree.
    Keyword arguments are broadcast, and arguments described by ``in_axes``
    must be supplied positionally. Supply ``length`` if no inputs are scanned.

    Without carry markers, the legacy API below remains unchanged.

    The transformed function accepts ``(init, xs, *args, **kwargs)``. Extra
    arguments are broadcast across iterations and passed to the scan body
    after ``carry`` and ``x``.

    in_axes is an integer, None, or a PyTree prefix of xs. Integer axes select
    the iteration dimension (negative axes are supported); None broadcasts a
    subtree unchanged. All mapped dimensions must have equal length. Supply
    length when there are no mapped leaves, including when xs is None.

    out_axes is an integer or PyTree prefix of body outputs, selecting where
    each stacked iteration axis appears. None discards a subtree before
    stacking; it does not select an invariant or final output. Put a single
    final result in carry. Final carry is never rearranged by out_axes.

    Carry structure, shapes and dtypes remain fixed, as in lax.scan. reverse,
    unroll and _split_transpose are forwarded to JAX. Reversing execution does
    not reverse the returned output order. Module metadata is untouched, and
    mutable state is not implicitly preserved. Thread RNGs/state through carry
    or supply independent keys in xs. Use SeqStack for module conveniences.

    Examples:
        >>> import jax.numpy as jnp
        >>> @scan(in_axes=('carry', 'carry', 0, 1),
        ...       out_axes=('carry', 'carry', 0))
        ... def step(total, count, x, context):
        ...     total = total + x + context
        ...     return total, count + 1, total
        >>> total, count, history = step(
        ...     jnp.zeros(2), 0, jnp.ones((3, 2)), jnp.ones((2, 3)))
        >>> total.tolist(), int(count), history.shape
        ([6.0, 6.0], 3, (3, 2))

        >>> @scan(in_axes=1, out_axes=-1)
        ... def accumulate(carry, x, *, factor=1):
        ...     carry = carry + x * factor
        ...     return carry, carry
        >>> final, history = accumulate(jnp.zeros(2), jnp.ones((2, 3)), factor=2)
        >>> final.tolist(), history.shape
        ([6.0, 6.0], (2, 3))

        >>> @scan(in_axes=None, out_axes=None, length=3)
        ... def repeat(carry, increment):
        ...     return carry + increment, carry
        >>> final, history = repeat(0, 2)
        >>> int(final), history
        (6, None)
    """

    def transform(function: Any) -> Any:
        if not callable(function):
            raise TypeError(
                f'fun must be callable, got {type(function).__name__}'
            )

        if any(
            _is_carry(axis)
            for spec in (in_axes, out_axes)
            for axis in (spec if isinstance(spec, (tuple, list)) else (spec,))
        ):
            return _positional_scan(
                function, in_axes, out_axes, length=length, reverse=reverse,
                unroll=unroll, _split_transpose=_split_transpose,
            )

        @wraps(function)
        def scanned(init: Any, xs: Any = None, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
            leaves, structure = jax.tree.flatten(xs)
            axes = _axes_for_tree(in_axes, xs, 'in_axes')
            mapped = tuple(
                jnp.moveaxis(jnp.asarray(leaf), axis, 0)
                for leaf, axis in zip(leaves, axes) if axis is not None
            )
            if not mapped and length is None:
                raise ValueError('length is required when xs has no scanned leaves')

            def body(carry: Any, slices: Any) -> Any:
                sliced = iter(()) if slices is None else iter(slices)
                x = jax.tree.unflatten(structure, [
                    leaf if axis is None else next(sliced)
                    for leaf, axis in zip(leaves, axes)
                ])
                carry, output = function(carry, x, *args, **kwargs)
                return carry, _drop_scan_outputs(output, out_axes)

            carry, outputs = jax.lax.scan(
                body,
                init,
                mapped if mapped else None,
                length=length,
                reverse=reverse,
                unroll=unroll,
                _split_transpose=_split_transpose,
            )
            output_leaves, output_structure = jax.tree.flatten(outputs)
            output_axes = _axes_for_tree(out_axes, outputs, 'out_axes')
            outputs = jax.tree.unflatten(output_structure, [
                jnp.moveaxis(leaf, 0, axis)
                for leaf, axis in zip(output_leaves, output_axes)
                if axis is not None
            ])
            return carry, outputs

        return scanned

    if fun is None:
        return transform
    return transform(fun)


__all__ = ['scan', 'vmap']
