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
