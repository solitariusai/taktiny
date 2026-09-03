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
"""Taktiny transformations built on JAX."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

import jax

from taktiny.nn.base import Module


def _add_mapped_axis(module: Module, axis: int) -> None:
    seen = set()
    for parameter in module.flat_parameter_dict().values():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))

        ndim = parameter.ndim
        mapped_axis = axis if axis >= 0 else axis + ndim

        axis_names = getattr(parameter, 'axis_names', None)
        if axis_names is not None and len(axis_names) == ndim - 1:
            parameter.axis_names = (
                tuple(axis_names[:mapped_axis])
                + (None,)
                + tuple(axis_names[mapped_axis:])
            )




def _update_output_axes(output: Any, out_axes: Any) -> None:
    if out_axes is None:
        return

    try:
        axis = operator.index(out_axes)
    except TypeError:
        axis = None

    if axis is not None:
        if isinstance(output, Module):
            _add_mapped_axis(output, axis)
        elif isinstance(output, dict):
            for value in output.values():
                _update_output_axes(value, axis)
        elif isinstance(output, (list, tuple)):
            for value in output:
                _update_output_axes(value, axis)
        return

    if isinstance(output, dict) and isinstance(out_axes, dict):
        for key, axes in out_axes.items():
            _update_output_axes(output[key], axes)
    elif (
        isinstance(output, (list, tuple))
        and isinstance(out_axes, (list, tuple))
    ):
        for value, axes in zip(output, out_axes):
            _update_output_axes(value, axes)


def vmap[F: Callable[..., Any]](
    fun: F | None = None,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: Any | None = None,
    axis_size: int | None = None,
    spmd_axis_name: Any | tuple[Any, ...] | None = None,
    sum_match: bool = False,
) -> Any:
    """Vectorize a function using :func:`jax.vmap`.

    The function can be supplied directly or through decorator syntax.

    Examples:
        >>> mapped = vmap(function, in_axes=0)
        >>> @vmap(in_axes=0)
        ... def mapped_function(x):
        ...     return x + 1
    """

    def transform(function: Any) -> Any:
        mapped = jax.vmap(
            function,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            sum_match=sum_match,
        )

        @wraps(function)
        def transformed(*args: Any, **kwargs: Any) -> Any:
            output = mapped(*args, **kwargs)
            _update_output_axes(output, out_axes)
            return output

        return transformed

    if fun is None:
        return transform
    if not callable(fun):
        raise TypeError(f'fun must be callable, got {type(fun).__name__}')
    return transform(fun)


def scan[F: Callable[..., Any]](
    fun: F | None = None,
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> Any:
    """Transform a scan body into a callable using :func:`jax.lax.scan`.

    The transformed function accepts ``(init, xs, *args, **kwargs)``. Extra
    arguments are broadcast across iterations and passed to the scan body
    after ``carry`` and ``x``.

    Examples:
        >>> scanned = scan(body, reverse=True)
        >>> final_carry, outputs = scanned(initial_carry, xs)
        >>> @scan(unroll=2)
        ... def scanned_body(carry, x):
        ...     return carry + x, carry
    """

    def transform(function: Any) -> Any:
        if not callable(function):
            raise TypeError(
                f'fun must be callable, got {type(function).__name__}'
            )

        @wraps(function)
        def scanned(init: Any, xs: Any=None, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            def body(carry: Any, x: Any) -> Any:
                return function(carry, x, *args, **kwargs)

            carry, outputs = jax.lax.scan(
                body,
                init,
                xs,
                length=length,
                reverse=reverse,
                unroll=unroll,
                _split_transpose=_split_transpose,
            )
            _update_output_axes(outputs, 0)
            return carry, outputs

        return scanned

    if fun is None:
        return transform
    return transform(fun)


__all__ = ['scan', 'vmap']