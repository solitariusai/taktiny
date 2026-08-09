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
"""Shape-only flattening modules."""
from __future__ import annotations
from collections.abc import Sequence
import math
import jax
import jax.numpy as jnp
from taktiny import nn
from taktiny.nn.utils import _canonical_axis


class Flatten(nn.Module):
    """Collapse a contiguous range of dimensions into one dimension."""

    def __init__(
        self,
        start_axis: int = 1,
        end_axis: int = -1,
    ) -> None:
        if not isinstance(start_axis, int) or isinstance(start_axis, bool):
            raise TypeError('start_axis must be an integer')
        if not isinstance(end_axis, int) or isinstance(end_axis, bool):
            raise TypeError('end_axis must be an integer')
        self.start_axis = start_axis
        self.end_axis = end_axis

    def __call__(self, x: jax.Array) -> jax.Array:
        start_axis = _canonical_axis(
            self.start_axis,
            x.ndim,
            name='start_axis',
            allow_scalar=True,
        )
        end_axis = _canonical_axis(
            self.end_axis,
            x.ndim,
            name='end_axis',
            allow_scalar=True,
        )
        if start_axis > end_axis:
            raise ValueError(
                'start_axis must refer to an axis before or equal to end_axis'
            )

        flattened_size = math.prod(x.shape[start_axis:end_axis + 1])
        shape = (
            *x.shape[:start_axis],
            flattened_size,
            *x.shape[end_axis + 1:],
        )
        return jnp.reshape(x, shape)

    def extra_repr(self) -> str:
        return f'start_axis={self.start_axis}, end_axis={self.end_axis}'


class Unflatten(nn.Module):
    """Expand one dimension into a specified sequence of dimensions."""

    def __init__(
        self,
        axis: int,
        unflattened_size: int | Sequence[int],
    ) -> None:
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise TypeError('axis must be an integer')
        if isinstance(unflattened_size, int):
            unflattened_size = (unflattened_size,)
        elif isinstance(unflattened_size, Sequence) and not isinstance(
            unflattened_size,
            (str, bytes),
        ):
            unflattened_size = tuple(unflattened_size)
        else:
            raise TypeError(
                'unflattened_size must be an integer or a sequence of integers'
            )

        if not unflattened_size:
            raise ValueError('unflattened_size must contain at least one dimension')
        if any(
            not isinstance(size, int) or isinstance(size, bool)
            for size in unflattened_size
        ):
            raise TypeError('unflattened_size values must be integers')
        if any(size < -1 for size in unflattened_size):
            raise ValueError(
                'unflattened_size values must be non-negative or -1'
            )
        if unflattened_size.count(-1) > 1:
            raise ValueError('only one unflattened dimension may be inferred')

        self.axis = axis
        self.unflattened_size = unflattened_size

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim == 0:
            raise ValueError('cannot unflatten a scalar input')
        axis = _canonical_axis(self.axis, x.ndim, name='axis')
        sizes = self.unflattened_size
        flattened_size = x.shape[axis]

        if -1 in sizes:
            known_size = math.prod(size for size in sizes if size != -1)
            if known_size == 0:
                raise ValueError(
                    'cannot infer an unflattened dimension when the known '
                    'dimensions have size zero'
                )
            if flattened_size % known_size:
                raise ValueError(
                    f'dimension of size {flattened_size} cannot be unflattened '
                    f'into {sizes}'
                )
            inferred_size = flattened_size // known_size
            sizes = tuple(
                inferred_size if size == -1 else size
                for size in sizes
            )
        elif math.prod(sizes) != flattened_size:
            raise ValueError(
                f'dimension of size {flattened_size} cannot be unflattened '
                f'into {sizes}'
            )

        shape = (*x.shape[:axis], *sizes, *x.shape[axis + 1:])
        return jnp.reshape(x, shape)

    def extra_repr(self) -> str:
        return f'axis={self.axis}, unflattened_size={self.unflattened_size}'


__all__ = ['Flatten', 'Unflatten']
