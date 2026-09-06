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

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from taktiny.nn.base import Module
from taktiny.nn.utils import _canonical_axis, _constrain
from taktiny.utils.typing import GenericShape


def _reshape(
    x: jax.Array,
    shape: tuple[int, ...],
    out_sharding: jax.sharding.Sharding | None,
) -> jax.Array:
    # Explicit layouts must reach reshape itself: splitting a sharded axis can
    # be ambiguous before a constraint on the result can be applied.
    if (isinstance(out_sharding, NamedSharding)
            and out_sharding.mesh.are_all_axes_explicit):
        return jnp.reshape(x, shape, out_sharding=out_sharding)
    return _constrain(jnp.reshape(x, shape), out_sharding)


class Flatten(Module):
    """Merge an inclusive, contiguous range of axes in row-major order.

    Args:
        start_axis: First axis to merge; defaults to 1 to preserve a leading
            batch axis. Negative axes count from the end of the input.
        end_axis: Last axis to merge, inclusive; defaults to -1.

    Axes outside the selected range, element order, and dtype are preserved.
    No batch or channel axes are inferred. A scalar can be flattened to (1,)
    using start_axis=0 (or -1); the default start_axis=1 requires rank >= 2.
    Zero-sized dimensions are supported. Axis bounds and order are checked
    against the input rank at call time. The module has no parameters or RNGs.

    __call__ accepts out_sharding describing the output's axes, not the input's.
    With None, layout follows JAX's reshape rules; it does not force replication.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.Flatten()(jnp.ones((2, 3, 4))).shape
        (2, 12)
        >>> nn.Flatten(1, -2)(jnp.ones((2, 3, 4, 5))).shape
        (2, 12, 5)
    """

    def __init__(
        self,
        start_axis: int = 1,
        end_axis: int = -1,
    ) -> None:
        """Initializes a Flatten module.

        Args:
            start_axis (int, optional): The first axis to flatten. Defaults to 1.
            end_axis (int, optional): The last axis to flatten. Defaults to -1.
        """
        if not isinstance(start_axis, int) or isinstance(start_axis, bool):
            raise TypeError('start_axis must be an integer')
        if not isinstance(end_axis, int) or isinstance(end_axis, bool):
            raise TypeError('end_axis must be an integer')
        self.start_axis = start_axis
        self.end_axis = end_axis

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Flattens the specified axes of the input tensor.

        Args:
            x (jax.Array): The input tensor to be flattened.
            out_sharding: Optional layout for the reshaped output.

        Returns:
            jax.Array: The flattened tensor.
        """
        x = jnp.asarray(x)
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
        return _reshape(x, shape, out_sharding)

    def extra_repr(self) -> str:
        return f'start_axis={self.start_axis}, end_axis={self.end_axis}'


class Unflatten(Module):
    """Replace one axis with a specified shape in row-major order.

    Args:
        axis: Axis to expand. Negative axes count from the end of the input.
        unflattened_size: An integer or nonempty sequence of integer sizes.
            Sizes may be nonnegative, with at most one -1 for inference. The
            sequence is stored as an immutable tuple.

    The new sizes must multiply to the selected axis size, independently of
    other axes (even if those axes have size zero). A -1 size is inferred when
    the product of known sizes divides the selected size. Combining -1 with
    zero is ambiguous and raises ValueError; inferring zero from a zero-sized
    axis and positive known sizes is supported. Scalar inputs are unsupported.
    Element order, dtype, and all other axes are preserved; no parameters or
    RNGs are used.

    __call__ accepts out_sharding describing the expanded output axes. With
    None, JAX infers the layout; explicitly sharded inputs may require a target
    layout when splitting an axis has multiple possible sharding assignments.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.Unflatten(-1, [3, -1])(jnp.ones((2, 12))).shape
        (2, 3, 4)
        >>> nn.Unflatten(0, (2, 0))(jnp.empty((0,))).shape
        (2, 0)
    """

    def __init__(
        self,
        axis: int,
        unflattened_size: GenericShape,
    ) -> None:
        """Initializes an Unflatten module.

        Args:
            axis (int): The axis to unflatten.
            unflattened_size (GenericShape): New dimensions, with at most one -1 for inference.
        """
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

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Unflattens the specified axis of the input tensor.

        Args:
            x (jax.Array): The input tensor to unflatten.
            out_sharding: Optional layout for the reshaped output.

        Returns:
            jax.Array: The unflattened tensor.
        """
        x = jnp.asarray(x)
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
        return _reshape(x, shape, out_sharding)

    def extra_repr(self) -> str:
        shape = '×'.join(map(str, self.unflattened_size))
        return f'axis={self.axis}, unflattened_size={shape}'


__all__ = ['Flatten', 'Unflatten']
