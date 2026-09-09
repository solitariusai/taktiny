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
"""Dimension-agnostic channels-last resampling modules."""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import jax
import jax.numpy as jnp
from jax.image import ResizeMethod
from jax.lax import PrecisionLike
from jax.sharding import NamedSharding

from taktiny.nn.base import Module
from taktiny.nn.utils import (
    _as_batched,
    _constrain,
    _restore_batch,
    _validate_positive_float,
)


def _normalize_size(size: int | Sequence[int | None]) -> tuple[int | None, ...]:
    """Validate spatial sizes without treating booleans as integers."""
    values: tuple[int | None, ...]
    if isinstance(size, bool):
        raise TypeError('size must be an integer or sequence of integers or None')
    if isinstance(size, int):
        values = (size,)
    elif isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        values = tuple(size)
    else:
        raise TypeError('size must be an integer or sequence of integers or None')
    if not values:
        raise ValueError('size must contain at least one dimension')
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError('size entries must be integers or None')
        if value <= 0:
            raise ValueError('size entries must be positive')
    return values


class _Resize(Module):
    """Shared configuration and channels-last spatial resizing."""

    def __init__(
        self,
        size: int | Sequence[int | None] | None,
        scale_factor: float | Sequence[float] | None,
        *,
        default_scale_factor: float,
        method: str | ResizeMethod,
        antialias: bool,
        precision: PrecisionLike,
    ) -> None:
        self.size: tuple[int | None, ...] | None
        self.scale_factor: tuple[float, ...] | None
        if size is not None and scale_factor is not None:
            raise ValueError('size and scale_factor are mutually exclusive')
        if size is None and scale_factor is None:
            scale_factor = default_scale_factor

        if size is not None:
            self.size = _normalize_size(size)
            self.scale_factor = None
            spatial_rank = len(self.size)
        else:
            values: tuple[float, ...]
            if isinstance(scale_factor, Real) and not isinstance(
                scale_factor,
                bool,
            ):
                values = (float(scale_factor),)
            elif isinstance(scale_factor, Sequence) and not isinstance(
                scale_factor, (str, bytes),
            ):
                values = tuple(scale_factor)
            else:
                raise TypeError('scale_factor must be a number or sequence of numbers')
            if not values:
                raise ValueError('scale_factor must contain at least one value')
            self.scale_factor = tuple(
                _validate_positive_float(value, f'scale_factor[{index}]')
                for index, value in enumerate(values)
            )
            self.size = None
            spatial_rank = len(self.scale_factor)

        if isinstance(method, str):
            method = ResizeMethod.from_string(method.lower())
        elif not isinstance(method, ResizeMethod):
            raise TypeError('method must be a string or jax.image.ResizeMethod')
        self.method = method
        if not isinstance(antialias, bool):
            raise TypeError('antialias must be a boolean')
        self.antialias = antialias
        self.precision = precision
        self.spatial_rank = spatial_rank

    def _scaled_size(self, current: int, scale: float) -> int:
        """Computes the scaled size for a single dimension.

        Args:
            current (int): Current size of the dimension.
            scale (float): Scale factor to apply.

        Returns:
            int: The new scaled size.
        """
        raise NotImplementedError

    def _spatial_shape(self, current_shape: Sequence[int]) -> tuple[int, ...]:
        """Computes the target spatial shape.

        Args:
            current_shape (Sequence[int]): The current spatial shape.

        Returns:
            tuple[int, ...]: The target spatial shape.
        """
        if self.size is not None:
            return tuple(
                current if requested is None else requested
                for current, requested in zip(current_shape, self.size)
            )
        assert self.scale_factor is not None
        return tuple(
            self._scaled_size(current, scale)
            for current, scale in zip(current_shape, self.scale_factor)
        )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the resize operation to the input array.

        Args:
            x (jax.Array): Input channels-last array to resize.
            out_sharding (jax.sharding.Sharding | None, optional): Target sharding. Defaults to None.

        Returns:
            jax.Array: Resized output array.
        """
        x, unbatched = _as_batched(jnp.asarray(x), self.spatial_rank)
        if any(size == 0 for size in x.shape[1:-1]):
            raise ValueError('input spatial dimensions must be nonempty')
        spatial_shape = self._spatial_shape(x.shape[1:-1])
        output = jax.image.resize(
            x,
            shape=(x.shape[0], *spatial_shape, x.shape[-1]),
            method=self.method,
            antialias=self.antialias,
            precision=self.precision,
        )
        output = _restore_batch(output, unbatched)
        if (isinstance(out_sharding, NamedSharding)
                and out_sharding.mesh.are_all_axes_explicit):
            return jax.sharding.reshard(output, out_sharding)
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        """Returns extra representation string for the module.

        Returns:
            str: Extra representation of the module parameters.
        """
        if self.size is not None:
            target = 'size=' + '×'.join(
                '*' if size is None else str(size) for size in self.size
            )
        else:
            target = f'scale_factor={self.scale_factor}'
        return (f'{target}, method={self.method.name.lower()}, '
                f'antialias={self.antialias}, precision={self.precision}')


class Upsample(_Resize):
    """Resize channels-last spatial dimensions by a multiplicative factor.

    For n spatial dimensions, accepts ``(*spatial, channels)`` or
    ``(batch, *spatial, channels)``. Scalar size/scale_factor means n=1; a sequence
    specifies n explicitly. For 2-D images, use e.g. scale_factor=(2, 2).
    Batch and channel dimensions are preserved. Only one channel axis is
    supported; trailing N-D feature blocks are not inferred.

    Args:
        size: Positive target sizes. None entries in a sequence preserve the
            corresponding input dimensions. Mutually exclusive with scale_factor.
        scale_factor: Finite positive multipliers. Each output size is
            max(1, floor(input_size * factor)). Defaults to 2 for 1-D input
            when neither size nor scale_factor is given. Factors below one
            and smaller explicit sizes remain supported for compatibility.
        method: JAX resize method string or ResizeMethod enum. Defaults to
            nearest; supports linear, cubic, Lanczos and JAX's method aliases.
        antialias: Apply filtering when shrinking. No effect when enlarging
            or using nearest-neighbor interpolation. Defaults to True.
        precision: Interpolation contraction precision; defaults to HIGHEST,
            matching jax.image.resize. Ignored for nearest interpolation.

    __call__ accepts out_sharding for the final output. Nearest preserves the
    input dtype; other methods follow JAX's floating-point promotion. Input
    spatial dimensions must be nonempty. No parameters or RNGs are used.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> layer = nn.Upsample(scale_factor=(2, 3), method='linear')
        >>> layer(jnp.ones((4, 5, 3))).shape
        (8, 15, 3)
    """
    def __init__(
        self,
        size: int | Sequence[int | None] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        method: str | ResizeMethod = 'nearest',
        antialias: bool = True,
        *,
        precision: PrecisionLike = jax.lax.Precision.HIGHEST,
    ) -> None:
        super().__init__(
            size,
            scale_factor,
            default_scale_factor=2.0,
            method=method,
            antialias=antialias,
            precision=precision,
        )

    def _scaled_size(self, current: int, scale: float) -> int:
        """Computes the upscaled size for a single dimension.

        Args:
            current (int): Current size of the dimension.
            scale (float): Scale factor to apply.

        Returns:
            int: The new upscaled size.
        """
        return max(1, math.floor(current * scale))


class Downsample(_Resize):
    """Reduce channels-last spatial dimensions by a divisive factor.

    For n spatial dimensions, accepts ``(*spatial, channels)`` or
    ``(batch, *spatial, channels)``. Scalar size/scale_factor describes 1-D input;
    use a sequence such as (2, 2) for 2-D resizing. Batch and the single trailing
    channel axis are preserved; N-D feature blocks are not inferred.

    Args:
        size: Positive target sizes no larger than the input. None entries
            preserve individual dimensions. Mutually exclusive with scale_factor.
        scale_factor: Finite divisors greater than or equal to one. Each output
            size is max(1, floor(input_size / factor)). Defaults to 2 for 1-D
            input when neither size nor scale_factor is supplied.
        method: JAX resize method string or ResizeMethod enum. Defaults to
            linear; supports nearest, cubic, Lanczos and JAX's method aliases.
        antialias: Filter when reducing spatial dimensions to limit aliasing.
            Defaults to True. Ignored for nearest-neighbor interpolation.
        precision: Interpolation contraction precision; defaults to HIGHEST,
            matching jax.image.resize. Ignored for nearest interpolation.

    __call__ accepts out_sharding for the final output. Nearest preserves the
    input dtype; other methods follow JAX's floating-point promotion. Input
    spatial dimensions must be nonempty. This operation has no RNG or parameters.

    Examples:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.Downsample(scale_factor=(2, 3))(jnp.ones((2, 8, 9, 4))).shape
        (2, 4, 3, 4)
        >>> nn.Downsample(size=(3, None))(jnp.ones((7, 5, 4))).shape
        (3, 5, 4)
    """

    def __init__(
        self,
        size: int | Sequence[int | None] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        method: str | ResizeMethod = 'linear',
        antialias: bool = True,
        *,
        precision: PrecisionLike = jax.lax.Precision.HIGHEST,
    ) -> None:
        super().__init__(
            size,
            scale_factor,
            default_scale_factor=2.0,
            method=method,
            antialias=antialias,
            precision=precision,
        )
        if self.scale_factor is not None and any(
            scale < 1.0 for scale in self.scale_factor
        ):
            raise ValueError(
                'Downsample scale_factor values must be greater than or '
                'equal to 1'
            )

    def _scaled_size(self, current: int, scale: float) -> int:
        """Computes the downscaled size for a single dimension.

        Args:
            current (int): Current size of the dimension.
            scale (float): Reduction factor to apply.

        Returns:
            int: The new downscaled size.
        """
        return max(1, math.floor(current / scale))

    def _spatial_shape(self, current_shape: Sequence[int]) -> tuple[int, ...]:
        """Computes and validates the target spatial shape.

        Args:
            current_shape (Sequence[int]): The current spatial shape.

        Returns:
            tuple[int, ...]: The target spatial shape.
        """
        spatial_shape = super()._spatial_shape(current_shape)
        if any(
            requested > current
            for requested, current in zip(spatial_shape, current_shape)
        ):
            raise ValueError(
                'Downsample size cannot exceed the input spatial shape'
            )
        return spatial_shape


__all__ = ['Downsample', 'Upsample']
