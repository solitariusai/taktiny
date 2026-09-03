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

from taktiny.nn.base import Module
from taktiny.nn.utils import (
    _as_batched,
    _constrain,
    _normalize_adaptive_size,
    _restore_batch,
    _validate_positive_float,
)

_METHOD_ALIASES = {
    'bilinear': 'linear',
    'trilinear': 'linear',
    'bicubic': 'cubic',
}


class _Resize(Module):
    """Base class for spatial dimension resampling modules."""

    def __init__(
        self,
        size: int | Sequence[int | None] | None,
        scale_factor: float | Sequence[float] | None,
        *,
        default_scale_factor: float,
        method: str,
        antialias: bool,
        ) -> None:
        """Initializes the resize module.

        Args:
            size (int | Sequence[int  |  None] | None): Target spatial size.
            scale_factor (float | Sequence[float] | None): Scaling factor for the spatial dimensions.
            default_scale_factor (float): Default scaling factor if none is provided.
            method (str): Resampling method to use (e.g., 'linear', 'nearest').
            antialias (bool): Whether to apply antialiasing.
        """
        if size is not None and scale_factor is not None:
            raise ValueError('size and scale_factor are mutually exclusive')
        if size is None and scale_factor is None:
            scale_factor = default_scale_factor

        if size is not None:
            self.size = _normalize_adaptive_size(size)
            self.scale_factor = None
            spatial_rank = len(self.size)
        else:
            if isinstance(scale_factor, Real) and not isinstance(
                scale_factor,
                bool,
            ):
                values = (scale_factor,)
            else:
                try:
                    values = tuple(scale_factor)
                except TypeError as error:
                    raise TypeError(
                        'scale_factor must be a number or sequence of numbers'
                    ) from error
            if not values:
                raise ValueError('scale_factor must contain at least one value')
            self.scale_factor = tuple(
                _validate_positive_float(value, f'scale_factor[{index}]')
                for index, value in enumerate(values)
            )
            self.size = None
            spatial_rank = len(self.scale_factor)

        if not isinstance(method, str):
            raise TypeError('method must be a string')
        method = method.lower()
        self.method = _METHOD_ALIASES.get(method, method)
        self.antialias = bool(antialias)
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
        x, unbatched = _as_batched(x, self.spatial_rank)
        spatial_shape = self._spatial_shape(x.shape[1:-1])
        output = jax.image.resize(
            x,
            shape=(x.shape[0], *spatial_shape, x.shape[-1]),
            method=self.method,
            antialias=self.antialias,
        )
        output = _restore_batch(output, unbatched)
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        """Returns extra representation string for the module.

        Returns:
            str: Extra representation of the module parameters.
        """
        if self.size is not None:
            target = f'size={self.size}'
        else:
            target = f'scale_factor={self.scale_factor}'
        return f'{target}, method={self.method}'


class Upsample(_Resize):
    """Upsamples the spatial dimensions of a channels-last array."""

    def __init__(
        self,
        size: int | Sequence[int | None] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        method: str = 'nearest',
        antialias: bool = True,
    ) -> None:
        """Initializes the upsampling module.

        Args:
            size (int | Sequence[int  |  None] | None, optional): Target spatial size. Defaults to None.
            scale_factor (float | Sequence[float] | None, optional): Scaling factor for spatial dimensions. Defaults to None.
            method (str, optional): Resampling method. Defaults to 'nearest'.
            antialias (bool, optional): Whether to apply antialiasing. Defaults to True.
            shard_mode (optional): Sharding mode for the output. Defaults to ShardMode.AUTO.
        """
        super().__init__(
            size,
            scale_factor,
            default_scale_factor=2.0,
            method=method,
            antialias=antialias,
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
    """Downsamples the spatial dimensions of a channels-last array."""

    def __init__(
        self,
        size: int | Sequence[int | None] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        method: str = 'linear',
        antialias: bool = True,
    ) -> None:
        """Initializes the downsampling module.

        Args:
            size (int | Sequence[int  |  None] | None, optional): Target spatial size. Defaults to None.
            scale_factor (float | Sequence[float] | None, optional): Reduction factor for spatial dimensions. Defaults to None.
            method (str, optional): Resampling method. Defaults to 'linear'.
            antialias (bool, optional): Whether to apply antialiasing. Defaults to True.
            shard_mode (optional): Sharding mode for the output. Defaults to ShardMode.AUTO.
        """
        super().__init__(
            size,
            scale_factor,
            default_scale_factor=2.0,
            method=method,
            antialias=antialias,
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
