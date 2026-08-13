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

from collections.abc import Sequence
import math
from numbers import Real

import jax

from taktiny import nn
from taktiny.nn._continuo import (
    _as_batched,
    _constrain,
    _normalize_adaptive_size,
    _restore_batch,
    _validate_positive_float,
)
from taktiny.utils.typing import ShardMode


_METHOD_ALIASES = {
    'bilinear': 'linear',
    'trilinear': 'linear',
    'bicubic': 'cubic',
}


class _Resize(nn.Module):
    """Shared implementation for spatial resize modules."""

    def __init__(
        self,
        size: int | Sequence[int | None] | None,
        scale_factor: float | Sequence[float] | None,
        *,
        default_scale_factor: float,
        method: str,
        antialias: bool,
        shard_mode: ShardMode,
    ) -> None:
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
        self.shard_mode = shard_mode

    def _scaled_size(self, current: int, scale: float) -> int:
        raise NotImplementedError

    def _spatial_shape(self, current_shape: Sequence[int]) -> tuple[int, ...]:
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
        x, unbatched = _as_batched(x, self.spatial_rank)
        spatial_shape = self._spatial_shape(x.shape[1:-1])
        output = jax.image.resize(
            x,
            shape=(x.shape[0], *spatial_shape, x.shape[-1]),
            method=self.method,
            antialias=self.antialias,
        )
        output = _restore_batch(output, unbatched)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        if self.size is not None:
            target = f'size={self.size}'
        else:
            target = f'scale_factor={self.scale_factor}'
        return f'{target}, method={self.method}'


class Upsample(_Resize):
    """Resize channels-last inputs to a size or by a spatial scale factor."""

    def __init__(
        self,
        size: int | Sequence[int | None] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        method: str = 'nearest',
        antialias: bool = True,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        super().__init__(
            size,
            scale_factor,
            default_scale_factor=2.0,
            method=method,
            antialias=antialias,
            shard_mode=shard_mode,
        )

    def _scaled_size(self, current: int, scale: float) -> int:
        return max(1, math.floor(current * scale))


class Downsample(_Resize):
    """Reduce channels-last spatial axes by an exact size or divisor.

    ``scale_factor`` is a downsampling divisor, so ``scale_factor=2`` halves
    each configured spatial axis. Inputs may be batched or unbatched and may
    contain any positive number of spatial dimensions.
    """

    def __init__(
        self,
        size: int | Sequence[int | None] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        method: str = 'linear',
        antialias: bool = True,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        super().__init__(
            size,
            scale_factor,
            default_scale_factor=2.0,
            method=method,
            antialias=antialias,
            shard_mode=shard_mode,
        )
        if self.scale_factor is not None and any(
            scale < 1.0 for scale in self.scale_factor
        ):
            raise ValueError(
                'Downsample scale_factor values must be greater than or '
                'equal to 1'
            )

    def _scaled_size(self, current: int, scale: float) -> int:
        return max(1, math.floor(current / scale))

    def _spatial_shape(self, current_shape: Sequence[int]) -> tuple[int, ...]:
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
