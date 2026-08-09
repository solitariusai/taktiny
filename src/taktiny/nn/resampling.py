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
"""Resampling module utilities"""
from __future__ import annotations
import math
import jax
from collections.abc import Sequence

from taktiny import nn
from taktiny.nn.utils import (
    _as_batched,
    _normalize_adaptive_size,
    _restore_batch,
)

class Upsample(nn.Module):
    """Resize channels-last inputs to a size or by a spatial scale factor."""

    def __init__(
        self,
        size: int | Sequence[int | None] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        method: str = 'nearest',
        antialias: bool = True,
    ) -> None:
        if size is not None and scale_factor is not None:
            raise ValueError('size and scale_factor are mutually exclusive')
        if size is None and scale_factor is None:
            scale_factor = 2.0
        if size is not None:
            self.size = _normalize_adaptive_size(size)
            self.scale_factor = None
            rank = len(self.size)
        else:
            if isinstance(scale_factor, (int, float)):
                scale_factor = (float(scale_factor),)
            else:
                scale_factor = tuple(float(value) for value in scale_factor)
            if not scale_factor or any(value <= 0 for value in scale_factor):
                raise ValueError('scale_factor values must be positive')
            self.size = None
            self.scale_factor = scale_factor
            rank = len(scale_factor)
        method = {
            'bilinear': 'linear',
            'trilinear': 'linear',
            'bicubic': 'cubic',
        }.get(method.lower(), method.lower())
        self.method = method
        self.antialias = antialias
        self.spatial_rank = rank

    def __call__(self, x: jax.Array) -> jax.Array:
        x, unbatched = _as_batched(x, self.spatial_rank)
        if self.size is not None:
            spatial_shape = tuple(
                current if requested is None else requested
                for current, requested in zip(x.shape[1:-1], self.size)
            )
        else:
            spatial_shape = tuple(
                max(1, math.floor(current * scale))
                for current, scale in zip(
                    x.shape[1:-1],
                    self.scale_factor,
                )
            )
        output = jax.image.resize(
            x,
            shape=(x.shape[0], *spatial_shape, x.shape[-1]),
            method=self.method,
            antialias=self.antialias,
        )
        return _restore_batch(output, unbatched)

__all__ = ['Upsample']
