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
"""Stochastic regularization modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

from taktiny.nn.module import Module
from taktiny.nn.rng import Rngs
from taktiny.nn.continuo import (
    _canonical_axis,
    _canonical_axes,
    _constrain,
    _validate_probability,
)
from taktiny.utils.typing import Axes, ShardMode


StochasticDepthMode: TypeAlias = Literal['batch', 'row']


def _mask_shape(
    shape: Sequence[int],
    broadcast_axes: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        1 if axis in broadcast_axes else size
        for axis, size in enumerate(shape)
    )


def _feature_broadcast_axes(
    ndim: int,
    channel_axis: int,
    batch_axis: int | None,
) -> tuple[int, ...]:
    channel_axis = _canonical_axis(channel_axis, ndim, name='channel_axis')
    canonical_batch_axis = None
    if batch_axis is not None:
        canonical_batch_axis = _canonical_axis(
            batch_axis,
            ndim,
            name='batch_axis',
        )
        if canonical_batch_axis == channel_axis:
            raise ValueError('batch_axis and channel_axis must be different')
    return tuple(
        axis
        for axis in range(ndim)
        if axis not in (canonical_batch_axis, channel_axis)
    )


def _next_key(rngs: Rngs | None) -> jax.Array:
    if rngs is None:
        raise ValueError('rngs is required in training mode when p is nonzero')
    return rngs()


def _validate_rngs(rngs: Rngs | None) -> Rngs | None:
    if rngs is None:
        return None
    if not isinstance(rngs, Rngs):
        raise TypeError('rngs must be an Rngs or None')
    return rngs


class Dropout(Module):
    """Randomly zero activation elements and rescale retained values.

    ``broadcast_axes`` shares one mask value across selected input dimensions.
    An empty tuple gives ordinary elementwise dropout. Randomness comes from
    ``rngs`` supplied during construction. :meth:`Module.train` and
    :meth:`Module.eval` control whether dropout is active.
    """

    def __init__(
        self,
        p: float = 0.5,
        *,
        broadcast_axes: Axes = (),
        rngs: Rngs | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.p = _validate_probability(p)
        self.broadcast_axes = (
            (broadcast_axes,)
            if isinstance(broadcast_axes, int)
            else tuple(broadcast_axes)
        )
        self.rngs = _validate_rngs(rngs)
        self.shard_mode = shard_mode

    def _apply(
        self,
        x: jax.Array,
        *,
        broadcast_axes: tuple[int, ...],
    ) -> jax.Array:
        if not self.training or self.p == 0:
            return x
        if self.p == 1:
            return jnp.zeros_like(x)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            raise TypeError('Dropout requires a floating-point or complex input')

        mask = jax.random.bernoulli(
            _next_key(self.rngs),
            p=1.0 - self.p,
            shape=_mask_shape(x.shape, broadcast_axes),
        )
        return jnp.where(mask, x / (1.0 - self.p), jnp.zeros_like(x))

    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        axes = _canonical_axes(
            self.broadcast_axes,
            x.ndim,
            name='broadcast_axes',
            allow_empty=True,
        )
        output = self._apply(
            x,
            broadcast_axes=axes,
        )
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return f'p={self.p:g}, broadcast_axes={self.broadcast_axes}'


class FeatureDropout(Dropout):
    """Drop complete feature maps in a channels-first or channels-last array."""

    def __init__(
        self,
        p: float = 0.5,
        *,
        channel_axis: int = -1,
        batch_axis: int | None = 0,
        rngs: Rngs | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        super().__init__(p, rngs=rngs, shard_mode=shard_mode)
        self.channel_axis = channel_axis
        self.batch_axis = batch_axis

    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        output = self._apply(
            x,
            broadcast_axes=_feature_broadcast_axes(
                x.ndim,
                self.channel_axis,
                self.batch_axis,
            ),
        )
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'p={self.p:g}, channel_axis={self.channel_axis}, '
            f'batch_axis={self.batch_axis}'
        )


class AlphaDropout(Module):
    """SELU-compatible dropout preserving zero mean and unit variance."""

    _alpha_prime = -1.7580993408473766

    def __init__(
        self,
        p: float = 0.5,
        *,
        broadcast_axes: Axes = (),
        rngs: Rngs | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.p = _validate_probability(p, allow_one=False)
        self.broadcast_axes = (
            (broadcast_axes,)
            if isinstance(broadcast_axes, int)
            else tuple(broadcast_axes)
        )
        self.rngs = _validate_rngs(rngs)
        self.shard_mode = shard_mode

    def _apply(
        self,
        x: jax.Array,
        *,
        broadcast_axes: tuple[int, ...],
    ) -> jax.Array:
        if not self.training or self.p == 0:
            return x
        if not jnp.issubdtype(x.dtype, jnp.floating):
            raise TypeError('AlphaDropout requires a floating-point input')

        keep_probability = 1.0 - self.p
        mask = jax.random.bernoulli(
            _next_key(self.rngs),
            p=keep_probability,
            shape=_mask_shape(x.shape, broadcast_axes),
        )
        alpha_prime = jnp.asarray(self._alpha_prime, dtype=x.dtype)
        scale = jax.lax.rsqrt(
            jnp.asarray(
                keep_probability
                * (1.0 + self.p * self._alpha_prime ** 2),
                dtype=x.dtype,
            )
        )
        bias = -scale * alpha_prime * self.p
        output = jnp.where(mask, x, alpha_prime)
        return scale * output + bias

    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        axes = _canonical_axes(
            self.broadcast_axes,
            x.ndim,
            name='broadcast_axes',
            allow_empty=True,
        )
        output = self._apply(
            x,
            broadcast_axes=axes,
        )
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return f'p={self.p:g}, broadcast_axes={self.broadcast_axes}'


class FeatureAlphaDropout(AlphaDropout):
    """SELU-compatible dropout sharing a mask across spatial dimensions."""

    def __init__(
        self,
        p: float = 0.5,
        *,
        channel_axis: int = -1,
        batch_axis: int | None = 0,
        rngs: Rngs | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        super().__init__(p, rngs=rngs, shard_mode=shard_mode)
        self.channel_axis = channel_axis
        self.batch_axis = batch_axis

    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        output = self._apply(
            x,
            broadcast_axes=_feature_broadcast_axes(
                x.ndim,
                self.channel_axis,
                self.batch_axis,
            ),
        )
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'p={self.p:g}, channel_axis={self.channel_axis}, '
            f'batch_axis={self.batch_axis}'
        )


class StochasticDepth(Dropout):
    """Randomly drop complete residual branches per batch or per row."""

    def __init__(
        self,
        p: float,
        mode: StochasticDepthMode = 'row',
        *,
        batch_axis: int = 0,
        rngs: Rngs | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        super().__init__(p, rngs=rngs, shard_mode=shard_mode)
        if mode not in {'batch', 'row'}:
            raise ValueError("mode must be 'batch' or 'row'")
        self.mode = mode
        self.batch_axis = batch_axis

    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        if self.mode == 'batch':
            broadcast_axes = tuple(range(x.ndim))
        else:
            batch_axis = _canonical_axis(
                self.batch_axis,
                x.ndim,
                name='batch_axis',
            )
            broadcast_axes = tuple(
                axis for axis in range(x.ndim) if axis != batch_axis
            )
        output = self._apply(
            x,
            broadcast_axes=broadcast_axes,
        )
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return f'p={self.p:g}, mode={self.mode!r}, batch_axis={self.batch_axis}'


__all__ = [
    'StochasticDepthMode',
    'Dropout',
    'FeatureDropout',
    'AlphaDropout',
    'FeatureAlphaDropout',
    'StochasticDepth',
]
