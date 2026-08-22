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
"""Rank-generic normalization modules."""
from __future__ import annotations
from collections.abc import Sequence
import math
import jax
import jax.numpy as jnp
from jax.core import Tracer

from taktiny.nn.module import Module, Parameter
from taktiny.nn.continuo import (
    _canonical_axis,
    _canonical_axes,
    _constrain,
    _normalize_shape,
    _resolve_training,
    _validate_integer,
    _validate_positive_float,
)
from taktiny.utils.typing import (
    Axes,
    AxisNames,
    DType,
    Initializer,
    MeshAxisName,
    ShardMode,
)

def _default_axes(rank: int) -> tuple[int, ...]:
    return tuple(range(-rank, 0))


def _parameter_axis_names(
    axis_names: AxisNames | None,
    rank: int,
) -> AxisNames | None:
    if axis_names is None:
        return None
    names = tuple(axis_names)
    if len(names) != rank:
        raise ValueError(
            f'axis_names length {len(names)} must match normalized rank {rank}'
        )
    return names


def _validate_input_shape(
    x: jax.Array,
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> None:
    actual = tuple(x.shape[axis] for axis in axes)
    if actual != shape:
        raise ValueError(
            f'expected normalized dimensions {shape} on axes {axes}, '
            f'got {actual} from shape {x.shape}'
        )


def _broadcast_parameter(
    parameter: jax.Array,
    ndim: int,
    axes: tuple[int, ...],
) -> jax.Array:
    trailing_axes = tuple(range(ndim - parameter.ndim, ndim))
    if axes == trailing_axes:
        return parameter

    dimension_order = tuple(
        sorted(range(len(axes)), key=axes.__getitem__)
    )
    if dimension_order != tuple(range(len(axes))):
        parameter = jnp.transpose(parameter, dimension_order)
    ordered_axes = tuple(axes[index] for index in dimension_order)
    shape = [1] * ndim
    for axis, size in zip(ordered_axes, parameter.shape):
        shape[axis] = size
    return parameter.reshape(shape)


def _statistics_value(x: jax.Array) -> jax.Array:
    if x.dtype in (jnp.float16, jnp.bfloat16):
        return x.astype(jnp.float32)
    return x


def _validate_floating(x: jax.Array) -> None:
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError('normalization requires a floating-point input')


class LayerNorm(Module):
    """Normalize one or more activation dimensions by mean and variance."""

    def __init__(
        self,
        normalized_shape: int | Sequence[int] | None,
        eps: float = 1e-5,
        *,
        elementwise_affine: bool = True,
        dtype: DType = jnp.float32,
        bias: bool = True,
        axes: Axes | None = None,
        initializer: Initializer = jnp.ones,
        bias_initializer: Initializer = jnp.zeros,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.normalized_shape = (
            None
            if normalized_shape is None
            else _normalize_shape(normalized_shape, 'normalized_shape')
        )
        self.hidden_size = (
            None
            if self.normalized_shape is None
            else (
                self.normalized_shape[0]
                if len(self.normalized_shape) == 1
                else self.normalized_shape
            )
        )
        self.eps = _validate_positive_float(eps, 'eps')
        self.elementwise_affine = bool(elementwise_affine)
        if self.normalized_shape is None and self.elementwise_affine:
            raise ValueError(
                'normalized_shape is required when elementwise_affine=True'
            )
        self.has_bias = self.elementwise_affine and bool(bias)
        requested_axes = (
            (
                -1
                if self.normalized_shape is None
                else _default_axes(len(self.normalized_shape))
            )
            if axes is None
            else axes
        )
        self.axes = (
            (requested_axes,)
            if isinstance(requested_axes, int)
            else tuple(requested_axes)
        )
        if (
            self.normalized_shape is not None
            and len(self.axes) != len(self.normalized_shape)
        ):
            raise ValueError(
                'axes and normalized_shape must have the same number '
                'of dimensions'
            )
        self.shard_mode = shard_mode

        if self.normalized_shape is None and axis_names is not None:
            raise ValueError(
                'axis_names requires a fixed normalized_shape'
            )
        names = (
            None
            if self.normalized_shape is None
            else _parameter_axis_names(
                axis_names,
                len(self.normalized_shape),
            )
        )
        if self.elementwise_affine:
            self.weight = Parameter(
                initializer(self.normalized_shape, dtype=dtype)
            )
            if names is not None:
                self.weight.axis_names = names
            if self.has_bias:
                self.bias = Parameter(
                    bias_initializer(self.normalized_shape, dtype=dtype)
                )
                if names is not None:
                    self.bias.axis_names = names

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        _validate_floating(x)
        axes = _canonical_axes(self.axes, x.ndim)
        if self.normalized_shape is not None:
            _validate_input_shape(x, self.normalized_shape, axes)
        input_dtype = x.dtype
        value = _statistics_value(x)
        mean = jnp.mean(value, axis=axes, keepdims=True)
        variance = jnp.var(value, axis=axes, keepdims=True)
        output = (value - mean) * jax.lax.rsqrt(variance + self.eps)
        if self.elementwise_affine:
            weight = _broadcast_parameter(
                self.weight.value.astype(value.dtype),
                x.ndim,
                axes,
            )
            output = output * weight
            if self.has_bias:
                bias = _broadcast_parameter(
                    self.bias.value.astype(value.dtype),
                    x.ndim,
                    axes,
                )
                output = output + bias
        output = output.astype(input_dtype)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        shape = (
            'dynamic'
            if self.normalized_shape is None
            else 'x'.join(map(str, self.normalized_shape))
        )
        return f'{shape}, eps={self.eps:g}, affine={self.elementwise_affine}'


class RMSNorm(Module):
    """Normalize one or more activation dimensions by root mean square."""

    def __init__(
        self,
        shape: int | Sequence[int] | None,
        epsilon: float = 1e-5,
        *,
        dtype: DType | None = None,
        with_scale: bool = True,
        bias: bool = False,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        initializer: Initializer = jnp.ones,
        bias_initializer: Initializer = jnp.zeros,
        axes: Axes | None = None,
    ) -> None:
        self.normalized_shape = (
            None
            if shape is None
            else _normalize_shape(shape, 'normalized_shape')
        )
        self.eps = _validate_positive_float(epsilon, 'eps')
        self.with_scale = bool(with_scale)
        self.has_bias = bool(bias)
        if self.normalized_shape is None and (self.with_scale or self.has_bias):
            raise ValueError(
                'normalized_shape is required for RMSNorm parameters'
            )
        requested_axes = (
            (
                -1
                if self.normalized_shape is None
                else _default_axes(len(self.normalized_shape))
            )
            if axes is None
            else axes
        )
        self.axes = (
            (requested_axes,)
            if isinstance(requested_axes, int)
            else tuple(requested_axes)
        )
        if (
            self.normalized_shape is not None
            and len(self.axes) != len(self.normalized_shape)
        ):
            raise ValueError(
                'axes and normalized_shape must have the same number '
                'of dimensions'
            )
        self.shard_mode = shard_mode

        if self.normalized_shape is None and axis_names is not None:
            raise ValueError(
                'axis_names requires a fixed normalized_shape'
            )
        names = (
            None
            if self.normalized_shape is None
            else _parameter_axis_names(
                axis_names,
                len(self.normalized_shape),
            )
        )
        dtype = dtype or jnp.float32
        if self.with_scale:
            self.weight = Parameter(
                initializer(self.normalized_shape, dtype=dtype), 
                axis_names=names
            )
        if self.has_bias:
            self.bias = Parameter(
                bias_initializer(self.normalized_shape, dtype=dtype), 
                axis_names=names
            )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        _validate_floating(x)
        axes = _canonical_axes(self.axes, x.ndim)
        if self.normalized_shape is not None:
            _validate_input_shape(x, self.normalized_shape, axes)
        input_dtype = x.dtype
        value = _statistics_value(x)
        variance = jnp.mean(jnp.square(value), axis=axes, keepdims=True)
        output = value * jax.lax.rsqrt(variance + self.eps)
        if self.with_scale:
            weight = _broadcast_parameter(
                self.weight.value.astype(value.dtype),
                x.ndim,
                axes,
            )
            output = output * weight
        if self.has_bias:
            bias = _broadcast_parameter(
                self.bias.value.astype(value.dtype),
                x.ndim,
                axes,
            )
            output = output + bias
        output = output.astype(input_dtype)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        shape = (
            'dynamic'
            if self.normalized_shape is None
            else 'x'.join(map(str, self.normalized_shape))
        )
        return (
            f'{shape}, eps={self.eps:g}, scale={self.with_scale}, '
            f'bias={self.has_bias}'
        )


class BatchNorm(Module):
    """Batch normalization for arrays of arbitrary rank.

    Training uses current-batch statistics. Set ``update_stats=True`` to also
    update the stored running statistics during eager execution. State updates
    are intentionally rejected while tracing; call ``statistics`` inside JIT
    and ``update_running_stats`` with those concrete results outside JIT.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        *,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DType | None = None,
        bias: bool = True,
        channel_axis: int = -1,
        initializer: Initializer = jnp.ones,
        bias_initializer: Initializer = jnp.zeros,
        axis_names: AxisNames | None = None,
        collective_axis_name: MeshAxisName = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.num_features = _validate_integer(num_features, 'num_features')
        self.eps = _validate_positive_float(eps, 'eps')
        if momentum is not None:
            if not math.isfinite(momentum) or not 0 <= momentum <= 1:
                raise ValueError('momentum must be None or between 0 and 1')
            momentum = float(momentum)
        self.momentum = momentum
        self.affine = bool(affine)
        self.has_bias = self.affine and bool(bias)
        self.track_running_stats = bool(track_running_stats)
        self.channel_axis = channel_axis
        self.collective_axis_name = collective_axis_name
        self.shard_mode = shard_mode
        dtype = jnp.float32 if dtype is None else dtype

        names = _parameter_axis_names(axis_names, 1)
        if self.affine:
            self.weight = Parameter(initializer((num_features,), dtype=dtype))
            if names is not None:
                self.weight.axis_names = names
            if self.has_bias:
                self.bias = Parameter(
                    bias_initializer((num_features,), dtype=dtype)
                )
                if names is not None:
                    self.bias.axis_names = names

        if self.track_running_stats:
            self.running_mean = Parameter(
                jnp.zeros((num_features,), dtype=dtype),
                trainable=False,
            )
            self.running_var = Parameter(
                jnp.ones((num_features,), dtype=dtype),
                trainable=False,
            )
            self.num_batches_tracked = Parameter(
                jnp.asarray(0, dtype=jnp.int32),
                trainable=False,
            )
            if names is not None:
                self.running_mean.axis_names = names
                self.running_var.axis_names = names

    def statistics(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return per-channel batch mean and variance without changing state."""
        x = jnp.asarray(x)
        _validate_floating(x)
        channel_axis = _canonical_axis(
            self.channel_axis,
            x.ndim,
            name='channel_axis',
        )
        if x.shape[channel_axis] != self.num_features:
            raise ValueError(
                f'expected {self.num_features} features on axis '
                f'{channel_axis}, got shape {x.shape}'
            )
        reduction_axes = tuple(
            axis for axis in range(x.ndim) if axis != channel_axis
        )
        if not reduction_axes:
            raise ValueError('BatchNorm requires at least one reduction axis')

        value = _statistics_value(x)
        mean = jnp.mean(value, axis=reduction_axes)
        if self.collective_axis_name is not None:
            mean_square = jnp.mean(jnp.square(value), axis=reduction_axes)
            mean = jax.lax.pmean(mean, self.collective_axis_name)
            mean_square = jax.lax.pmean(
                mean_square,
                self.collective_axis_name,
            )
            variance = jnp.maximum(mean_square - jnp.square(mean), 0)
        else:
            variance = jnp.var(value, axis=reduction_axes)
        return mean, variance

    def update_running_stats(
        self,
        mean: jax.Array,
        variance: jax.Array,
    ) -> None:
        """Update stored statistics from concrete, already-computed values."""
        if not self.track_running_stats:
            raise ValueError('running statistics are disabled')

        if isinstance(mean, Tracer) or isinstance(variance, Tracer):
            raise ValueError(
                'BatchNorm running statistics cannot be mutated while tracing'
            )
        if mean.shape != (self.num_features,) or variance.shape != (
            self.num_features,
        ):
            raise ValueError(
                'mean and variance must both have shape '
                f'({self.num_features},)'
            )

        count = int(jax.device_get(self.num_batches_tracked.value)) + 1
        factor = 1.0 / count if self.momentum is None else self.momentum
        mean = mean.astype(self.running_mean.value.dtype)
        variance = variance.astype(self.running_var.value.dtype)
        self.running_mean.value = (
            (1.0 - factor) * self.running_mean.value + factor * mean
        )
        self.running_var.value = (
            (1.0 - factor) * self.running_var.value + factor * variance
        )
        self.num_batches_tracked.value = jnp.asarray(count, dtype=jnp.int32)

    def reset_running_stats(self) -> None:
        """Reset running statistics to their initial values."""
        if not self.track_running_stats:
            raise ValueError('running statistics are disabled')
        self.running_mean.value = jnp.zeros_like(self.running_mean.value)
        self.running_var.value = jnp.ones_like(self.running_var.value)
        self.num_batches_tracked.value = jnp.asarray(0, dtype=jnp.int32)

    def __call__(
        self,
        x: jax.Array,
        *,
        training: bool | None = None,
        update_stats: bool = False,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        _validate_floating(x)
        training = _resolve_training(self.training, training)
        channel_axis = _canonical_axis(
            self.channel_axis,
            x.ndim,
            name='channel_axis',
        )
        if x.shape[channel_axis] != self.num_features:
            raise ValueError(
                f'expected {self.num_features} features on axis '
                f'{channel_axis}, got shape {x.shape}'
            )

        use_batch_stats = training or not self.track_running_stats
        if use_batch_stats:
            mean, variance = self.statistics(x)
            if training and update_stats:
                self.update_running_stats(mean, variance)
        else:
            mean = self.running_mean.value
            variance = self.running_var.value

        input_dtype = x.dtype
        value = _statistics_value(x)
        broadcast_shape = [1] * x.ndim
        broadcast_shape[channel_axis] = self.num_features
        mean = mean.astype(value.dtype).reshape(broadcast_shape)
        variance = variance.astype(value.dtype).reshape(broadcast_shape)
        output = (value - mean) * jax.lax.rsqrt(variance + self.eps)

        if self.affine:
            weight = self.weight.value.astype(value.dtype).reshape(
                broadcast_shape
            )
            output = output * weight
            if self.has_bias:
                bias = self.bias.value.astype(value.dtype).reshape(
                    broadcast_shape
                )
                output = output + bias

        output = output.astype(input_dtype)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, eps={self.eps:g}, '
            f'momentum={self.momentum}, axis={self.channel_axis}'
        )


class GroupNorm(Module):
    """Group normalization for arrays of arbitrary rank and channel layout."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        *,
        affine: bool = True,
        dtype: DType | None = None,
        bias: bool = True,
        channel_axis: int = -1,
        batch_axis: int | None = 0,
        initializer: Initializer = jnp.ones,
        bias_initializer: Initializer = jnp.zeros,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.num_groups = _validate_integer(num_groups, 'num_groups')
        self.num_channels = _validate_integer(num_channels, 'num_channels')
        if self.num_channels % self.num_groups != 0:
            raise ValueError(
                f'num_channels ({num_channels}) must be divisible by '
                f'num_groups ({num_groups})'
            )
        self.eps = _validate_positive_float(eps, 'eps')
        self.affine = bool(affine)
        self.has_bias = self.affine and bool(bias)
        self.channel_axis = channel_axis
        self.batch_axis = batch_axis
        self.shard_mode = shard_mode
        dtype = jnp.float32 if dtype is None else dtype

        names = _parameter_axis_names(axis_names, 1)
        if self.affine:
            self.weight = Parameter(initializer((num_channels,), dtype=dtype))
            if names is not None:
                self.weight.axis_names = names
            if self.has_bias:
                self.bias = Parameter(
                    bias_initializer((num_channels,), dtype=dtype)
                )
                if names is not None:
                    self.bias.axis_names = names

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        _validate_floating(x)
        channel_axis = _canonical_axis(
            self.channel_axis,
            x.ndim,
            name='channel_axis',
        )
        if x.shape[channel_axis] != self.num_channels:
            raise ValueError(
                f'expected {self.num_channels} channels on axis '
                f'{channel_axis}, got shape {x.shape}'
            )

        if self.batch_axis is None:
            batch_axis = None
            permutation = tuple(
                axis for axis in range(x.ndim) if axis != channel_axis
            ) + (channel_axis,)
        else:
            batch_axis = _canonical_axis(
                self.batch_axis,
                x.ndim,
                name='batch_axis',
            )
            if batch_axis == channel_axis:
                raise ValueError('batch_axis and channel_axis must be different')
            permutation = (
                batch_axis,
                *(
                    axis
                    for axis in range(x.ndim)
                    if axis not in (batch_axis, channel_axis)
                ),
                channel_axis,
            )

        value = _statistics_value(x)
        transposed = jnp.transpose(value, permutation)
        if batch_axis is None:
            transposed = transposed[None, ...]
        grouped = transposed.reshape(
            *transposed.shape[:-1],
            self.num_groups,
            self.num_channels // self.num_groups,
        )
        group_axis = grouped.ndim - 2
        reduction_axes = tuple(
            axis
            for axis in range(1, grouped.ndim)
            if axis != group_axis
        )
        mean = jnp.mean(grouped, axis=reduction_axes, keepdims=True)
        variance = jnp.var(grouped, axis=reduction_axes, keepdims=True)
        output = (
            (grouped - mean) * jax.lax.rsqrt(variance + self.eps)
        ).reshape(transposed.shape)
        if batch_axis is None:
            output = output[0]
        inverse_permutation = tuple(
            sorted(range(x.ndim), key=permutation.__getitem__)
        )
        output = jnp.transpose(output, inverse_permutation)

        if self.affine:
            broadcast_shape = [1] * x.ndim
            broadcast_shape[channel_axis] = self.num_channels
            weight = self.weight.value.astype(output.dtype).reshape(
                broadcast_shape
            )
            output = output * weight
            if self.has_bias:
                bias = self.bias.value.astype(output.dtype).reshape(
                    broadcast_shape
                )
                output = output + bias

        output = output.astype(x.dtype)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'{self.num_groups} groups, {self.num_channels} channels, '
            f'eps={self.eps:g}, axis={self.channel_axis}'
        )


__all__ = [
    'LayerNorm',
    'RMSNorm',
    'BatchNorm',
    'GroupNorm',
]
