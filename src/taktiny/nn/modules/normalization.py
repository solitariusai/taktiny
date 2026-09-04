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

import math
from itertools import chain

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jax.nn import initializers
from jax.sharding import PartitionSpec

from taktiny.nn.base import Module, Parameter
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import (
    _canonical_axes,
    _constrain,
    _normalize_shape,
    _validate_positive_float,
)
from taktiny.utils.quantization import resolve_quantization_rule
from taktiny.utils.spmd import with_logical_partitioning
from taktiny.utils.typing import (
    Axes,
    AxisNames,
    DType,
    GenericShape,
    Initializer,
    MetaData,
    QuantConfig,
)


def _default_axes(rank: int) -> tuple[int, ...]:
    return tuple(range(-rank, 0))


def _as_axes(axes: Axes, name: str) -> tuple[int, ...]:
    values = (axes,) if isinstance(axes, int) else tuple(axes)
    if not all(
        isinstance(axis, int) and not isinstance(axis, bool)
        for axis in values
    ):
        raise TypeError(f'{name} must contain only integers')
    return values


def _feature_axes(
    axes: Axes | None,
    feature_rank: int,
    name: str,
) -> tuple[int, ...]:
    values = _default_axes(feature_rank) if axes is None else _as_axes(axes, name)
    if len(values) != feature_rank:
        raise ValueError(
            f'{name} and num_features must have the same number of dimensions'
        )
    return values


def _parameter_axis_names(
    axis_names: AxisNames | None,
    rank: int,
) -> AxisNames | None:
    if axis_names is None:
        return None
    names = tuple(axis_names)
    if len(names) != rank:
        raise ValueError(
            f'axis_names length {len(names)} must match feature rank {rank}'
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
            f'expected feature dimensions {shape} on axes {axes}, '
            f'got {actual} from shape {x.shape}'
        )


def _broadcast_parameter(
    parameter: jax.Array,
    ndim: int,
    axes: tuple[int, ...],
) -> jax.Array:
    """Broadcast a parameter whose dimensions follow declared axis order."""
    dimension_order = tuple(sorted(range(len(axes)), key=axes.__getitem__))
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


def _validate_quant(quant: QuantConfig) -> None:
    if quant is not None and resolve_quantization_rule(
        quant,
        '',
        op_name='normalization',
    ) is not None:
        raise NotImplementedError(
            'quantized normalization parameters are not supported'
        )


def _new_parameter(
    initializer: Initializer,
    rngs: Rngs,
    shape: tuple[int, ...],
    dtype: DType | None,
    axis_names: AxisNames | None,
    partition_spec: PartitionSpec | None,
    metadata: MetaData | None = None,
    *,
    trainable: bool = True,
) -> Parameter:
    if axis_names is not None or partition_spec is not None:
        initializer = with_logical_partitioning(
            initializer,
            axis_names,
            partition_spec,
        )
    return Parameter(
        initializer(rngs(), shape, dtype),
        trainable=trainable,
        axis_names=axis_names,
        partition_spec=partition_spec,
        metadata=metadata,
    )


def _feature_statistics(
    value: jax.Array,
    feature_axes: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Return statistics with dimensions in declared feature-axis order."""
    reduction_axes = tuple(
        axis for axis in range(value.ndim) if axis not in feature_axes
    )
    if not reduction_axes:
        raise ValueError('BatchNorm requires at least one reduction axis')
    permutation = reduction_axes + feature_axes
    transposed = jnp.transpose(value, permutation)
    leading_axes = tuple(range(len(reduction_axes)))
    return (
        jnp.mean(transposed, axis=leading_axes),
        jnp.var(transposed, axis=leading_axes),
    )


def _shape_repr(shape: tuple[int, ...]) -> str:
    return '×'.join(map(str, shape))


class LayerNorm(Module):
    """Apply layer normalization over selected feature dimensions.

    For every independent slice of the input, LayerNorm computes the mean and
    variance over ``axes`` and applies

    ``output = (x - mean) / sqrt(variance + epsilon) * scale + bias``.

    Unlike BatchNorm, its statistics do not depend on other samples and it
    performs the same calculation in training and evaluation modes.
    ``num_features`` declares the sizes of the normalized dimensions in the
    same order as ``axes``. If ``axes`` is omitted, the final
    ``len(num_features)`` dimensions are normalized. For example,
    ``num_features=(3, 4)`` expects trailing dimensions of shape ``(3, 4)``.

    Statistics for float16 and bfloat16 inputs are computed in float32 for
    numerical stability. The result is converted back to the input dtype.

    Args:
        num_features: Size of every normalized input dimension. An integer is
            treated as a one-dimensional feature shape.
        epsilon: Positive value added to the variance before the reciprocal
            square root. Defaults to ``1e-6``.
        bias: Whether to create an additive bias when affine parameters are
            enabled. Defaults to ``True``.
        dtype: Dtype passed to the parameter initializers. Defaults to each
            initializer's default dtype.
        rngs: Random stream used by parameter initializers. A deterministic
            stream seeded with zero is used when omitted.
        elementwise_affine: Whether to create a learnable scale and optional
            bias with shape ``num_features``. Defaults to ``True``.
        axes: Input dimensions corresponding to ``num_features``, in the same
            order. Negative dimensions are supported. Defaults to the trailing
            feature dimensions.
        scale_initializer: Initializer for the multiplicative scale.
            Defaults to ones.
        bias_initializer: Initializer for the additive bias.
            Defaults to zeros.
        quant: Optional Qwix configuration. Rules for unrelated operation
            types are ignored; quantized normalization parameters are not yet
            supported.
        axis_names: Logical names for all ``num_features`` dimensions.
        partition_spec: Explicit partition specification for affine
            parameters. A mapping obtained from ``axis_names`` overrides this
            specification.
        scale_metadata: Metadata attached to the scale parameter.
        bias_metadata: Metadata attached to the bias parameter.

    Attributes:
        num_features: Normalized feature shape as a tuple.
        axes: Input dimensions normalized by this module.
        scale: Learnable multiplicative parameter, or ``None`` when
            ``elementwise_affine=False``.
        bias: Learnable additive parameter, or ``None`` when disabled.

    Example:
        Normalize the final two dimensions independently for each item in the
        leading batch dimension:

        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> layer = nn.LayerNorm((3, 4))
        >>> x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
        >>> layer(x).shape
        (2, 3, 4)

    References:
        Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton,
        "Layer Normalization" (2016).
        https://arxiv.org/abs/1607.06450
    """

    def __init__(
        self,
        num_features: GenericShape,
        epsilon: float = 1e-6,
        *,
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs | None = None,
        elementwise_affine: bool = True,
        axes: GenericShape | None = None,
        scale_initializer: Initializer = initializers.ones,
        bias_initializer: Initializer = initializers.zeros,
        quant: QuantConfig = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        scale_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
    ) -> None:
        _validate_quant(quant)
        self.num_features = _normalize_shape(num_features, 'num_features')
        self.epsilon = _validate_positive_float(epsilon, 'epsilon')
        self.elementwise_affine = bool(elementwise_affine)
        self.axes = _feature_axes(axes, len(self.num_features), 'axes')
        names = _parameter_axis_names(axis_names, len(self.num_features))
        rngs = Rngs(0) if rngs is None else rngs

        self.scale = None
        self.bias = None
        if self.elementwise_affine:
            self.scale = _new_parameter(
                scale_initializer,
                rngs,
                self.num_features,
                dtype,
                names,
                partition_spec,
                scale_metadata,
            )
            if bias:
                self.bias = _new_parameter(
                    bias_initializer,
                    rngs,
                    self.num_features,
                    dtype,
                    names,
                    partition_spec,
                    bias_metadata,
                )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Apply layer normalization."""
        x = jnp.asarray(x)
        _validate_floating(x)
        axes = _canonical_axes(self.axes, x.ndim)
        _validate_input_shape(x, self.num_features, axes)
        value = _statistics_value(x)
        mean = jnp.mean(value, axis=axes, keepdims=True)
        variance = jnp.var(value, axis=axes, keepdims=True)
        output = (value - mean) * jax.lax.rsqrt(variance + self.epsilon)
        if self.scale is not None:
            output *= _broadcast_parameter(
                self.scale.value.astype(value.dtype), x.ndim, axes
            )
        if self.bias is not None:
            output += _broadcast_parameter(
                self.bias.value.astype(value.dtype), x.ndim, axes
            )
        return _constrain(output.astype(x.dtype), out_sharding)

    def extra_repr(self) -> str:
        return (
            f'{_shape_repr(self.num_features)}, epsilon={self.epsilon:g}, '
            f'affine={self.elementwise_affine}'
        )


class RMSNorm(Module):
    """Apply root mean square normalization over selected dimensions.

    RMSNorm scales each independent input slice using its root mean square:

    ``output = x / sqrt(mean(x ** 2) + epsilon) * scale + bias``.

    It does not subtract the mean. Consequently, RMSNorm provides re-scaling
    invariance without the re-centering step performed by LayerNorm. The
    statistics are local to each input slice, so training and evaluation modes
    behave identically.

    ``num_features`` gives the sizes of the normalized dimensions in the same
    order as ``axes``. When ``axes`` is omitted, the final
    ``len(num_features)`` dimensions are used. Statistics for float16 and
    bfloat16 inputs are computed in float32, after which the result is cast
    back to the input dtype.

    Args:
        num_features: Size of every normalized input dimension. An integer is
            treated as a one-dimensional feature shape.
        epsilon: Positive value added to the mean square before the reciprocal
            square root. Defaults to ``1e-6``.
        bias: Whether to create an additive bias when affine parameters are
            enabled. Defaults to ``False``.
        dtype: Dtype passed to the parameter initializers. Defaults to each
            initializer's default dtype.
        rngs: Random stream used by parameter initializers. A deterministic
            stream seeded with zero is used when omitted.
        elementwise_affine: Whether to create a learnable scale and optional
            bias with shape ``num_features``. Defaults to ``True``.
        axes: Input dimensions corresponding to ``num_features``, in the same
            order. Negative dimensions are supported. Defaults to the trailing
            feature dimensions.
        scale_initializer: Initializer for the multiplicative scale.
            Defaults to ones.
        bias_initializer: Initializer for the additive bias.
            Defaults to zeros.
        quant: Optional Qwix configuration. Rules for unrelated operation
            types are ignored; quantized normalization parameters are not yet
            supported.
        axis_names: Logical names for all ``num_features`` dimensions.
        partition_spec: Explicit partition specification for affine
            parameters. A mapping obtained from ``axis_names`` overrides this
            specification.
        scale_metadata: Metadata attached to the scale parameter.
        bias_metadata: Metadata attached to the bias parameter.

    Attributes:
        num_features: Normalized feature shape as a tuple.
        axes: Input dimensions normalized by this module.
        scale: Learnable multiplicative parameter, or ``None`` when
            ``elementwise_affine=False``.
        bias: Learnable additive parameter, or ``None`` when disabled.

    Example:
        Normalize the last feature dimension without adding a bias:

        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> layer = nn.RMSNorm(4)
        >>> x = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        >>> layer(x).shape
        (2, 4)

    References:
        Biao Zhang and Rico Sennrich,
        "Root Mean Square Layer Normalization" (2019).
        https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        num_features: GenericShape,
        epsilon: float = 1e-6,
        *,
        bias: bool = False,
        dtype: DType | None = None,
        rngs: Rngs | None = None,
        elementwise_affine: bool = True,
        axes: GenericShape | None = None,
        scale_initializer: Initializer = initializers.ones,
        bias_initializer: Initializer = initializers.zeros,
        quant: QuantConfig = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        scale_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
    ) -> None:
        _validate_quant(quant)
        self.num_features = _normalize_shape(num_features, 'num_features')
        self.epsilon = _validate_positive_float(epsilon, 'epsilon')
        self.elementwise_affine = bool(elementwise_affine)
        self.axes = _feature_axes(axes, len(self.num_features), 'axes')
        names = _parameter_axis_names(axis_names, len(self.num_features))
        rngs = Rngs(0) if rngs is None else rngs

        self.scale = None
        self.bias = None
        if self.elementwise_affine:
            self.scale = _new_parameter(
                scale_initializer,
                rngs,
                self.num_features,
                dtype,
                names,
                partition_spec,
                scale_metadata,
            )
            if bias:
                self.bias = _new_parameter(
                    bias_initializer,
                    rngs,
                    self.num_features,
                    dtype,
                    names,
                    partition_spec,
                    bias_metadata,
                )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Apply root mean square normalization."""
        x = jnp.asarray(x)
        _validate_floating(x)
        axes = _canonical_axes(self.axes, x.ndim)
        _validate_input_shape(x, self.num_features, axes)
        value = _statistics_value(x)
        mean_square = jnp.mean(jnp.square(value), axis=axes, keepdims=True)
        output = value * jax.lax.rsqrt(mean_square + self.epsilon)
        if self.scale is not None:
            output *= _broadcast_parameter(
                self.scale.value.astype(value.dtype), x.ndim, axes
            )
        if self.bias is not None:
            output += _broadcast_parameter(
                self.bias.value.astype(value.dtype), x.ndim, axes
            )
        return _constrain(output.astype(x.dtype), out_sharding)

    def extra_repr(self) -> str:
        return (
            f'{_shape_repr(self.num_features)}, epsilon={self.epsilon:g}, '
            f'affine={self.elementwise_affine}'
        )


class BatchNorm(Module):
    """Normalize feature dimensions using mini-batch statistics.

    BatchNorm retains the dimensions selected by ``axes`` and computes a mean
    and variance by reducing every other input dimension. Its transformation
    is

    ``output = (x - mean) / sqrt(variance + epsilon) * scale + bias``.

    ``num_features`` may contain one or more dimensions. Their sizes and order
    must match ``axes`` and determine the shapes of the affine parameters and
    running statistics. If ``axes`` is omitted, the final
    ``len(num_features)`` input dimensions are treated as features.

    Mode is controlled by :attr:`Module.is_training`, which is changed through
    :meth:`Module.train` and :meth:`Module.eval`. During training, current
    batch statistics normalize the input. If ``track_running_stats=True``, the
    same call also updates ``running_mean``, ``running_var``, and
    ``num_batches_tracked``. During evaluation, the stored statistics are used.
    When tracking is disabled, current input statistics are used in both modes.

    Running-state mutation is an eager side effect and is therefore rejected
    while tracing with ``jax.jit``. Use evaluation mode or set
    ``track_running_stats=False`` for a directly jitted call. Float16 and
    bfloat16 statistics are accumulated in float32 before the output is cast
    back to its input dtype.

    Args:
        num_features: Size of every retained feature dimension. An integer is
            treated as a one-dimensional feature shape.
        epsilon: Positive value added to the variance before the reciprocal
            square root. Defaults to ``1e-6``.
        bias: Whether to create an additive bias when affine parameters are
            enabled. Defaults to ``False``.
        dtype: Dtype used for affine parameters and running statistics.
            Defaults to float32 through the default initializers.
        rngs: Random stream used by parameter initializers. A deterministic
            stream seeded with zero is used when omitted.
        elementwise_affine: Whether to create a learnable scale and optional
            bias with shape ``num_features``. Defaults to ``True``.
        momentum: Weight assigned to the newest batch statistics when updating
            running state. ``None`` selects a cumulative moving average.
            Defaults to ``0.1``.
        track_running_stats: Whether to maintain statistics for evaluation.
            Defaults to ``True``.
        axes: Input feature dimensions retained by the statistics, in the same
            order as ``num_features``. Every remaining dimension is reduced.
            Defaults to the trailing feature dimensions.
        scale_initializer: Initializer for the multiplicative scale.
            Defaults to ones.
        bias_initializer: Initializer for the additive bias.
            Defaults to zeros.
        quant: Optional Qwix configuration. Rules for unrelated operation
            types are ignored; quantized normalization parameters are not yet
            supported.
        axis_names: Logical names for all feature and running-state dimensions.
        partition_spec: Explicit partition specification for affine parameters
            and running statistics. A mapping obtained from ``axis_names``
            overrides this specification.
        scale_metadata: Metadata attached to the scale parameter.
        bias_metadata: Metadata attached to the bias parameter.

    Attributes:
        num_features: Retained feature shape as a tuple.
        axes: Input dimensions treated as features.
        scale: Learnable multiplicative parameter, or ``None`` when affine
            parameters are disabled.
        bias: Learnable additive parameter, or ``None`` when disabled.
        running_mean: Non-trainable running mean, or ``None`` when tracking is
            disabled.
        running_var: Non-trainable running variance, or ``None`` when tracking
            is disabled.
        num_batches_tracked: Number of eager training batches incorporated into
            the running statistics, or ``None`` when tracking is disabled.

    Example:
        Train once to record statistics, then switch to evaluation mode:

        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> layer = nn.BatchNorm(4, momentum=1.0)
        >>> x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
        >>> training_output = layer(x)
        >>> _ = layer.eval()
        >>> evaluation_output = layer(x)
        >>> training_output.shape == evaluation_output.shape
        True

    References:
        Sergey Ioffe and Christian Szegedy,
        "Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift" (2015).
        https://arxiv.org/abs/1502.03167
    """

    def __init__(
        self,
        num_features: GenericShape,
        epsilon: float = 1e-6,
        *,
        bias: bool = False,
        dtype: DType | None = None,
        rngs: Rngs | None = None,
        elementwise_affine: bool = True,
        momentum: float | None = 0.1,
        track_running_stats: bool = True,
        axes: GenericShape | None = None,
        scale_initializer: Initializer = initializers.ones,
        bias_initializer: Initializer = initializers.zeros,
        quant: QuantConfig = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        scale_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
    ) -> None:
        _validate_quant(quant)
        self.num_features = _normalize_shape(num_features, 'num_features')
        self.epsilon = _validate_positive_float(epsilon, 'epsilon')
        if momentum is not None:
            if (
                isinstance(momentum, bool)
                or not isinstance(momentum, (int, float))
                or not math.isfinite(momentum)
                or not 0 <= momentum <= 1
            ):
                raise ValueError('momentum must be None or between 0 and 1')
            momentum = float(momentum)
        self.momentum = momentum
        self.elementwise_affine = bool(elementwise_affine)
        self.track_running_stats = bool(track_running_stats)
        self.axes = _feature_axes(axes, len(self.num_features), 'axes')
        names = _parameter_axis_names(axis_names, len(self.num_features))
        rngs = Rngs(0) if rngs is None else rngs

        self.scale = None
        self.bias = None
        if self.elementwise_affine:
            self.scale = _new_parameter(
                scale_initializer,
                rngs,
                self.num_features,
                dtype,
                names,
                partition_spec,
                scale_metadata,
            )
            if bias:
                self.bias = _new_parameter(
                    bias_initializer,
                    rngs,
                    self.num_features,
                    dtype,
                    names,
                    partition_spec,
                    bias_metadata,
                )

        self.running_mean = None
        self.running_var = None
        self.num_batches_tracked = None
        if self.track_running_stats:
            state_dtype = jnp.float32 if dtype is None else dtype
            self.running_mean = _new_parameter(
                initializers.zeros,
                rngs,
                self.num_features,
                state_dtype,
                names,
                partition_spec,
                trainable=False,
            )
            self.running_var = _new_parameter(
                initializers.ones,
                rngs,
                self.num_features,
                state_dtype,
                names,
                partition_spec,
                trainable=False,
            )
            self.num_batches_tracked = Parameter(
                jnp.asarray(0, dtype=jnp.int32),
                trainable=False,
            )

    def statistics(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute batch mean and variance in feature-axis order."""
        x = jnp.asarray(x)
        _validate_floating(x)
        axes = _canonical_axes(self.axes, x.ndim)
        _validate_input_shape(x, self.num_features, axes)
        return _feature_statistics(_statistics_value(x), axes)

    def update_running_stats(
        self,
        mean: jax.Array,
        variance: jax.Array,
    ) -> None:
        """Update tracked statistics from an eager training call."""
        if not self.track_running_stats:
            return
        assert self.running_mean is not None
        assert self.running_var is not None
        assert self.num_batches_tracked is not None
        if isinstance(mean, Tracer) or isinstance(variance, Tracer):
            raise TypeError(
                'BatchNorm running statistics cannot be mutated while tracing; '
                'use eval mode or track_running_stats=False under jax.jit'
            )
        if mean.shape != self.num_features or variance.shape != self.num_features:
            raise ValueError(
                f'expected statistics with shape {self.num_features}, got '
                f'{mean.shape} and {variance.shape}'
            )

        count = int(self.num_batches_tracked.value) + 1
        factor = (1.0 / count) if self.momentum is None else self.momentum
        self.running_mean._value = (
            (1.0 - factor) * self.running_mean.value
            + factor * mean.astype(self.running_mean.value.dtype)
        )
        self.running_var._value = (
            (1.0 - factor) * self.running_var.value
            + factor * variance.astype(self.running_var.value.dtype)
        )
        self.num_batches_tracked._value = jnp.asarray(count, dtype=jnp.int32)

    def reset_running_stats(self) -> None:
        """Reset tracked statistics to mean zero and variance one."""
        if not self.track_running_stats:
            return
        assert self.running_mean is not None
        assert self.running_var is not None
        assert self.num_batches_tracked is not None
        self.running_mean._value = jnp.zeros_like(self.running_mean.value)
        self.running_var._value = jnp.ones_like(self.running_var.value)
        self.num_batches_tracked._value = jnp.asarray(0, dtype=jnp.int32)

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Apply batch normalization using the module's current mode."""
        x = jnp.asarray(x)
        _validate_floating(x)
        axes = _canonical_axes(self.axes, x.ndim)
        _validate_input_shape(x, self.num_features, axes)
        value = _statistics_value(x)

        if self.is_training or not self.track_running_stats:
            mean, variance = _feature_statistics(value, axes)
            if self.is_training and self.track_running_stats:
                self.update_running_stats(mean, variance)
        else:
            assert self.running_mean is not None
            assert self.running_var is not None
            mean = self.running_mean.value.astype(value.dtype)
            variance = self.running_var.value.astype(value.dtype)

        mean = _broadcast_parameter(mean, x.ndim, axes)
        variance = _broadcast_parameter(variance, x.ndim, axes)
        output = (value - mean) * jax.lax.rsqrt(variance + self.epsilon)
        if self.scale is not None:
            output *= _broadcast_parameter(
                self.scale.value.astype(value.dtype), x.ndim, axes
            )
        if self.bias is not None:
            output += _broadcast_parameter(
                self.bias.value.astype(value.dtype), x.ndim, axes
            )
        return _constrain(output.astype(x.dtype), out_sharding)

    def extra_repr(self) -> str:
        return (
            f'{_shape_repr(self.num_features)}, epsilon={self.epsilon:g}, '
            f'momentum={self.momentum}, affine={self.elementwise_affine}'
        )


class GroupNorm(Module):
    """Normalize grouped channels independently of the batch size.

    GroupNorm divides each channel dimension into a corresponding number of
    groups. For every sample and Cartesian combination of channel groups, it
    computes mean and variance across the within-group channel dimensions and
    every spatial dimension:

    ``output = (x - mean) / sqrt(variance + epsilon) * scale + bias``.

    ``num_channels`` and ``num_groups`` may both be N-D shapes. They must have
    equal rank, and every ``num_channels[i]`` must be divisible by
    ``num_groups[i]``. Their dimensions correspond in order to
    ``channel_axes``. With N-D channels and the default ``channel_axes=-1``,
    the final ``len(num_channels)`` dimensions are selected automatically.

    Dimensions in ``batch_axes`` are kept independent and never contribute to
    the statistics. The default is the leading dimension ``0``. Pass an empty
    tuple for unbatched input. Because GroupNorm never aggregates across batch
    dimensions and stores no running statistics, training and evaluation modes
    behave identically. Float16 and bfloat16 statistics are computed in
    float32, and the result is cast back to the input dtype.

    Args:
        num_groups: Number of groups used to split each channel dimension. An
            integer is treated as a one-dimensional group shape.
        num_channels: Size of every channel dimension. It must have the same
            rank as ``num_groups``.
        epsilon: Positive value added to the variance before the reciprocal
            square root. Defaults to ``1e-6``.
        bias: Whether to create an additive bias when affine parameters are
            enabled. Defaults to ``False``.
        dtype: Dtype passed to the parameter initializers. Defaults to each
            initializer's default dtype.
        rngs: Random stream used by parameter initializers. A deterministic
            stream seeded with zero is used when omitted.
        elementwise_affine: Whether to create a learnable scale and optional
            bias with shape ``num_channels``. Defaults to ``True``.
        channel_axes: Input dimensions corresponding to ``num_channels``, in
            the same order. Defaults to the trailing channel dimensions.
        batch_axes: Input dimensions kept independent during normalization.
            Defaults to the leading dimension ``0``. Use ``()`` for unbatched
            input.
        scale_initializer: Initializer for the multiplicative scale.
            Defaults to ones.
        bias_initializer: Initializer for the additive bias.
            Defaults to zeros.
        quant: Optional Qwix configuration. Rules for unrelated operation
            types are ignored; quantized normalization parameters are not yet
            supported.
        axis_names: Logical names for all ``num_channels`` dimensions.
        partition_spec: Explicit partition specification for affine
            parameters. A mapping obtained from ``axis_names`` overrides this
            specification.
        scale_metadata: Metadata attached to the scale parameter.
        bias_metadata: Metadata attached to the bias parameter.

    Attributes:
        num_groups: Group shape as a tuple.
        num_channels: Channel shape as a tuple.
        channel_axes: Input dimensions containing channels.
        batch_axes: Input dimensions kept independent.
        scale: Learnable multiplicative parameter, or ``None`` when affine
            parameters are disabled.
        bias: Learnable additive parameter, or ``None`` when disabled.

    Example:
        Normalize a tensor with two structured channel dimensions. The final
        dimensions ``(4, 6)`` are divided into ``(2, 2)`` groups:

        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> layer = nn.GroupNorm((2, 2), (4, 6))
        >>> x = jnp.ones((2, 5, 4, 6), dtype=jnp.float32)
        >>> layer(x).shape
        (2, 5, 4, 6)

    References:
        Yuxin Wu and Kaiming He, "Group Normalization" (2018).
        https://arxiv.org/abs/1803.08494
    """

    def __init__(
        self,
        num_groups: GenericShape,
        num_channels: GenericShape,
        epsilon: float = 1e-6,
        *,
        bias: bool = False,
        dtype: DType | None = None,
        rngs: Rngs | None = None,
        elementwise_affine: bool = True,
        channel_axes: Axes = -1,
        batch_axes: Axes = 0,
        scale_initializer: Initializer = initializers.ones,
        bias_initializer: Initializer = initializers.zeros,
        quant: QuantConfig = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        scale_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
    ) -> None:
        _validate_quant(quant)
        self.num_groups = _normalize_shape(num_groups, 'num_groups')
        self.num_channels = _normalize_shape(num_channels, 'num_channels')
        if len(self.num_groups) != len(self.num_channels):
            raise ValueError(
                'num_groups and num_channels must have the same number of dimensions'
            )
        for index, (groups, channels) in enumerate(
            zip(self.num_groups, self.num_channels)
        ):
            if channels % groups:
                raise ValueError(
                    f'num_channels[{index}] must be divisible by num_groups[{index}]'
                )

        self.epsilon = _validate_positive_float(epsilon, 'epsilon')
        self.elementwise_affine = bool(elementwise_affine)
        if (
            isinstance(channel_axes, int)
            and channel_axes == -1
            and len(self.num_channels) > 1
        ):
            self.channel_axes = _default_axes(len(self.num_channels))
        else:
            self.channel_axes = _as_axes(channel_axes, 'channel_axes')
        if len(self.channel_axes) != len(self.num_channels):
            raise ValueError(
                'channel_axes and num_channels must have the same number of dimensions'
            )
        self.batch_axes = _as_axes(batch_axes, 'batch_axes')

        names = _parameter_axis_names(axis_names, len(self.num_channels))
        rngs = Rngs(0) if rngs is None else rngs
        self.scale = None
        self.bias = None
        if self.elementwise_affine:
            self.scale = _new_parameter(
                scale_initializer,
                rngs,
                self.num_channels,
                dtype,
                names,
                partition_spec,
                scale_metadata,
            )
            if bias:
                self.bias = _new_parameter(
                    bias_initializer,
                    rngs,
                    self.num_channels,
                    dtype,
                    names,
                    partition_spec,
                    bias_metadata,
                )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Apply group normalization."""
        x = jnp.asarray(x)
        _validate_floating(x)
        channel_axes = _canonical_axes(self.channel_axes, x.ndim)
        batch_axes = _canonical_axes(
            self.batch_axes,
            x.ndim,
            name='batch_axes',
            allow_empty=True,
        )
        if set(channel_axes) & set(batch_axes):
            raise ValueError('channel_axes and batch_axes must not overlap')
        _validate_input_shape(x, self.num_channels, channel_axes)

        spatial_axes = tuple(
            axis
            for axis in range(x.ndim)
            if axis not in batch_axes and axis not in channel_axes
        )
        permutation = batch_axes + spatial_axes + channel_axes
        transposed = jnp.transpose(_statistics_value(x), permutation)
        prefix_shape = transposed.shape[:-len(self.num_channels)]
        split_channels = tuple(
            chain.from_iterable(
                (groups, channels // groups)
                for groups, channels in zip(self.num_groups, self.num_channels)
            )
        )
        grouped = transposed.reshape(prefix_shape + split_channels)

        prefix_rank = len(prefix_shape)
        spatial_positions = tuple(range(len(batch_axes), prefix_rank))
        within_group_positions = tuple(
            prefix_rank + 2 * index + 1
            for index in range(len(self.num_channels))
        )
        reduction_axes = spatial_positions + within_group_positions
        mean = jnp.mean(grouped, axis=reduction_axes, keepdims=True)
        variance = jnp.var(grouped, axis=reduction_axes, keepdims=True)
        normalized = (
            (grouped - mean) * jax.lax.rsqrt(variance + self.epsilon)
        ).reshape(transposed.shape)
        inverse_permutation = tuple(
            sorted(range(x.ndim), key=permutation.__getitem__)
        )
        output = jnp.transpose(normalized, inverse_permutation)

        if self.scale is not None:
            output *= _broadcast_parameter(
                self.scale.value.astype(output.dtype), x.ndim, channel_axes
            )
        if self.bias is not None:
            output += _broadcast_parameter(
                self.bias.value.astype(output.dtype), x.ndim, channel_axes
            )
        return _constrain(output.astype(x.dtype), out_sharding)

    def extra_repr(self) -> str:
        return (
            f'{_shape_repr(self.num_groups)} groups, '
            f'{_shape_repr(self.num_channels)} channels, '
            f'epsilon={self.epsilon:g}'
        )


__all__ = [
    'BatchNorm',
    'GroupNorm',
    'LayerNorm',
    'RMSNorm',
]
