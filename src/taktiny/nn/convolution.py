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
"""Convolution modules"""
from __future__ import annotations
from collections.abc import Sequence
from itertools import product
import math
import jax
import jax.numpy as jnp
import typing as tp
from jax.nn.initializers import lecun_uniform, zeros
from taktiny import nn
from taktiny.utils.typing import AxisNames, DType, Initializer, ShardMode
from taktiny.nn.utils import (
    _adaptive_pool,
    _as_batched,
    _canonical_padding,
    _constrain,
    _conv_dimension_numbers,
    _max_identity,
    _normalize_adaptive_size,
    _normalize_nonnegative,
    _pool_padding,
    _reduce_window_config,
    _restore_batch,
    _scatter_indices,
    _window_output_shape
)

default_conv_initializer = lecun_uniform()
class Conv(nn.Module):
    """
    N-dimensional channels-last convolution.

    The spatial rank is inferred from ``kernel_size``. For example,
    ``kernel_size=3`` creates a 1D convolution, while ``kernel_size=(3, 3)``
    creates a 2D convolution. Inputs may be batched as
    ``[batch, *spatial, channels]`` or unbatched as ``[*spatial, channels]``.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Kernel extent for each spatial dimension.
        stride: Window stride for each spatial dimension.
        padding: ``"SAME"``, ``"VALID"``, a symmetric integer, one symmetric
            integer per dimension, or explicit ``(low, high)`` pairs.
        dilation: Kernel dilation for each spatial dimension.
        groups: Number of blocked channel groups.
        bias: Whether to add a learned output-channel bias.
        padding_mode: ``"zeros"``, ``"reflect"``, ``"replicate"``, or
            ``"circular"``. Nonzero modes require explicit numeric padding.
        dtype: Parameter dtype.
        rngs: Random stream used for parameter initialization.
        initializer: Weight initializer receiving ``(key, shape, dtype)``.
        bias_initializer: Bias initializer receiving ``(key, shape, dtype)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        dtype: DType = jnp.float32,
        *,
        bias: bool = True,
        rngs: nn.Rngs,
        initializer: Initializer = default_conv_initializer,
        bias_initializer: Initializer = zeros,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError('in_channels must be a positive integer')
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError('out_channels must be a positive integer')
        if not isinstance(groups, int) or groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError(
                f'in_channels ({in_channels}) must be divisible by groups '
                f'({groups})'
            )
        if out_channels % groups != 0:
            raise ValueError(
                f'out_channels ({out_channels}) must be divisible by groups '
                f'({groups})'
            )

        kernel_size = self._normalize_spatial(kernel_size, name='kernel_size')
        spatial_rank = len(kernel_size)
        stride = self._normalize_spatial(
            stride,
            rank=spatial_rank,
            name='stride',
        )
        dilation = self._normalize_spatial(
            dilation,
            rank=spatial_rank,
            name='dilation',
        )
        padding = self._normalize_padding(padding, spatial_rank)

        padding_mode = padding_mode.lower()
        padding_modes = {
            'zeros': 'constant',
            'reflect': 'reflect',
            'replicate': 'edge',
            'circular': 'wrap',
        }
        if padding_mode not in padding_modes:
            choices = ', '.join(padding_modes)
            raise ValueError(
                f'padding_mode must be one of {choices}, got {padding_mode!r}'
            )
        if padding_mode != 'zeros' and isinstance(padding, str):
            raise ValueError(
                'nonzero padding modes require explicit numeric padding'
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode
        self.spatial_rank = spatial_rank
        self.shard_mode = shard_mode

        weight_shape = (
            *kernel_size,
            in_channels // groups,
            out_channels,
        )
        self.weight = nn.Parameter(
            initializer(rngs(), weight_shape, dtype)
        )
        if axis_names is not None:
            if len(axis_names) != len(weight_shape):
                raise ValueError(
                    f'axis_names length {len(axis_names)} must match '
                    f'weight dimensions {len(weight_shape)}'
                )
            self.weight.axis_names = axis_names

        if bias:
            self.bias = nn.Parameter(
                bias_initializer(rngs(), (out_channels,), dtype)
            )
            if axis_names is not None:
                self.bias.axis_names = axis_names[-1:]

    @staticmethod
    def _normalize_spatial(
        value: int | Sequence[int],
        *,
        rank: int | None = None,
        name: str,
    ) -> tuple[int, ...]:
        if isinstance(value, int):
            values = (value,) if rank is None else (value,) * rank
        else:
            values = tuple(value)
        if not values:
            raise ValueError(f'{name} must contain at least one dimension')
        if rank is not None and len(values) != rank:
            raise ValueError(
                f'{name} must contain {rank} values, got {len(values)}'
            )
        if any(not isinstance(item, int) or item <= 0 for item in values):
            raise ValueError(f'{name} values must be positive integers')
        return values

    @staticmethod
    def _normalize_padding(
        padding: str | int | Sequence[int | tuple[int, int]],
        rank: int,
    ) -> str | tuple[tuple[int, int], ...]:
        if isinstance(padding, str):
            padding = padding.upper()
            if padding not in {'SAME', 'VALID'}:
                raise ValueError(
                    "padding must be 'SAME', 'VALID', or explicit integers"
                )
            return padding
        if isinstance(padding, int):
            pairs = ((padding, padding),) * rank
        else:
            values = tuple(padding)
            if rank == 1 and len(values) == 2 and all(
                isinstance(value, int) for value in values
            ):
                pairs = (tp.cast(tuple[int, int], values),)
            elif len(values) == rank and all(
                isinstance(value, int) for value in values
            ):
                pairs = tuple((value, value) for value in values)
            elif len(values) == rank and all(
                isinstance(value, Sequence)
                and len(value) == 2
                and all(isinstance(side, int) for side in value)
                for value in values
            ):
                pairs = tuple(
                    tp.cast(tuple[int, int], tuple(value))
                    for value in values
                )
            else:
                raise ValueError(
                    f'padding must describe {rank} spatial dimensions'
                )
        if any(side < 0 for pair in pairs for side in pair):
            raise ValueError('padding values must be non-negative')
        return pairs

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        expected_batched_rank = self.spatial_rank + 2
        if x.ndim not in {expected_batched_rank - 1, expected_batched_rank}:
            raise ValueError(
                f'expected an unbatched rank-{expected_batched_rank - 1} or '
                f'batched rank-{expected_batched_rank} input, got rank {x.ndim}'
            )
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f'expected {self.in_channels} input channels, got {x.shape[-1]}'
            )

        unbatched = x.ndim == expected_batched_rank - 1
        if unbatched:
            x = x[None, ...]

        padding = self.padding
        if self.padding_mode != 'zeros':
            pad_width = ((0, 0), *padding, (0, 0))
            mode = {
                'reflect': 'reflect',
                'replicate': 'edge',
                'circular': 'wrap',
            }[self.padding_mode]
            x = jnp.pad(x, pad_width, mode=mode)
            padding = 'VALID'

        lhs_spec = (
            0,
            self.spatial_rank + 1,
            *range(1, self.spatial_rank + 1),
        )
        rhs_spec = (
            self.spatial_rank + 1,
            self.spatial_rank,
            *range(self.spatial_rank),
        )
        dimension_numbers = jax.lax.ConvDimensionNumbers(
            lhs_spec,
            rhs_spec,
            lhs_spec,
        )
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight.value,
            window_strides=self.stride,
            padding=padding,
            rhs_dilation=self.dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.groups,
        )
        if self.has_bias:
            output = output + self.bias.value
        if unbatched:
            output = output[0]
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'{self.in_channels} -> {self.out_channels}, '
            f'k={self.kernel_size}, s={self.stride}'
        )

class ConvTranspose(nn.Module):
    """N-dimensional channels-last transposed convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        output_padding: int | Sequence[int] = 0,
        groups: int = 1,
        dilation: int | Sequence[int] = 1,
        padding_mode: str = 'zeros',
        dtype: DType = jnp.float32,
        *,
        bias: bool = True,
        rngs: nn.Rngs,
        initializer: Initializer = default_conv_initializer,
        bias_initializer: Initializer = zeros,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError('in_channels must be a positive integer')
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError('out_channels must be a positive integer')
        if not isinstance(groups, int) or groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError(
                f'in_channels ({in_channels}) must be divisible by groups '
                f'({groups})'
            )
        if out_channels % groups != 0:
            raise ValueError(
                f'out_channels ({out_channels}) must be divisible by groups '
                f'({groups})'
            )
        if padding_mode != 'zeros':
            raise ValueError("ConvTranspose supports only padding_mode='zeros'")
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        stride = Conv._normalize_spatial(stride, rank=rank, name='stride')
        dilation = Conv._normalize_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        padding = Conv._normalize_padding(padding, rank)
        output_padding = _normalize_nonnegative(
            output_padding,
            rank,
            name='output_padding',
        )
        for index, (extra, step, spacing) in enumerate(
            zip(output_padding, stride, dilation)
        ):
            if extra >= step and extra >= spacing:
                raise ValueError(
                    f'output_padding[{index}] must be smaller than stride or '
                    'dilation'
                )
        if isinstance(padding, str) and any(output_padding):
            raise ValueError(
                'output_padding requires explicit numeric padding'
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.spatial_rank = rank
        self.shard_mode = shard_mode

        weight_shape = (
            *kernel_size,
            in_channels,
            out_channels // groups,
        )
        self.weight = nn.Parameter(
            initializer(rngs(), weight_shape, dtype)
        )
        if axis_names is not None:
            if len(axis_names) != len(weight_shape):
                raise ValueError(
                    f'axis_names length {len(axis_names)} must match '
                    f'weight dimensions {len(weight_shape)}'
                )
            self.weight.axis_names = axis_names

        if bias:
            self.bias = nn.Parameter(
                bias_initializer(rngs(), (out_channels,), dtype)
            )
            if axis_names is not None:
                self.bias.axis_names = axis_names[-1:]

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x, unbatched = _as_batched(
            x,
            self.spatial_rank,
            channels=self.in_channels,
        )
        dimension_numbers = _conv_dimension_numbers(self.spatial_rank)
        if isinstance(self.padding, str):
            transpose_padding = self.padding
        else:
            transpose_padding = tuple(
                (
                    dilation * (kernel - 1) - low,
                    dilation * (kernel - 1) - high + extra,
                )
                for kernel, dilation, (low, high), extra in zip(
                    self.kernel_size,
                    self.dilation,
                    self.padding,
                    self.output_padding,
                )
            )

        inputs_per_group = self.in_channels // self.groups
        outputs = []
        for group in range(self.groups):
            start = group * inputs_per_group
            stop = start + inputs_per_group
            kernel = jnp.flip(
                self.weight.value[..., start:stop, :],
                axis=tuple(range(self.spatial_rank)),
            )
            outputs.append(
                jax.lax.conv_transpose(
                    x[..., start:stop],
                    kernel,
                    strides=self.stride,
                    padding=transpose_padding,
                    rhs_dilation=self.dilation,
                    dimension_numbers=dimension_numbers,
                )
            )
        output = jnp.concatenate(outputs, axis=-1)
        if self.has_bias:
            output = output + self.bias.value
        output = _restore_batch(output, unbatched)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'{self.in_channels} -> {self.out_channels}, '
            f'k={self.kernel_size}, s={self.stride}'
        )

class Unfold(nn.Module):
    """Extract sliding blocks into ``[batch, windows, patch_width]``."""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        stride: int | Sequence[int] = 1,
    ) -> None:
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.kernel_size = kernel_size
        self.dilation = Conv._normalize_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.padding = Conv._normalize_padding(padding, rank)
        self.stride = Conv._normalize_spatial(
            stride,
            rank=rank,
            name='stride',
        )
        self.spatial_rank = rank

    def __call__(self, x: jax.Array) -> jax.Array:
        x, unbatched = _as_batched(x, self.spatial_rank)
        patches = jax.lax.conv_general_dilated_patches(
            x,
            filter_shape=self.kernel_size,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            dimension_numbers=_conv_dimension_numbers(self.spatial_rank),
        )
        patches = patches.reshape(
            patches.shape[0],
            math.prod(patches.shape[1:-1]),
            patches.shape[-1],
        )
        return _restore_batch(patches, unbatched)

class Fold(nn.Module):
    """Overlap-add a matrix of flattened sliding blocks into an array."""

    def __init__(
        self,
        output_size: int | Sequence[int],
        kernel_size: int | Sequence[int],
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        stride: int | Sequence[int] = 1,
    ) -> None:
        output_size = Conv._normalize_spatial(output_size, name='output_size')
        rank = len(output_size)
        self.output_size = output_size
        self.kernel_size = Conv._normalize_spatial(
            kernel_size,
            rank=rank,
            name='kernel_size',
        )
        self.dilation = Conv._normalize_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.padding = Conv._normalize_padding(padding, rank)
        self.stride = Conv._normalize_spatial(
            stride,
            rank=rank,
            name='stride',
        )
        self.spatial_rank = rank

    def __call__(self, patches: jax.Array) -> jax.Array:
        if patches.ndim not in {2, 3}:
            raise ValueError(
                'Fold expects [windows, patch] or [batch, windows, patch]'
            )
        unbatched = patches.ndim == 2
        if unbatched:
            patches = patches[None, ...]

        padding = _canonical_padding(
            self.padding,
            self.output_size,
            self.kernel_size,
            self.stride,
            self.dilation,
        )
        grid_shape = _window_output_shape(
            self.output_size,
            self.kernel_size,
            self.stride,
            self.dilation,
            padding,
        )
        windows = math.prod(grid_shape)
        if patches.shape[1] != windows:
            raise ValueError(
                f'expected {windows} windows for output_size={self.output_size}, '
                f'got {patches.shape[1]}'
            )
        kernel_volume = math.prod(self.kernel_size)
        if patches.shape[-1] % kernel_volume:
            raise ValueError(
                'patch width must be divisible by the kernel volume '
                f'({kernel_volume})'
            )
        channels = patches.shape[-1] // kernel_volume
        patches = patches.reshape(
            patches.shape[0],
            *grid_shape,
            channels,
            *self.kernel_size,
        )
        output = jnp.zeros(
            (patches.shape[0], *self.output_size, channels),
            dtype=patches.dtype,
        )
        indices = _scatter_indices(
            patches.shape[0],
            channels,
            grid_shape,
            self.stride,
            padding,
        )
        grid_slices = (slice(None),) * self.spatial_rank
        for kernel_index in product(
            *(range(size) for size in self.kernel_size)
        ):
            spatial_indices = tuple(
                index + offset * spacing
                for index, offset, spacing in zip(
                    indices[1:-1],
                    kernel_index,
                    self.dilation,
                )
            )
            values = patches[
                (slice(None), *grid_slices, slice(None), *kernel_index)
            ]
            output = output.at[
                (indices[0], *spatial_indices, indices[-1])
            ].add(values, mode='drop')
        return _restore_batch(output, unbatched)

class MaxPool(nn.Module):
    """N-dimensional max pooling with optional flattened spatial indices."""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.kernel_size = kernel_size
        self.stride = Conv._normalize_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        self.padding = Conv._normalize_padding(padding, rank)
        self.dilation = Conv._normalize_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.spatial_rank = rank

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        x, unbatched = _as_batched(x, self.spatial_rank)
        spatial_shape = x.shape[1:-1]
        padding = _pool_padding(
            self.padding,
            spatial_shape,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.ceil_mode,
        )
        window, strides, reduce_padding, window_dilation = (
            _reduce_window_config(
                self.spatial_rank,
                self.kernel_size,
                self.stride,
                self.dilation,
                padding,
            )
        )
        initial = _max_identity(x.dtype)
        if not self.return_indices:
            output = jax.lax.reduce_window(
                x,
                initial,
                jax.lax.max,
                window,
                strides,
                reduce_padding,
                window_dilation=window_dilation,
            )
            return _restore_batch(output, unbatched)

        flat_indices = jnp.arange(
            math.prod(spatial_shape),
            dtype=jnp.int32,
        ).reshape((1, *spatial_shape, 1))
        flat_indices = jnp.broadcast_to(flat_indices, x.shape)
        no_index = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)

        def select_max(
            left: tuple[jax.Array, jax.Array],
            right: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            left_value, left_index = left
            right_value, right_index = right
            choose_right = (right_value > left_value) | (
                (right_value == left_value) & (right_index < left_index)
            )
            return (
                jnp.where(choose_right, right_value, left_value),
                jnp.where(choose_right, right_index, left_index),
            )

        output, indices = jax.lax.reduce_window(
            (x, flat_indices),
            (initial, no_index),
            select_max,
            window,
            strides,
            reduce_padding,
            window_dilation=window_dilation,
        )
        return (
            _restore_batch(output, unbatched),
            _restore_batch(indices, unbatched),
        )

class MaxUnpool(nn.Module):
    """Scatter pooled values back to flattened indices from :class:`MaxPool`."""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int | tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> None:
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.kernel_size = kernel_size
        self.stride = Conv._normalize_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        normalized_padding = Conv._normalize_padding(padding, rank)
        if isinstance(normalized_padding, str):
            raise ValueError('MaxUnpool requires explicit numeric padding')
        self.padding = normalized_padding
        self.dilation = Conv._normalize_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.spatial_rank = rank

    def __call__(
        self,
        x: jax.Array,
        indices: jax.Array,
        output_size: int | Sequence[int] | None = None,
    ) -> jax.Array:
        x, unbatched = _as_batched(x, self.spatial_rank)
        indices, indices_unbatched = _as_batched(indices, self.spatial_rank)
        if indices_unbatched != unbatched or indices.shape != x.shape:
            raise ValueError('indices must have the same shape as the pooled input')

        if output_size is None:
            output_size = tuple(
                (size - 1) * step
                - low
                - high
                + spacing * (kernel - 1)
                + 1
                for size, step, (low, high), spacing, kernel in zip(
                    x.shape[1:-1],
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.kernel_size,
                )
            )
        else:
            output_size = Conv._normalize_spatial(
                output_size,
                rank=self.spatial_rank,
                name='output_size',
            )

        batch_size, channels = x.shape[0], x.shape[-1]
        values = x.reshape(batch_size, -1, channels)
        flat_indices = indices.reshape(batch_size, -1, channels)
        output = jnp.zeros(
            (batch_size, math.prod(output_size), channels),
            dtype=x.dtype,
        )
        batch = jnp.arange(batch_size).reshape(batch_size, 1, 1)
        channel = jnp.arange(channels).reshape(1, 1, channels)
        output = output.at[batch, flat_indices, channel].set(
            values,
            mode='drop',
        )
        output = output.reshape(batch_size, *output_size, channels)
        return _restore_batch(output, unbatched)

class AvgPool(nn.Module):
    """N-dimensional average pooling with configurable divisor semantics."""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        if divisor_override is not None and divisor_override <= 0:
            raise ValueError('divisor_override must be positive')
        self.kernel_size = kernel_size
        self.stride = Conv._normalize_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        self.padding = Conv._normalize_padding(padding, rank)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.spatial_rank = rank

    def __call__(self, x: jax.Array) -> jax.Array:
        x, unbatched = _as_batched(x, self.spatial_rank)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            x = x.astype(jnp.float32)
        configured_padding = _canonical_padding(
            self.padding,
            x.shape[1:-1],
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
        )
        padding = _pool_padding(
            self.padding,
            x.shape[1:-1],
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
            self.ceil_mode,
        )
        window, strides, reduce_padding, _ = _reduce_window_config(
            self.spatial_rank,
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
            padding,
        )
        total = jax.lax.reduce_window(
            x,
            jnp.asarray(0, dtype=x.dtype),
            jax.lax.add,
            window,
            strides,
            reduce_padding,
        )
        if self.divisor_override is not None:
            divisor = self.divisor_override
        elif self.count_include_pad and not self.ceil_mode:
            divisor = math.prod(self.kernel_size)
        else:
            count_input = jnp.ones_like(x[..., :1])
            count_padding = reduce_padding
            if self.count_include_pad:
                count_input = jnp.pad(
                    count_input,
                    ((0, 0), *configured_padding, (0, 0)),
                    mode='constant',
                    constant_values=1,
                )
                count_padding = (
                    (0, 0),
                    *(
                        (
                            total_low - configured_low,
                            total_high - configured_high,
                        )
                        for (total_low, total_high), (
                            configured_low,
                            configured_high,
                        ) in zip(padding, configured_padding)
                    ),
                    (0, 0),
                )
            valid = jax.lax.reduce_window(
                count_input,
                jnp.asarray(0, dtype=x.dtype),
                jax.lax.add,
                window,
                strides,
                count_padding,
            )
            divisor = valid
        return _restore_batch(total / divisor, unbatched)

class FractionalMaxPool(nn.Module):
    """N-dimensional max pooling over reproducible fractional intervals."""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        output_size: int | Sequence[int] | None = None,
        output_ratio: float | Sequence[float] | None = None,
        return_indices: bool = False,
        random_samples: jax.Array | Sequence[float] | None = None,
        *,
        rngs: nn.Rngs | None = None,
    ) -> None:
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        if (output_size is None) == (output_ratio is None):
            raise ValueError(
                'exactly one of output_size or output_ratio must be provided'
            )
        if output_size is not None:
            output_size = Conv._normalize_spatial(
                output_size,
                rank=rank,
                name='output_size',
            )
        if output_ratio is not None:
            if isinstance(output_ratio, (int, float)):
                output_ratio = (float(output_ratio),) * rank
            else:
                output_ratio = tuple(float(value) for value in output_ratio)
            if len(output_ratio) != rank:
                raise ValueError(
                    f'output_ratio must contain {rank} values, '
                    f'got {len(output_ratio)}'
                )
            if any(value <= 0 or value > 1 for value in output_ratio):
                raise ValueError('output_ratio values must be in (0, 1]')

        if random_samples is None:
            if rngs is None:
                samples = jnp.full((rank,), 0.5, dtype=jnp.float32)
            else:
                samples = jax.random.uniform(rngs(), (rank,))
        else:
            if isinstance(random_samples, Sequence) and any(
                float(value) < 0 or float(value) >= 1
                for value in random_samples
            ):
                raise ValueError('random_samples values must be in [0, 1)')
            samples = jnp.asarray(random_samples, dtype=jnp.float32)
            if samples.shape != (rank,):
                raise ValueError(
                    f'random_samples must have shape ({rank},), '
                    f'got {samples.shape}'
                )

        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices
        self.random_samples = samples
        self.spatial_rank = rank

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        x, unbatched = _as_batched(x, self.spatial_rank)
        spatial_shape = x.shape[1:-1]
        if self.output_size is None:
            output_size = tuple(
                max(1, math.floor(size * ratio))
                for size, ratio in zip(spatial_shape, self.output_ratio)
            )
        else:
            output_size = self.output_size
        if any(
            kernel > size or output > size - kernel + 1
            for kernel, size, output in zip(
                self.kernel_size,
                spatial_shape,
                output_size,
            )
        ):
            raise ValueError(
                'kernel_size and output size must fit within the input'
            )

        starts = []
        for size, kernel, output, sample in zip(
            spatial_shape,
            self.kernel_size,
            output_size,
            self.random_samples,
        ):
            maximum = size - kernel
            if output == 1:
                positions = jnp.asarray(
                    [jnp.floor(sample * (maximum + 1))],
                    dtype=jnp.int32,
                )
            else:
                alpha = maximum / (output - 1)
                positions = jnp.floor(
                    (jnp.arange(output) + sample) * alpha
                ) - jnp.floor(sample * alpha)
                positions = positions.astype(jnp.int32).at[-1].set(maximum)
            starts.append(positions)

        values = []
        indices = []
        batch_size, channels = x.shape[0], x.shape[-1]
        kernel_volume = math.prod(self.kernel_size)
        for output_index in product(*(range(size) for size in output_size)):
            start = tuple(
                starts[axis][position]
                for axis, position in enumerate(output_index)
            )
            patch = jax.lax.dynamic_slice(
                x,
                (0, *start, 0),
                (batch_size, *self.kernel_size, channels),
            )
            patch = patch.reshape(batch_size, kernel_volume, channels)
            local_index = jnp.argmax(patch, axis=1).astype(jnp.int32)
            values.append(jnp.take_along_axis(
                patch,
                local_index[:, None, :],
                axis=1,
            )[:, 0, :])
            if self.return_indices:
                remainder = local_index
                coordinates = []
                for kernel in reversed(self.kernel_size):
                    coordinates.append(remainder % kernel)
                    remainder = remainder // kernel
                coordinates.reverse()
                global_index = jnp.zeros_like(local_index)
                for size, offset, coordinate in zip(
                    spatial_shape,
                    start,
                    coordinates,
                ):
                    global_index = global_index * size + offset + coordinate
                indices.append(global_index)

        output = jnp.stack(values, axis=1).reshape(
            batch_size,
            *output_size,
            channels,
        )
        output = _restore_batch(output, unbatched)
        if not self.return_indices:
            return output
        index_output = jnp.stack(indices, axis=1).reshape(
            batch_size,
            *output_size,
            channels,
        )
        return output, _restore_batch(index_output, unbatched)

class LPPool(nn.Module):
    """N-dimensional pooling using the sum-based p-norm of each window."""

    def __init__(
        self,
        norm_type: float,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        ceil_mode: bool = False,
    ) -> None:
        if norm_type <= 0:
            raise ValueError('norm_type must be positive')
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = Conv._normalize_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        self.ceil_mode = ceil_mode
        self.spatial_rank = rank

    def __call__(self, x: jax.Array) -> jax.Array:
        x, unbatched = _as_batched(x, self.spatial_rank)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            x = x.astype(jnp.float32)
        padding = _pool_padding(
            ((0, 0),) * self.spatial_rank,
            x.shape[1:-1],
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
            self.ceil_mode,
        )
        window, strides, reduce_padding, _ = _reduce_window_config(
            self.spatial_rank,
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
            padding,
        )
        powered = jnp.abs(x) ** self.norm_type
        total = jax.lax.reduce_window(
            powered,
            jnp.asarray(0, dtype=x.dtype),
            jax.lax.add,
            window,
            strides,
            reduce_padding,
        )
        output = total ** (1.0 / self.norm_type)
        return _restore_batch(output, unbatched)

class AdaptiveMaxPool(nn.Module):
    """Max-pool adaptive regions to a requested spatial output size."""

    def __init__(
        self,
        output_size: int | Sequence[int | None],
        return_indices: bool = False,
    ) -> None:
        self.output_size = _normalize_adaptive_size(output_size)
        self.return_indices = return_indices

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        rank = len(self.output_size)
        x, unbatched = _as_batched(x, rank)
        spatial_shape = x.shape[1:-1]
        output_size = tuple(
            size if requested is None else requested
            for size, requested in zip(spatial_shape, self.output_size)
        )
        values, indices = _adaptive_pool(
            x,
            output_size,
            reduction='max',
            return_indices=self.return_indices,
        )
        values = _restore_batch(values, unbatched)
        if not self.return_indices:
            return values
        return values, _restore_batch(indices, unbatched)

class AdaptiveAvgPool(nn.Module):
    """Average-pool adaptive regions to a requested spatial output size."""

    def __init__(
        self,
        output_size: int | Sequence[int | None],
    ) -> None:
        self.output_size = _normalize_adaptive_size(output_size)

    def __call__(self, x: jax.Array) -> jax.Array:
        rank = len(self.output_size)
        x, unbatched = _as_batched(x, rank)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            x = x.astype(jnp.float32)
        output_size = tuple(
            size if requested is None else requested
            for size, requested in zip(x.shape[1:-1], self.output_size)
        )
        values, _ = _adaptive_pool(
            x,
            output_size,
            reduction='mean',
            return_indices=False,
        )
        return _restore_batch(values, unbatched)

class Padding(nn.Module):
    """Apply rank-generic ``jax.numpy.pad`` padding to an array."""

    def __init__(
        self,
        padding: int | Sequence[int] | Sequence[tuple[int, int]],
        mode: str = 'constant',
        value: float = 0.0,
    ) -> None:
        aliases = {
            'zeros': 'constant',
            'replicate': 'edge',
            'circular': 'wrap',
        }
        mode = aliases.get(mode.lower(), mode.lower())
        supported = {
            'constant',
            'edge',
            'reflect',
            'symmetric',
            'wrap',
        }
        if mode not in supported:
            choices = ', '.join(sorted(supported | set(aliases)))
            raise ValueError(f'padding mode must be one of {choices}')
        self.padding = padding
        self.mode = mode
        self.value = value

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.mode == 'constant':
            return jnp.pad(
                x,
                self.padding,
                mode=self.mode,
                constant_values=self.value,
            )
        return jnp.pad(x, self.padding, mode=self.mode)

__all__ = [
    'Conv',
    'ConvTranspose',
    'Unfold',
    'Fold',
    'MaxPool',
    'MaxUnpool',
    'AvgPool',
    'FractionalMaxPool',
    'LPPool',
    'AdaptiveMaxPool',
    'AdaptiveAvgPool',
    'Padding',
]
