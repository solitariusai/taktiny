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
from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product
from numbers import Real

import jax
import jax.numpy as jnp


def _constrain(
    value: jax.Array,
    sharding: jax.sharding.Sharding | None,
) -> jax.Array:
    """Applies a sharding constraint to the given array if explicitly requested.

    Args:
        value (jax.Array): The array to constrain.
        sharding (jax.sharding.Sharding | None): The sharding configuration.

    Returns:
        jax.Array: The constrained array, or the original if unconstrained.
    """
    if sharding is not None:
        return jax.lax.with_sharding_constraint(value, sharding)
    return value


def _canonical_axis(
    axis: int,
    ndim: int,
    *,
    name: str = 'axis',
    allow_scalar: bool = False,
) -> int:
    """Converts a potentially negative axis index into a canonical non-negative index.

    Args:
        axis (int): The axis index to resolve.
        ndim (int): The rank (number of dimensions) of the array.
        name (str, optional): The name of the value for error messages. Defaults to 'axis'.
        allow_scalar (bool, optional): Whether to allow scalar handling. Defaults to False.

    Returns:
        int: The canonical non-negative axis index.
    """
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError(f'{name} must be an integer')
    if ndim == 0 and allow_scalar and axis in {-1, 0}:
        return 0
    canonical = axis + ndim if axis < 0 else axis
    if canonical < 0 or canonical >= ndim:
        raise ValueError(
            f'{name}={axis} is out of range for an array of rank {ndim}'
        )
    return canonical


def _canonical_axes(
    axes: Axes,
    ndim: int,
    *,
    name: str = 'axes',
    allow_empty: bool = False,
) -> tuple[int, ...]:
    """Converts a sequence of axes into a canonical sequence of unique non-negative indices.

    Args:
        axes (Axes): An axis or sequence of axes.
        ndim (int): The rank (number of dimensions) of the array.
        name (str, optional): The name of the value for error messages. Defaults to 'axes'.
        allow_empty (bool, optional): Whether an empty sequence is allowed. Defaults to False.

    Returns:
        tuple[int, ...]: A tuple of unique canonical non-negative axes.
    """
    values = (axes,) if isinstance(axes, int) else tuple(axes)
    if not values and not allow_empty:
        raise ValueError(f'{name} must contain at least one axis')
    canonical = tuple(
        _canonical_axis(axis, ndim, name=f'{name} value')
        for axis in values
    )
    if len(set(canonical)) != len(canonical):
        raise ValueError(f'{name} must not contain duplicates')
    return canonical


def _validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 1,
) -> int:
    """Validates that a value is an integer meeting a minimum constraint.

    Args:
        value (int): The value to validate.
        name (str): The name of the value for error messages.
        minimum (int, optional): The minimum allowed value. Defaults to 1.

    Returns:
        int: The validated integer value.
    """
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        if minimum == 1:
            requirement = 'a positive integer'
        elif minimum == 0:
            requirement = 'a non-negative integer'
        else:
            requirement = f'an integer greater than or equal to {minimum}'
        raise ValueError(f'{name} must be {requirement}')
    return value


def _normalize_shape(
    value: int | Sequence[int],
    name: str,
) -> tuple[int, ...]:
    """Normalizes a shape to a tuple of positive integers.

    Args:
        value (int | Sequence[int]): A single integer or sequence of integers.
        name (str): The name of the value for error messages.

    Returns:
        tuple[int, ...]: A normalized shape tuple containing at least one dimension.
    """
    values = (value,) if isinstance(value, int) else tuple(value)
    if not values:
        raise ValueError(f'{name} must contain at least one dimension')
    for index, size in enumerate(values):
        _validate_integer(size, f'{name}[{index}]')
    return values


def _resolve_training(default: bool, training: bool | None) -> bool:
    """Resolves the training flag, falling back to a default if None.

    Args:
        default (bool): The default boolean value.
        training (bool | None): The specific training flag or None.

    Returns:
        bool: The resolved boolean training flag.
    """
    if training is None:
        return default
    if not isinstance(training, bool):
        raise TypeError('training must be a bool or None')
    return training


def _validate_positive_float(value: float, name: str) -> float:
    """Validates that a value is a finite positive float.

    Args:
        value (float): The value to validate.
        name (str): The name of the value for error messages.

    Returns:
        float: The validated positive float value.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _adaptive_pool(
    x: jax.Array,
    output_size: Sequence[int],
    *,
    reduction: str,
    return_indices: bool,
) -> tuple[jax.Array, jax.Array | None]:
    """Performs adaptive pooling to a dynamically determined spatial shape.

    Args:
        x (jax.Array): The batched, channels-last input tensor.
        output_size (Sequence[int]): The desired output spatial dimensions.
        reduction (str): The reduction method ('mean' or 'max').
        return_indices (bool): Whether to return spatial indices (only valid for 'max').

    Returns:
        tuple[jax.Array, jax.Array | None]: The pooled array and optional indices.
    """
    spatial_shape = x.shape[1:-1]
    if len(output_size) != len(spatial_shape):
        raise ValueError('output_size rank does not match the input')
    batch_size, channels = x.shape[0], x.shape[-1]
    values = []
    indices = []
    for output_index in product(*(range(size) for size in output_size)):
        starts = tuple(
            math.floor(index * input_size / output)
            for index, input_size, output in zip(
                output_index,
                spatial_shape,
                output_size,
            )
        )
        ends = tuple(
            math.ceil((index + 1) * input_size / output)
            for index, input_size, output in zip(
                output_index,
                spatial_shape,
                output_size,
            )
        )
        slices = tuple(slice(start, end) for start, end in zip(starts, ends))
        patch = x[(slice(None), *slices, slice(None))]
        window_shape = tuple(end - start for start, end in zip(starts, ends))
        patch = patch.reshape(batch_size, math.prod(window_shape), channels)
        if reduction == 'mean':
            values.append(jnp.mean(patch, axis=1))
            continue

        local_index = jnp.argmax(patch, axis=1).astype(jnp.int32)
        values.append(jnp.take_along_axis(
            patch,
            local_index[:, None, :],
            axis=1,
        )[:, 0, :])
        if return_indices:
            remainder = local_index
            coordinates = []
            for window in reversed(window_shape):
                coordinates.append(remainder % window)
                remainder = remainder // window
            coordinates.reverse()
            global_index = jnp.zeros_like(local_index)
            for size, start, coordinate in zip(
                spatial_shape,
                starts,
                coordinates,
            ):
                global_index = global_index * size + start + coordinate
            indices.append(global_index)

    output = jnp.stack(values, axis=1).reshape(
        batch_size,
        *output_size,
        channels,
    )
    if not return_indices:
        return output, None
    index_output = jnp.stack(indices, axis=1).reshape(
        batch_size,
        *output_size,
        channels,
    )
    return output, index_output


def _as_batched(
    x: jax.Array,
    rank: int,
    *,
    channels: int | None = None,
) -> tuple[jax.Array, bool]:
    """Ensures the input tensor has a batch dimension, optionally verifying channels.

    Args:
        x (jax.Array): The input array.
        rank (int): The spatial rank of the operation.
        channels (int | None, optional): Expected number of channels. Defaults to None.

    Returns:
        tuple[jax.Array, bool]: The batched array and a flag indicating if it was initially unbatched.
    """
    batched_rank = rank + 2
    if x.ndim not in {batched_rank - 1, batched_rank}:
        raise ValueError(
            f'expected an unbatched rank-{batched_rank - 1} or batched '
            f'rank-{batched_rank} input, got rank {x.ndim}'
        )
    if channels is not None and x.shape[-1] != channels:
        raise ValueError(f'expected {channels} input channels, got {x.shape[-1]}')
    unbatched = x.ndim == batched_rank - 1
    return (x[None, ...] if unbatched else x), unbatched


def _canonical_padding(
    padding: str | tuple[tuple[int, int], ...],
    input_shape: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Computes explicit padding sizes given convolution parameters.

    Args:
        padding (str | tuple[tuple[int, int], ...]): Padding string or explicit padding tuple.
        input_shape (Sequence[int]): Spatial shape of the input.
        kernel_size (Sequence[int]): Spatial sizes of the kernel.
        stride (Sequence[int]): Stride of the convolution.
        dilation (Sequence[int]): Dilation of the kernel.

    Returns:
        tuple[tuple[int, int], ...]: Computed explicit (low, high) padding per spatial dimension.
    """
    if not isinstance(padding, str):
        return padding
    if padding == 'VALID':
        return ((0, 0),) * len(kernel_size)

    pairs = []
    for size, kernel, step, spacing in zip(
        input_shape,
        kernel_size,
        stride,
        dilation,
    ):
        effective_kernel = spacing * (kernel - 1) + 1
        output = math.ceil(size / step)
        total = max((output - 1) * step + effective_kernel - size, 0)
        low = total // 2
        pairs.append((low, total - low))
    return tuple(pairs)


def _window_output_shape(
    input_shape: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Calculates the output spatial shape of a sliding window operation.

    Args:
        input_shape (Sequence[int]): Spatial shape of the input.
        kernel_size (Sequence[int]): Spatial sizes of the kernel.
        stride (Sequence[int]): Stride of the operation.
        dilation (Sequence[int]): Dilation of the kernel.
        padding (Sequence[tuple[int, int]]): Explicit padding per dimension.

    Returns:
        tuple[int, ...]: The spatial shape of the output.
    """
    output = []
    for size, kernel, step, spacing, (low, high) in zip(
        input_shape,
        kernel_size,
        stride,
        dilation,
        padding,
    ):
        effective_kernel = spacing * (kernel - 1) + 1
        length = (size + low + high - effective_kernel) // step + 1
        if length <= 0:
            raise ValueError('kernel and padding produce an empty output')
        output.append(length)
    return tuple(output)


def _pool_padding(
    padding: str | tuple[tuple[int, int], ...],
    input_shape: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
) -> tuple[tuple[int, int], ...]:
    """Computes explicit padding sizes for pooling operations, handling ceil_mode.

    Args:
        padding (str | tuple[tuple[int, int], ...]): Padding string or explicit padding tuple.
        input_shape (Sequence[int]): Spatial shape of the input.
        kernel_size (Sequence[int]): Spatial sizes of the pooling kernel.
        stride (Sequence[int]): Stride of the pooling operation.
        dilation (Sequence[int]): Dilation of the pooling kernel.
        ceil_mode (bool): Whether to use ceil instead of floor for output shape.

    Returns:
        tuple[tuple[int, int], ...]: Computed explicit padding per spatial dimension.
    """
    pairs = list(
        _canonical_padding(
            padding,
            input_shape,
            kernel_size,
            stride,
            dilation,
        )
    )
    if not ceil_mode:
        return tuple(pairs)

    result = []
    for size, kernel, step, spacing, (low, high) in zip(
        input_shape,
        kernel_size,
        stride,
        dilation,
        pairs,
    ):
        effective_kernel = spacing * (kernel - 1) + 1
        output = math.floor(
            (size + low + high - effective_kernel + step - 1) / step
        ) + 1
        if output > 0 and (output - 1) * step >= size + low:
            output -= 1
        required_high = max(
            0,
            (output - 1) * step + effective_kernel - size - low,
        )
        result.append((low, max(high, required_high)))
    return tuple(result)


def _reduce_window_config(
    rank: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    tuple[int, ...],
]:
    """Formats window configuration for JAX reduce_window operations with channels-last data.

    Args:
        rank (int): Spatial rank.
        kernel_size (Sequence[int]): Spatial kernel sizes.
        stride (Sequence[int]): Spatial strides.
        dilation (Sequence[int]): Spatial dilations.
        padding (Sequence[tuple[int, int]]): Spatial paddings.

    Returns:
        tuple: Expanded kernel, stride, padding, and dilation for batch and channel dimensions.
    """
    return (
        (1, *kernel_size, 1),
        (1, *stride, 1),
        ((0, 0), *padding, (0, 0)),
        (1, *dilation, 1),
    )


def _conv_dimension_numbers(rank: int) -> jax.lax.ConvDimensionNumbers:
    """Generates standard convolution dimension numbers for channels-last format.

    Args:
        rank (int): The spatial rank of the convolution (e.g., 2 for 2D).

    Returns:
        jax.lax.ConvDimensionNumbers: The computed dimension numbers for convolution.
    """
    lhs_spec = (0, rank + 1, *range(1, rank + 1))
    rhs_spec = (rank + 1, rank, *range(rank))
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, lhs_spec)


def _max_identity(dtype: jnp.dtype) -> jax.Array:
    """Returns the identity value for the max reduction operation for a given dtype.

    Args:
        dtype (jnp.dtype): The JAX data type.

    Returns:
        jax.Array: The identity value (e.g., -inf or min value) for max reduction.
    """
    if jnp.issubdtype(dtype, jnp.bool_):
        return jnp.asarray(False, dtype=dtype)
    if jnp.issubdtype(dtype, jnp.integer):
        return jnp.asarray(jnp.iinfo(dtype).min, dtype=dtype)
    return jnp.asarray(-jnp.inf, dtype=dtype)


def _normalize_adaptive_size(
    output_size: int | Sequence[int | None],
) -> tuple[int | None, ...]:
    """Normalizes the output size for adaptive pooling operations.

    Args:
        output_size (int | Sequence[int | None]): Target spatial size.

    Returns:
        tuple[int | None, ...]: A normalized tuple of target spatial sizes.
    """
    if isinstance(output_size, int):
        values: tuple[int | None, ...] = (output_size,)
    else:
        values = tuple(output_size)
    if not values:
        raise ValueError('output_size must contain at least one dimension')
    if any(
        value is not None
        and (not isinstance(value, int) or value <= 0)
        for value in values
    ):
        raise ValueError('output_size values must be positive integers or None')
    return values


def _normalize_nonnegative(
    value: int | Sequence[int],
    rank: int,
    *,
    name: str,
) -> tuple[int, ...]:
    """Normalizes a value into a sequence of non-negative integers of a given rank.

    Args:
        value (int | Sequence[int]): A single integer or sequence of integers.
        rank (int): The expected length of the resulting tuple.
        name (str): The name of the value for error messages.

    Returns:
        tuple[int, ...]: A normalized tuple of non-negative integers.
    """
    if isinstance(value, int):
        values = (value,) * rank
    else:
        values = tuple(value)
    if len(values) != rank:
        raise ValueError(f'{name} must contain {rank} values, got {len(values)}')
    if any(not isinstance(item, int) or item < 0 for item in values):
        raise ValueError(f'{name} values must be non-negative integers')
    return values


def _restore_batch(x: jax.Array, unbatched: bool) -> jax.Array:
    """Removes the batch dimension if the original input was unbatched.

    Args:
        x (jax.Array): The potentially batched array.
        unbatched (bool): Whether the original input lacked a batch dimension.

    Returns:
        jax.Array: The array with its original batched or unbatched shape.
    """
    return x[0] if unbatched else x


def _scatter_indices(
    batch_size: int,
    channels: int,
    grid_shape: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> tuple[jax.Array, ...]:
    """Generates broadcastable indices for scatter operations across batch, spatial, and channel dimensions.

    Args:
        batch_size (int): Size of the batch dimension.
        channels (int): Number of channels.
        grid_shape (Sequence[int]): Spatial shape of the output grid.
        stride (Sequence[int]): Stride of the operation.
        padding (Sequence[tuple[int, int]]): Applied padding.

    Returns:
        tuple[jax.Array, ...]: A tuple of JAX arrays containing indices for each dimension.
    """
    rank = len(grid_shape)
    batch = jnp.arange(batch_size).reshape(
        (batch_size, *(1,) * rank, 1)
    )
    spatial = []
    for axis, (size, step, (low, _)) in enumerate(
        zip(grid_shape, stride, padding)
    ):
        shape = (1, *((1,) * axis), size, *((1,) * (rank - axis - 1)), 1)
        spatial.append((jnp.arange(size) * step - low).reshape(shape))
    channel = jnp.arange(channels).reshape((1, *(1,) * rank, channels))
    return (batch, *spatial, channel)


def _validate_probability(
    value: float,
    name: str = 'p',
    *,
    allow_one: bool = True,
) -> float:
    """Validates that a value is a valid probability between 0 and 1.

    Args:
        value (float): The probability value to validate.
        name (str, optional): The name of the value for error messages. Defaults to 'p'.
        allow_one (bool, optional): Whether 1.0 is considered a valid probability. Defaults to True.

    Returns:
        float: The validated probability value.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    value = float(value)
    upper_valid = value <= 1 if allow_one else value < 1
    if not math.isfinite(value) or value < 0 or not upper_valid:
        interval = '[0, 1]' if allow_one else '[0, 1)'
        raise ValueError(f'{name} must be finite and in {interval}')
    return value


__all__ = [
    '_adaptive_pool',
    '_as_batched',
    '_canonical_axes',
    '_canonical_axis',
    '_canonical_padding',
    '_constrain',
    '_conv_dimension_numbers',
    '_max_identity',
    '_normalize_adaptive_size',
    '_normalize_nonnegative',
    '_normalize_shape',
    '_pool_padding',
    '_reduce_window_config',
    '_resolve_training',
    '_restore_batch',
    '_scatter_indices',
    '_validate_integer',
    '_validate_positive_float',
    '_validate_probability',
    '_window_output_shape',
]