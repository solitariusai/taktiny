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
from collections.abc import Callable, Sequence
from itertools import product
import math
from numbers import Real
import jax
import jax.numpy as jnp

from taktiny.utils.typing import Activation, Axes, ShardMode


def _validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 1,
) -> int:
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
    values = (value,) if isinstance(value, int) else tuple(value)
    if not values:
        raise ValueError(f'{name} must contain at least one dimension')
    for index, size in enumerate(values):
        _validate_integer(size, f'{name}[{index}]')
    return values


def _validate_positive_float(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _validate_probability(
    value: float,
    name: str = 'p',
    *,
    allow_one: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    value = float(value)
    upper_valid = value <= 1 if allow_one else value < 1
    if not math.isfinite(value) or value < 0 or not upper_valid:
        interval = '[0, 1]' if allow_one else '[0, 1)'
        raise ValueError(f'{name} must be finite and in {interval}')
    return value


def _resolve_training(default: bool, training: bool | None) -> bool:
    if training is None:
        return default
    if not isinstance(training, bool):
        raise TypeError('training must be a bool or None')
    return training


def _canonical_axis(
    axis: int,
    ndim: int,
    *,
    name: str = 'axis',
    allow_scalar: bool = False,
) -> int:
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


def _resolve_activation(
    activation: Activation | None,
    *,
    allow_none: bool = False,
) -> Callable[[jax.Array], jax.Array]:
    if activation is None:
        if allow_none:
            return lambda value: value
        raise TypeError('activation must be a string or callable')
    if callable(activation):
        return activation
    if not isinstance(activation, str):
        expected = 'a string, callable, or None' if allow_none else 'a string or callable'
        raise TypeError(f'activation must be {expected}')
    normalized = activation.lower().replace('-', '_')
    compact = normalized.replace('_', '')
    if compact == 'relu6':
        return lambda value: jnp.minimum(jax.nn.relu(value), 6)
    if compact in {
        'gelupytorchtanh',
        'gelunew',
        'gelufast',
        'geluapproximate',
    }:
        return lambda value: jax.nn.gelu(value, approximate=True)
    if compact == 'swish':
        return jax.nn.silu
    function = getattr(jax.nn, normalized, None)
    if function is None or not callable(function):
        raise ValueError(f'unsupported activation: {activation!r}')
    return function


def _constrain(
    value: jax.Array,
    sharding: jax.sharding.Sharding | None,
    shard_mode: ShardMode,
) -> jax.Array:
    if shard_mode == ShardMode.EXPLICIT and sharding is not None:
        return jax.lax.with_sharding_constraint(value, sharding)
    return value

def _normalize_nonnegative(
    value: int | Sequence[int],
    rank: int,
    *,
    name: str,
) -> tuple[int, ...]:
    if isinstance(value, int):
        values = (value,) * rank
    else:
        values = tuple(value)
    if len(values) != rank:
        raise ValueError(f'{name} must contain {rank} values, got {len(values)}')
    if any(not isinstance(item, int) or item < 0 for item in values):
        raise ValueError(f'{name} values must be non-negative integers')
    return values


def _conv_dimension_numbers(rank: int) -> jax.lax.ConvDimensionNumbers:
    lhs_spec = (0, rank + 1, *range(1, rank + 1))
    rhs_spec = (rank + 1, rank, *range(rank))
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, lhs_spec)


def _as_batched(
    x: jax.Array,
    rank: int,
    *,
    channels: int | None = None,
) -> tuple[jax.Array, bool]:
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


def _restore_batch(x: jax.Array, unbatched: bool) -> jax.Array:
    return x[0] if unbatched else x


def _canonical_padding(
    padding: str | tuple[tuple[int, int], ...],
    input_shape: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> tuple[tuple[int, int], ...]:
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
    return (
        (1, *kernel_size, 1),
        (1, *stride, 1),
        ((0, 0), *padding, (0, 0)),
        (1, *dilation, 1),
    )


def _scatter_indices(
    batch_size: int,
    channels: int,
    grid_shape: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> tuple[jax.Array, ...]:
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


def _max_identity(dtype: jnp.dtype) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.bool_):
        return jnp.asarray(False, dtype=dtype)
    if jnp.issubdtype(dtype, jnp.integer):
        return jnp.asarray(jnp.iinfo(dtype).min, dtype=dtype)
    return jnp.asarray(-jnp.inf, dtype=dtype)


def _normalize_adaptive_size(
    output_size: int | Sequence[int | None],
) -> tuple[int | None, ...]:
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


def _adaptive_pool(
    x: jax.Array,
    output_size: Sequence[int],
    *,
    reduction: str,
    return_indices: bool,
) -> tuple[jax.Array, jax.Array | None]:
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

__all__ = [
    '_validate_integer',
    '_normalize_shape',
    '_validate_positive_float',
    '_resolve_training',
    '_canonical_axis',
    '_canonical_axes',
    '_resolve_activation',
    '_constrain',
    '_normalize_nonnegative',
    '_conv_dimension_numbers',
    '_as_batched',
    '_restore_batch',
    '_canonical_padding',
    '_window_output_shape',
    '_pool_padding',
    '_reduce_window_config',
    '_scatter_indices',
    '_max_identity',
    '_normalize_adaptive_size',
    '_adaptive_pool',
]
