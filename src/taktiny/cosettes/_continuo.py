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

from collections.abc import Sequence
import typing as tp

import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.maestro.config import ModelConfig
from taktiny.utils.typing import DType, ShardMode


def _positive_int(value: tp.Any, name: str) -> int:
    """Validate and return a positive, non-boolean integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _config_value(
    config: ModelConfig,
    *names: str,
    default: tp.Any = None,
) -> tp.Any:
    """Return the first explicitly configured, non-None value."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _require_config(config: ModelConfig, *names: str) -> tp.Any:
    value = _config_value(config, *names)
    if value is None:
        choices = ', '.join(names)
        raise ValueError(f'config must define one of: {choices}')
    return value


def _hidden_size(config: ModelConfig) -> int:
    value = _config_value(config, 'hidden_size', 'inner_dim', 'dim')
    if value is None:
        heads = _config_value(config, 'num_attention_heads')
        head_dim = _config_value(config, 'head_dim', 'attention_head_dim')
        if heads is not None and head_dim is not None:
            value = heads * head_dim
    if not isinstance(value, int) or value <= 0:
        raise ValueError('config must define a positive hidden size')
    return value


def _model_dtype(
    config: ModelConfig,
    *,
    default: DType | str = 'bfloat16',
) -> DType | str:
    return _config_value(config, 'torch_dtype', 'dtype', default=default)


def _shard_mode(config: ModelConfig) -> ShardMode:
    value = _config_value(config, 'shard_mode', default=ShardMode.AUTO)
    if isinstance(value, ShardMode):
        return value
    try:
        return ShardMode(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'unsupported shard_mode: {value!r}') from error


def _activation(config: ModelConfig, *, default: str = 'silu') -> tp.Any:
    value = _config_value(
        config,
        'hidden_act',
        'hidden_activation',
        'activation',
        'activation_fn',
        'act',
        default=default,
    )
    if isinstance(value, str) and value in {
        'gelu_pytorch_tanh',
        'gelu_new',
        'gelu_fast',
        'gelu-approximate',
        'gelu_approximate',
    }:
        return _approximate_gelu
    return value


def _integer_tuple(value: tp.Any, name: str) -> tuple[int, ...]:
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f'{name} must be a sequence of integers') from error
    if not values:
        raise ValueError(f'{name} must not be empty')
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in values
    ):
        raise ValueError(f'{name} values must be non-negative integers')
    return values


def _stage_values(
    value: str | Sequence[str],
    count: int,
    name: str,
) -> tuple[str, ...]:
    values = (value,) * count if isinstance(value, str) else tuple(value)
    if len(values) != count:
        raise ValueError(f'{name} must contain {count} values')
    if any(not isinstance(item, str) or not item for item in values):
        raise TypeError(f'{name} values must be non-empty strings')
    return values


def _multiscales(
    value: tp.Any,
    count: int,
    name: str,
) -> tuple[tuple[int, ...], ...]:
    try:
        values = tuple(tuple(stage) for stage in value)
    except TypeError as error:
        raise TypeError(
            f'{name} must be a sequence of kernel sequences'
        ) from error
    if len(values) != count:
        raise ValueError(f'{name} must contain {count} values')
    for stage in values:
        if any(
            not isinstance(kernel, int)
            or isinstance(kernel, bool)
            or kernel <= 0
            or kernel % 2 == 0
            for kernel in stage
        ):
            raise ValueError(f'{name} kernels must be positive odd integers')
    return values


def _pixel_unshuffle(x: jax.Array, factor: int = 2) -> jax.Array:
    """Rearrange NHWC spatial neighborhoods into the channel dimension."""
    x = jnp.asarray(x)
    if x.ndim != 4:
        raise ValueError(
            'pixel unshuffle expects [batch, height, width, channels]'
        )
    batch, height, width, channels = x.shape
    if height % factor or width % factor:
        raise ValueError(
            f'spatial dimensions {(height, width)} must divide by {factor}'
        )
    x = x.reshape(
        batch,
        height // factor,
        factor,
        width // factor,
        factor,
        channels,
    )
    x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))
    return x.reshape(
        batch,
        height // factor,
        width // factor,
        channels * factor * factor,
    )


def _pixel_shuffle(x: jax.Array, factor: int = 2) -> jax.Array:
    """Rearrange NHWC channels into a higher-resolution spatial grid."""
    x = jnp.asarray(x)
    if x.ndim != 4:
        raise ValueError('pixel shuffle expects [batch, height, width, channels]')
    batch, height, width, channels = x.shape
    ratio = factor * factor
    if channels % ratio:
        raise ValueError(f'channel dimension {channels} must divide by {ratio}')
    out_channels = channels // ratio
    x = x.reshape(
        batch,
        height,
        width,
        out_channels,
        factor,
        factor,
    )
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(
        batch,
        height * factor,
        width * factor,
        out_channels,
    )


class _Identity(nn.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


def _normalization(
    norm_type: str | None,
    channels: int,
    *,
    dtype: DType,
    shard_mode: ShardMode,
) -> nn.Module:
    if norm_type is None:
        return _Identity()
    normalized = norm_type.lower().replace('-', '').replace('_', '')
    if normalized in {'rms', 'rmsnorm'}:
        return nn.RMSNorm(
            channels,
            eps=1e-5,
            dtype=jnp.float32,
            bias=True,
            axis_names=('embed',),
            shard_mode=shard_mode,
        )
    if normalized in {'batch', 'batchnorm'}:
        return nn.BatchNorm(
            channels,
            eps=1e-5,
            dtype=dtype,
            channel_axis=-1,
            axis_names=('embed',),
            shard_mode=shard_mode,
        )
    raise ValueError(f'unsupported autoencoder normalization: {norm_type!r}')


def image_transformer_dimensions(
    config: ModelConfig,
) -> tuple[int, int, int, int]:
    num_heads = _config_value(config, 'num_attention_heads')
    head_dim = _config_value(config, 'attention_head_dim', 'head_dim')
    hidden_size = _config_value(config, 'hidden_size', 'inner_dim', 'dim')
    if hidden_size is None and num_heads is not None and head_dim is not None:
        hidden_size = num_heads * head_dim
    values = {
        'hidden_size': hidden_size,
        'num_attention_heads': num_heads,
        'attention_head_dim': head_dim,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            'Missing required image-transformer config values: '
            + ', '.join(missing)
        )
    if not all(
        isinstance(value, int) and value > 0
        for value in values.values()
    ):
        raise ValueError(
            'image-transformer dimensions must be positive integers'
        )
    if hidden_size != num_heads * head_dim:
        raise ValueError(
            'hidden_size must equal num_attention_heads * attention_head_dim'
        )

    intermediate_size = _config_value(config, 'intermediate_size')
    if intermediate_size is None:
        intermediate_size = int(
            hidden_size * _config_value(config, 'mlp_ratio', default=4.0)
        )
    if not isinstance(intermediate_size, int) or intermediate_size <= 0:
        raise ValueError('intermediate_size must be a positive integer')
    return hidden_size, num_heads, head_dim, intermediate_size


def multi_axis_position_embedding(
    config: ModelConfig,
) -> nn.Module | None:
    configured = _config_value(config, 'pos_emb', 'position_embedding')
    if configured is not None:
        if not isinstance(configured, nn.Module):
            raise TypeError('configured position embedding must be a Module')
        return configured
    axes_dim: Sequence[int] | None = _config_value(config, 'axes_dims_rope')
    if axes_dim is None:
        return None

    from taktiny.layers import MultiAxisRotaryEmbedding

    return MultiAxisRotaryEmbedding(
        axes_dim,
        theta=_config_value(config, 'rope_theta', default=10_000.0),
    )


def combine_joint_positions(
    text_position_idx: jax.Array | None,
    image_position_idx: jax.Array | None,
    *,
    batch_size: int,
) -> jax.Array | None:
    if text_position_idx is None and image_position_idx is None:
        return None
    if text_position_idx is None or image_position_idx is None:
        raise ValueError(
            'text and image position IDs must be provided together'
        )

    def batched(value: jax.Array) -> jax.Array:
        value = jnp.asarray(value)
        if value.ndim == 2:
            value = jnp.broadcast_to(
                value[None, ...],
                (batch_size, *value.shape),
            )
        if value.ndim != 3 or value.shape[0] != batch_size:
            raise ValueError(
                'position IDs must have shape [sequence, axes] or '
                '[batch, sequence, axes]'
            )
        return value

    text = batched(text_position_idx)
    image = batched(image_position_idx)
    if text.shape[-1] != image.shape[-1]:
        raise ValueError('text and image position IDs must use the same axes')
    return jnp.concatenate((text, image), axis=1)


def pairwise_attention_mask(
    attention_mask: jax.Array | None,
) -> jax.Array | None:
    if attention_mask is None:
        return None
    attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
    if attention_mask.ndim == 2:
        return (
            attention_mask[:, None, None, :]
            & attention_mask[:, None, :, None]
        )
    return attention_mask


def flatten_modulation(
    modulation: jax.Array,
    *,
    batch_size: int,
    groups: int,
    hidden_size: int,
    name: str = 'modulation',
) -> jax.Array:
    modulation = jnp.asarray(modulation)
    expected = groups * hidden_size
    if modulation.shape == (batch_size, groups, hidden_size):
        return modulation.reshape(batch_size, expected)
    if modulation.shape == (batch_size, expected):
        return modulation
    raise ValueError(
        f'{name} must have shape [{batch_size}, {groups}, {hidden_size}] '
        f'or [{batch_size}, {expected}], got {modulation.shape}'
    )


def _approximate_gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x, approximate=True)
