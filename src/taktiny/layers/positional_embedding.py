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
"""Position embedding modules"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Literal
import math
import jax
import jax.numpy as jnp
from taktiny import nn
from taktiny.nn._continuo import _constrain
from taktiny.utils.typing import AxisNames, DType, ShardMode

def rotate_half(x: jax.Array) -> jax.Array:
    """Rotates half the hidden dims of the input."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        rope_scaling: Mapping[str, Any] | None = None,
    ) -> None:
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        position_idx: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        # q and k are expected to have shape [batch, seq_len, num_heads, head_dim]
        seq_len = q.shape[1]

        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))

        if self.rope_scaling is not None and self.rope_scaling.get("rope_type") == "llama3":
            import math
            factor = self.rope_scaling.get("factor", 8.0)
            low_freq_factor = self.rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = self.rope_scaling.get("high_freq_factor", 4.0)
            old_context_len = self.rope_scaling.get("original_max_position_embeddings", 8192)

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq

            inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama

            is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
            inv_freq = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        positions = jnp.arange(seq_len, dtype=jnp.float32)
        if position_idx is not None:
            position_idx = jnp.asarray(position_idx, dtype=jnp.float32)
            if position_idx.ndim == 0:
                positions = positions + position_idx
            elif position_idx.ndim == 1:
                positions = position_idx[:, None] + positions[None, :]
            elif position_idx.ndim == 2:
                if position_idx.shape[1] != seq_len:
                    raise ValueError(
                        'per-token position_ids must match sequence length'
                    )
                positions = position_idx
            else:
                raise ValueError(
                    'position_idx must be a scalar, batch vector, or '
                    'per-token matrix'
                )

        if positions.ndim == 1:
            freqs = jnp.einsum('s,d->sd', positions, inv_freq)
        else:
            freqs = jnp.einsum('bs,d->bsd', positions, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        if emb.ndim == 2:
            cos = jnp.cos(emb)[None, :, None, :].astype(q.dtype)
            sin = jnp.sin(emb)[None, :, None, :].astype(q.dtype)
        else:
            cos = jnp.cos(emb)[:, :, None, :].astype(q.dtype)
            sin = jnp.sin(emb)[:, :, None, :].astype(q.dtype)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


class MultiAxisRotaryEmbedding(nn.Module):
    """Apply rotary embeddings whose head width is split across position axes.

    Position IDs use ``[sequence, axes]`` or ``[batch, sequence, axes]``
    layout. Each position axis rotates its corresponding slice of the query
    and key head dimensions independently.
    """

    def __init__(
        self,
        axes_dim: Sequence[int],
        *,
        theta: float = 10_000.0,
    ) -> None:
        self.axes_dim = tuple(axes_dim)
        self.theta = float(theta)
        if self.theta <= 0:
            raise ValueError('theta must be positive')
        if not self.axes_dim or any(
            not isinstance(size, int) or size <= 0 or size % 2
            for size in self.axes_dim
        ):
            raise ValueError('axes_dim must contain positive even integers')

    @staticmethod
    def _rotate_pairs(value: jax.Array) -> jax.Array:
        pairs = value.reshape(*value.shape[:-1], -1, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        return jnp.stack((-odd, even), axis=-1).reshape(value.shape)

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        position_idx: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if position_idx is None:
            return query, key
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError(
                'query and key must use [batch, sequence, heads, head_dim]'
            )
        if query.shape[:2] != key.shape[:2]:
            raise ValueError('query and key batch and sequence axes must match')
        if query.shape[-1] != sum(self.axes_dim) or key.shape[-1] != sum(
            self.axes_dim
        ):
            raise ValueError(
                'sum(axes_dim) must equal the attention head dimension'
            )

        positions = jnp.asarray(position_idx, dtype=jnp.float32)
        if positions.ndim == 2:
            positions = positions[None, ...]
        expected_tail = (query.shape[1], len(self.axes_dim))
        if positions.ndim != 3 or positions.shape[1:] != expected_tail:
            raise ValueError(
                'position_idx must have shape [sequence, axes] or '
                '[batch, sequence, axes]'
            )
        if positions.shape[0] not in {1, query.shape[0]}:
            raise ValueError('position_idx batch size does not match query')

        cosines: list[jax.Array] = []
        sines: list[jax.Array] = []
        for axis, size in enumerate(self.axes_dim):
            frequencies = 1.0 / (
                self.theta
                ** (jnp.arange(0, size, 2, dtype=jnp.float32) / size)
            )
            phase = positions[..., axis, None] * frequencies
            phase = jnp.repeat(phase, 2, axis=-1)
            cosines.append(jnp.cos(phase))
            sines.append(jnp.sin(phase))
        cosine = jnp.concatenate(cosines, axis=-1)[:, :, None, :]
        sine = jnp.concatenate(sines, axis=-1)[:, :, None, :]

        def rotate(value: jax.Array) -> jax.Array:
            return (
                value * cosine.astype(value.dtype)
                + self._rotate_pairs(value) * sine.astype(value.dtype)
            )

        return rotate(query), rotate(key)

    def extra_repr(self) -> str:
        return f'theta={self.theta:g}, axes_dim={self.axes_dim}'

class FrequencyEmbedding(nn.Module):
    """Encode scalar values with deterministic or Gaussian frequencies.

    ``kind='sinusoidal'`` uses the exponentially spaced frequencies commonly
    used for timestep and positional embeddings. ``kind='gaussian'`` samples
    a Fourier basis and stores it as a parameter, which can optionally be
    trainable. Inputs of any shape receive one trailing embedding dimension.

    Args:
        embedding_dim: Size of the generated embedding.
        kind: ``'sinusoidal'`` or ``'gaussian'``.
        max_period: Largest sinusoidal wavelength.
        frequency_shift: Sinusoidal frequency denominator adjustment.
        scale: Multiplier applied to frequencies.
        flip_sin_to_cos: Emit cosine features before sine features.
        log_input: Apply a logarithm before frequency projection.
        trainable: Whether a Gaussian frequency basis is trainable.
        dtype: Output and Gaussian parameter dtype.
        rngs: Required when ``kind='gaussian'``.
        axis_names: Optional logical axis for the Gaussian frequency vector.
        shard_mode: Automatic or explicit output-sharding behavior.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        kind: Literal['sinusoidal', 'gaussian'] = 'sinusoidal',
        max_period: float = 10_000.0,
        frequency_shift: float = 1.0,
        flip_sin_to_cos: bool = False,
        scale: float = 1.0,
        log_input: bool = False,
        trainable: bool = False,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs | None = None,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError('embedding_dim must be a positive integer')
        if kind not in {'sinusoidal', 'gaussian'}:
            raise ValueError("kind must be 'sinusoidal' or 'gaussian'")
        if not math.isfinite(max_period) or max_period <= 0:
            raise ValueError('max_period must be finite and positive')
        if not math.isfinite(frequency_shift):
            raise ValueError('frequency_shift must be finite')
        if not math.isfinite(scale):
            raise ValueError('scale must be finite')

        half_dim = embedding_dim // 2
        if half_dim > 1 and half_dim - frequency_shift <= 0:
            raise ValueError(
                'frequency_shift must be smaller than half the embedding '
                'dimension'
            )
        if axis_names is not None and len(axis_names) != 1:
            raise ValueError('axis_names must contain one frequency axis')
        if kind == 'sinusoidal' and trainable:
            raise ValueError('only Gaussian frequencies can be trainable')
        if kind == 'sinusoidal' and axis_names is not None:
            raise ValueError('axis_names only applies to Gaussian frequencies')

        self.embedding_dim = embedding_dim
        self.kind = kind
        self.max_period = float(max_period)
        self.frequency_shift = float(frequency_shift)
        self.flip_sin_to_cos = flip_sin_to_cos
        self.scale = float(scale)
        self.log_input = bool(log_input)
        self.dtype = dtype
        self.shard_mode = shard_mode

        if kind == 'gaussian':
            if rngs is None:
                raise ValueError(
                    "rngs is required when kind='gaussian'"
                )
            frequencies = jax.random.normal(
                rngs(),
                (half_dim,),
                dtype=dtype,
            )
            self.frequencies = nn.Parameter(
                frequencies,
                trainable=trainable,
            )
            if axis_names is not None:
                self.frequencies.axis_names = tuple(axis_names)

    def _frequencies(self) -> jax.Array:
        half_dim = self.embedding_dim // 2
        if self.kind == 'gaussian':
            return self.frequencies.value.astype(jnp.float32) * self.scale
        denominator = (
            1.0
            if half_dim <= 1
            else half_dim - self.frequency_shift
        )
        return jnp.exp(
            -math.log(self.max_period)
            * jnp.arange(half_dim, dtype=jnp.float32)
            / denominator
        ) * self.scale

    def __call__(
        self,
        values: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        values = jnp.asarray(values, dtype=jnp.float32)
        if self.log_input:
            values = jnp.log(values)
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            embedding = jnp.empty((*values.shape, 0), dtype=jnp.float32)
        else:
            angles = values[..., None] * self._frequencies()
            if self.kind == 'gaussian':
                angles = angles * (2.0 * math.pi)
            sin = jnp.sin(angles)
            cos = jnp.cos(angles)
            components = (cos, sin) if self.flip_sin_to_cos else (sin, cos)
            embedding = jnp.concatenate(components, axis=-1)

        if self.embedding_dim % 2 == 1:
            embedding = jnp.concatenate(
                [
                    embedding,
                    jnp.zeros((*values.shape, 1), dtype=jnp.float32),
                ],
                axis=-1,
            )

        embedding = embedding.astype(self.dtype)
        return _constrain(embedding, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return f'{self.embedding_dim}, kind={self.kind}'


class SinusoidalPositionalEmbedding(FrequencyEmbedding):
    """Encode scalar or tensor positions as sinusoidal Fourier features."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        max_period: float = 10_000.0,
        frequency_shift: float = 1.0,
        flip_sin_to_cos: bool = False,
        scale: float = 1.0,
        dtype: DType = jnp.float32,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        super().__init__(
            embedding_dim,
            kind='sinusoidal',
            max_period=max_period,
            frequency_shift=frequency_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            scale=scale,
            dtype=dtype,
            shard_mode=shard_mode,
        )

    def extra_repr(self) -> str:
        return f'{self.embedding_dim}, max_period={self.max_period:g}'


__all__ = [
    'FrequencyEmbedding',
    'MultiAxisRotaryEmbedding',
    'rotate_half',
    'RotaryEmbedding',
    'SinusoidalPositionalEmbedding',
]
