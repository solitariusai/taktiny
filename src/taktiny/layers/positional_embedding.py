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
from collections.abc import Mapping
from typing import Any
import math
import jax
import jax.numpy as jnp
from taktiny import nn
from taktiny.nn.utils import _constrain
from taktiny.utils.typing import DType, ShardMode

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

class SinusoidalPositionalEmbedding(nn.Module):
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
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError('embedding_dim must be a positive integer')
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

        self.embedding_dim = embedding_dim
        self.max_period = float(max_period)
        self.frequency_shift = float(frequency_shift)
        self.flip_sin_to_cos = flip_sin_to_cos
        self.scale = float(scale)
        self.dtype = dtype
        self.shard_mode = shard_mode

    def __call__(
        self,
        positions: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        positions = jnp.asarray(positions, dtype=jnp.float32) * self.scale
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            embedding = jnp.empty((*positions.shape, 0), dtype=jnp.float32)
        else:
            denominator = (
                1.0
                if half_dim == 1
                else half_dim - self.frequency_shift
            )
            frequencies = jnp.exp(
                -math.log(self.max_period)
                * jnp.arange(half_dim, dtype=jnp.float32)
                / denominator
            )
            angles = positions[..., None] * frequencies
            sin = jnp.sin(angles)
            cos = jnp.cos(angles)
            components = (cos, sin) if self.flip_sin_to_cos else (sin, cos)
            embedding = jnp.concatenate(components, axis=-1)

        if self.embedding_dim % 2 == 1:
            embedding = jnp.concatenate(
                [
                    embedding,
                    jnp.zeros((*positions.shape, 1), dtype=jnp.float32),
                ],
                axis=-1,
            )

        embedding = embedding.astype(self.dtype)
        return _constrain(embedding, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return f'{self.embedding_dim}, max_period={self.max_period:g}'


__all__ = [
    'rotate_half',
    'RotaryEmbedding',
    'SinusoidalPositionalEmbedding',
]
