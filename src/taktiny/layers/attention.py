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
"""Attention modules"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import math

from taktiny import nn
from taktiny.layers.positional_embedding import RotaryEmbedding
from taktiny.utils.typing import AxisNames, DType, ShardMode, Sharding

class SegmentIds(NamedTuple):
    """Compact query and key/value segment identifiers."""
    q: jax.Array
    kv: jax.Array

class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        context_dim: int | None = None,
        pos_emb: nn.Module | None = None,
        bias: bool = False,
        use_qkv_norm: bool = False,
        qkv_norm_eps: float = 1e-5,
        dtype: DType | str | None = None,
        window_size: int | None = None,
        rngs: nn.Rngs | None = None,
        q_axis_names: AxisNames | None = None,
        k_axis_names: AxisNames | None = None,
        v_axis_names: AxisNames | None = None,
        o_axis_names: AxisNames | None = None,
        q_bias: bool | None = None,
        k_bias: bool | None = None,
        v_bias: bool | None = None,
        o_bias: bool | None = None,
        scaling: float | None = None,
        softcap: float | None = None,
        dropout: float | int = 0.0,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any=None,
        dot_general: Any=None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                'num_heads must be divisible by num_kv_heads, got '
                f'{self.num_heads} and {self.num_kv_heads}'
            )
        self.context_dim = hidden_size if context_dim is None else context_dim
        self.use_qkv_norm = use_qkv_norm
        self.qkv_norm_eps = qkv_norm_eps
        self.window_size = window_size
        self.scaling = scaling
        self.softcap = softcap
        self.dropout = dropout

        # For Grouped Query Attention (GQA)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.pos_emb = pos_emb
        if bias and any(
            value is not None
            for value in (q_bias, k_bias, v_bias, o_bias)
        ):
            bias = False
            q_bias = q_bias is True
            k_bias = k_bias is True
            v_bias = v_bias is True
            o_bias = o_bias is True

        # Projections (Leveraging General Linear!)
        self.q_proj = nn.Linear(
            hidden_size, (self.num_heads, self.head_dim),
            dtype=dtype, bias=q_bias or bias, rngs=rngs,
            axis_names=q_axis_names, shard_mode=shard_mode,
            quant=quant, dot_general=dot_general
        )
        self.k_proj = nn.Linear(
            self.context_dim, (self.num_kv_heads, self.head_dim),
            dtype=dtype, bias=k_bias or bias, rngs=rngs, axis_names=k_axis_names,
            shard_mode=shard_mode, quant=quant, dot_general=dot_general
        )
        self.v_proj = nn.Linear(
            self.context_dim, (self.num_kv_heads, self.head_dim),
            dtype=dtype, bias=v_bias or bias, rngs=rngs, axis_names=v_axis_names,
            shard_mode=shard_mode, quant=quant, dot_general=dot_general
        )
        self.o_proj = nn.Linear(
            (self.num_heads, self.head_dim), hidden_size,
            dtype=dtype, bias=o_bias or bias, rngs=rngs, axis_names=o_axis_names,
            shard_mode=shard_mode, quant=quant, dot_general=dot_general
        )

        self.q_norm = self.k_norm = None

        if getattr(self, 'use_qkv_norm', False) or getattr(self, 'use_q_norm', False):
            self.q_norm = nn.RMSNorm(
                self.head_dim,
                eps=self.qkv_norm_eps,
                dtype=dtype,
                axis_names=('head_dim',),
                shard_mode=shard_mode,
            )

        if getattr(self, 'use_qkv_norm', False) or getattr(self, 'use_k_norm', False):
            self.k_norm = nn.RMSNorm(
                self.head_dim,
                eps=self.qkv_norm_eps,
                dtype=dtype,
                axis_names=('head_dim',),
                shard_mode=shard_mode,
            )

    def _scale_query(
        self,
        query: jax.Array,
        position_idx: jax.Array | None = None,
    ) -> jax.Array:
        return query

    @staticmethod
    def _validate_qkv(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
    ) -> tuple[int, int, int, int, int, int]:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                'Attention kernels expect query, key, and value in '
                '[batch, sequence, heads, head_dim] layout'
            )

        batch_size, query_length, query_heads, head_dim = query.shape
        key_batch, key_length, key_heads, key_head_dim = key.shape
        if value.shape[:3] != (key_batch, key_length, key_heads):
            raise ValueError(
                'key and value must have matching batch, sequence, and head '
                f'dimensions, got {key.shape} and {value.shape}'
            )
        if batch_size != key_batch or head_dim != key_head_dim:
            raise ValueError(
                'query and key must have matching batch and head dimensions, '
                f'got {query.shape} and {key.shape}'
            )
        if query_heads % key_heads != 0:
            raise ValueError(
                'query heads must be divisible by key/value heads, got '
                f'{query_heads} and {key_heads}'
            )
        return (
            batch_size,
            query_length,
            key_length,
            query_heads,
            key_heads,
            head_dim,
        )

    @staticmethod
    def _merge_causal_mask(
        mask: jax.Array | None,
        query_length: int,
        key_length: int,
    ) -> jax.Array:
        causal_mask = (
            jnp.arange(key_length)[None, :]
            <= jnp.arange(query_length)[:, None]
        )
        if mask is None:
            return causal_mask
        return jnp.asarray(mask, dtype=jnp.bool_) & causal_mask

    @staticmethod
    def _broadcast_attention_mask(
        mask: jax.Array | None,
        *,
        batch_size: int,
        num_heads: int,
        query_length: int,
        key_length: int,
    ) -> jax.Array:
        if mask is None:
            mask = jnp.ones(
                (query_length, key_length),
                dtype=jnp.bool_,
            )
        mask = jnp.asarray(mask, dtype=jnp.bool_)
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        elif mask.ndim != 4:
            raise ValueError(
                'attention mask must have 2, 3, or 4 dimensions, got '
                f'{mask.ndim}'
            )
        try:
            return jnp.broadcast_to(
                mask,
                (batch_size, num_heads, query_length, key_length),
            )
        except ValueError as error:
            raise ValueError(
                'attention mask is not broadcastable to '
                f'[{batch_size}, {num_heads}, {query_length}, {key_length}], '
                f'got {mask.shape}'
            ) from error

    @classmethod
    def _normalize_segment_ids(
        cls,
        segment_ids: SegmentIds | jax.Array | None,
        *,
        batch_size: int,
        query_length: int,
        key_length: int,
    ) -> SegmentIds | None:
        if segment_ids is None:
            return None
        if hasattr(segment_ids, 'q') and hasattr(segment_ids, 'kv'):
            query_segments = jnp.asarray(segment_ids.q, dtype=jnp.int32)
            key_segments = jnp.asarray(segment_ids.kv, dtype=jnp.int32)
        else:
            query_segments = jnp.asarray(segment_ids, dtype=jnp.int32)
            key_segments = query_segments

        if query_segments.ndim == 1:
            query_segments = query_segments[None, :]
        if key_segments.ndim == 1:
            key_segments = key_segments[None, :]
        try:
            query_segments = jnp.broadcast_to(
                query_segments,
                (batch_size, query_length),
            )
            key_segments = jnp.broadcast_to(
                key_segments,
                (batch_size, key_length),
            )
        except ValueError as error:
            raise ValueError(
                'segment IDs must be broadcastable to [batch, sequence]'
            ) from error
        return SegmentIds(query_segments, key_segments)

    @staticmethod
    def _merge_segment_mask(
        mask: jax.Array | None,
        segment_ids: SegmentIds,
    ) -> jax.Array:
        segment_mask = (
            segment_ids.q[:, None, :, None]
            == segment_ids.kv[:, None, None, :]
        )
        if mask is None:
            return segment_mask
        return jnp.asarray(mask, dtype=jnp.bool_) & segment_mask

    @classmethod
    def apply_flash_attention(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        segment_ids: SegmentIds | jax.Array | None = None,
        block_kv: int = 128,
        block_q: int = 128,
        mask: jax.Array | None = None,
        mask_value: float = -1e9,
        cap: float | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply FlashAttention block masked kernel on Q, K, V."""
        from taktiny.kernels.attention.flash_attention import flash_attention_block_masked
        (
            batch_size,
            query_length,
            key_length,
            query_heads,
            _,
            head_dim,
        ) = cls._validate_qkv(query, key, value)
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        if mask is not None:
            mask = jnp.asarray(mask, dtype=jnp.bool_)
            if mask.ndim == 4:
                if mask.shape[-3] != 1:
                    raise ValueError(
                        'Flash attention supports only masks shared across '
                        f'heads, got {mask.shape}'
                    )
                mask = mask[..., 0, :, :]
            if mask.ndim not in (2, 3):
                raise ValueError(
                    'Flash attention mask must have shape [query, key], '
                    '[batch, query, key], or [batch, 1, query, key]'
                )

        q_in = query.transpose(0, 2, 1, 3) * scale
        k_in = key.transpose(0, 2, 1, 3)
        v_in = value.transpose(0, 2, 1, 3)
        bq = math.gcd(query_length, min(block_q, query_length))
        bk = math.gcd(key_length, min(block_kv, key_length))

        out = flash_attention_block_masked(
            q_in, k_in, v_in,
            segment_ids=segment_ids,
            block_kv=bk,
            block_q=bq,
            mask=mask,
            mask_value=mask_value,
            cap=cap,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(
            batch_size,
            query_heads,
            query_length,
            value.shape[-1],
        ).transpose(0, 2, 1, 3)

    @classmethod
    def apply_ragged_attention(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        lengths: jax.Array | None = None,
        scale: float | None = None,
        block_size: int = 256,
        interpret: bool | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply the decode-only Ragged Attention kernel."""
        from taktiny.kernels.attention.ragged_attention import (
            ragged_gqa,
            ragged_mha,
        )
        (
            batch_size,
            query_length,
            key_length,
            query_heads,
            key_heads,
            head_dim,
        ) = cls._validate_qkv(query, key, value)
        if query_length != 1:
            raise ValueError(
                'Ragged attention is a decode kernel and requires query '
                f'length 1, got {query_length}'
            )
        if lengths is None:
            lengths = jnp.full(
                (batch_size,),
                key_length,
                dtype=jnp.int32,
            )
        else:
            lengths = jnp.asarray(lengths)
        if lengths.shape != (batch_size,) or lengths.dtype != jnp.int32:
            raise ValueError(
                'lengths must have shape [batch] and dtype int32, got '
                f'{lengths.shape} and {lengths.dtype}'
            )
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        if interpret is None:
            interpret = jax.default_backend() == 'cpu'
        block_size = math.gcd(
            key_length,
            min(block_size, key_length),
        )

        ragged = ragged_mha if query_heads == key_heads else ragged_gqa
        out, _, denominator = ragged(
            query * scale,
            key,
            value,
            lengths,
            block_size=block_size,
            interpret=interpret,
            **kwargs,
        )
        return out / denominator

    @classmethod
    def apply_splash_attention(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask: jax.Array | None = None,
        segment_ids: SegmentIds | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply the Splash Attention reference with dense masks."""
        from taktiny.kernels.attention.splash_attention import attention_reference

        (
            batch_size,
            query_length,
            key_length,
            query_heads,
            key_heads,
            head_dim,
        ) = cls._validate_qkv(query, key, value)
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        mask = cls._broadcast_attention_mask(
            mask,
            batch_size=batch_size,
            num_heads=query_heads,
            query_length=query_length,
            key_length=key_length,
        )
        if segment_ids is not None:
            query_segments = jnp.asarray(segment_ids.q)
            key_segments = jnp.asarray(segment_ids.kv)
            if query_segments.ndim == 1:
                query_segments = query_segments[None, :]
            if key_segments.ndim == 1:
                key_segments = key_segments[None, :]
            try:
                query_segments = jnp.broadcast_to(
                    query_segments,
                    (batch_size, query_length),
                )
                key_segments = jnp.broadcast_to(
                    key_segments,
                    (batch_size, key_length),
                )
            except ValueError as error:
                raise ValueError(
                    'segment IDs must be broadcastable to [batch, sequence]'
                ) from error
            mask = mask & (
                query_segments[:, None, :, None]
                == key_segments[:, None, None, :]
            )

        kv_head_indices = (
            jnp.arange(query_heads)
            // (query_heads // key_heads)
        )
        query = query.transpose(0, 2, 1, 3) * scale
        key = jnp.take(
            key.transpose(0, 2, 1, 3),
            kv_head_indices,
            axis=1,
        )
        value = jnp.take(
            value.transpose(0, 2, 1, 3),
            kv_head_indices,
            axis=1,
        )

        def apply_head(
            q: jax.Array,
            k: jax.Array,
            v: jax.Array,
            head_mask: jax.Array,
        ) -> jax.Array:
            out = attention_reference(
                head_mask,
                q,
                k,
                v,
                None,
                **kwargs,
            )
            return out[0] if isinstance(out, tuple) else out

        out = jax.vmap(
            jax.vmap(apply_head),
        )(query, key, value, mask)
        return out.transpose(0, 2, 1, 3)

    @classmethod
    def apply_ring_attention(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        *,
        ring_kernel: Callable[..., jax.Array] | None = None,
        segment_ids: SegmentIds | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply a prebuilt Ring Splash Attention kernel."""
        from taktiny.kernels.attention.tokamax_splash import (
            ring_attention_kernel,
        )

        (
            batch_size,
            _,
            _,
            _,
            _,
            head_dim,
        ) = cls._validate_qkv(query, key, value)
        if ring_kernel is None:
            raise ValueError(
                'ring_kernel is required; construct it with '
                'make_ring_attention before calling Attention.apply'
            )
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        query = query.transpose(0, 2, 1, 3) * scale
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        if segment_ids is None:
            out = jax.vmap(
                lambda q, k, v: ring_kernel(q, k, v, None, **kwargs)
            )(query, key, value)
        else:
            query_segments = jnp.asarray(segment_ids.q)
            key_segments = jnp.asarray(segment_ids.kv)
            if query_segments.shape[0] != batch_size:
                raise ValueError(
                    'batched ring segment IDs must match the query batch'
                )

            def apply_one(
                q: jax.Array,
                k: jax.Array,
                v: jax.Array,
                q_segments: jax.Array,
                kv_segments: jax.Array,
            ) -> jax.Array:
                segments = ring_attention_kernel.SegmentIds(
                    q_segments,
                    kv_segments,
                )
                return ring_kernel(q, k, v, segments, **kwargs)

            out = jax.vmap(apply_one)(
                query,
                key,
                value,
                query_segments,
                key_segments,
            )
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(0, 2, 1, 3)

    @classmethod
    def apply(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        kernel: str = "dot_product",
        mask: jax.Array | None = None,
        bias: jax.Array | None = None,
        scale: float | None = None,
        is_causal: bool = False,
        segment_ids: SegmentIds | jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """
        Unified Entry Point for Attention Kernel Applications.
        Supported kernel methods: 'dot_product', 'flash', 'ragged', 'splash', 'ring'.
        """
        if not isinstance(kernel, str):
            raise TypeError(
                f'kernel must be a string, got {type(kernel).__name__}'
        )
        kernel = kernel.lower()
        segment_ids = cls._normalize_segment_ids(
            segment_ids,
            batch_size=query.shape[0],
            query_length=query.shape[1],
            key_length=key.shape[1],
        )
        if kernel in ("dot_product", "standard", "jax"):
            if segment_ids is not None:
                mask = cls._merge_segment_mask(mask, segment_ids)
            return jax.nn.dot_product_attention(
                query=query,
                key=key,
                value=value,
                bias=bias,
                mask=mask,
                scale=scale,
                is_causal=is_causal,
                **kwargs,
            )
        elif kernel in ("flash", "flash_attention"):
            if bias is not None:
                raise ValueError(
                    'Flash attention does not support additive bias'
                )
            if is_causal:
                mask = cls._merge_causal_mask(
                    mask,
                    query.shape[1],
                    key.shape[1],
                )
            return cls.apply_flash_attention(
                query,
                key,
                value,
                mask=mask,
                scale=scale,
                segment_ids=segment_ids,
                **kwargs,
            )
        elif kernel in ("ragged", "ragged_attention"):
            if mask is not None or bias is not None or segment_ids is not None:
                raise ValueError(
                    'Ragged attention uses lengths for prefix masking and '
                    'does not support mask, bias, or segment IDs'
                )
            return cls.apply_ragged_attention(
                query,
                key,
                value,
                scale=scale,
                **kwargs,
            )
        elif kernel in ("splash", "splash_attention"):
            if bias is not None:
                raise ValueError(
                    'Splash attention does not support additive bias'
                )
            if is_causal:
                mask = cls._merge_causal_mask(
                    mask,
                    query.shape[1],
                    key.shape[1],
                )
            return cls.apply_splash_attention(
                query,
                key,
                value,
                mask=mask,
                scale=scale,
                segment_ids=segment_ids,
                **kwargs,
            )
        elif kernel in ("ring", "ring_attention"):
            if mask is not None or bias is not None:
                raise ValueError(
                    'Ring attention masking is fixed when ring_kernel is '
                    'constructed; mask and bias cannot be supplied at call time'
                )
            return cls.apply_ring_attention(
                query,
                key,
                value,
                scale=scale,
                segment_ids=segment_ids,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown attention kernel method: '{kernel}'. Choice of ['dot_product', 'flash', 'ragged', 'splash', 'ring']"
            )

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        is_causal: bool = False,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
        position_idx: jax.Array | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
        kernel: str = "dot_product",
    ) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array]]:

        context_in = context if context is not None else x

        # Project directly to [B, L, Heads, HeadDim] thanks to General Linear
        q = self.q_proj(x)
        k = self.k_proj(context_in)
        v = self.v_proj(context_in)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Apply Positional Embeddings (if provided)
        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, position_idx)

        q = self._scale_query(q, position_idx)

        segment_ids = None
        if position_idx is not None:
            token_positions = jnp.asarray(position_idx, dtype=jnp.int32)
            if token_positions.ndim == 2:
                expected_shape = (q.shape[0], q.shape[1])
                if token_positions.shape != expected_shape:
                    raise ValueError(
                        'position_ids must have shape '
                        f'{expected_shape}, got {token_positions.shape}'
                    )
                packed_segments = jnp.cumsum(
                    token_positions == 0,
                    axis=-1,
                    dtype=jnp.int32,
                )
                segment_ids = SegmentIds(
                    packed_segments,
                    packed_segments,
                )

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            cache_position = jnp.asarray(position_idx, dtype=jnp.int32)
            if cache_position.ndim == 0:
                k_cache = jax.lax.dynamic_update_slice(
                    k_cache,
                    k,
                    (0, cache_position, 0, 0),
                )
                v_cache = jax.lax.dynamic_update_slice(
                    v_cache,
                    v,
                    (0, cache_position, 0, 0),
                )
            elif cache_position.ndim == 1:
                def update_cache(
                    cache_row: jax.Array,
                    value_row: jax.Array,
                    row_position: jax.Array,
                ) -> jax.Array:
                    return jax.lax.dynamic_update_slice(
                        cache_row,
                        value_row,
                        (row_position, 0, 0),
                    )

                k_cache = jax.vmap(update_cache)(
                    k_cache,
                    k,
                    cache_position,
                )
                v_cache = jax.vmap(update_cache)(
                    v_cache,
                    v,
                    cache_position,
                )
            else:
                raise ValueError(
                    'position_idx must be a scalar or a batch vector'
                )

            # Use the full cache for attention
            k = k_cache
            v = v_cache

        # Sliding Window / Causal Masking
        if is_causal or self.window_size is not None:
            if self.window_size is not None:
                q_len = q.shape[1]
                k_len = k.shape[1]

                query_start = (
                    jnp.asarray(0, dtype=jnp.int32)
                    if position_idx is None
                    else jnp.asarray(position_idx, dtype=jnp.int32)
                )
                query_offsets = jnp.arange(q_len, dtype=jnp.int32)
                if query_start.ndim == 0:
                    query_positions = query_start + query_offsets
                elif query_start.ndim == 1:
                    query_positions = (
                        query_start[:, None] + query_offsets[None, :]
                    )
                elif query_start.ndim == 2:
                    if query_start.shape != (q.shape[0], q_len):
                        raise ValueError(
                            'position_ids must match [batch, sequence]'
                        )
                    query_positions = query_start
                else:
                    raise ValueError(
                        'position_idx must be a scalar, batch vector, or '
                        'per-token matrix'
                    )

                # Cached keys use absolute positions from the start of the
                # sequence. Uncached keys belong to the same local chunk as Q.
                if kv_cache is not None:
                    key_positions = jnp.arange(k_len, dtype=jnp.int32)
                elif query_start.ndim == 2:
                    key_positions = query_start
                else:
                    key_positions = query_start + jnp.arange(
                        k_len,
                        dtype=jnp.int32,
                    )

                if query_positions.ndim == 1:
                    causal_mask = (
                        key_positions[None, :]
                        <= query_positions[:, None]
                    )
                    window_mask = key_positions[None, :] >= (
                        query_positions[:, None] - self.window_size + 1
                    )
                else:
                    if key_positions.ndim == 1:
                        key_positions = key_positions[None, :]
                    causal_mask = (
                        key_positions[:, None, :]
                        <= query_positions[:, :, None]
                    )
                    window_mask = key_positions[:, None, :] >= (
                        query_positions[:, :, None]
                        - self.window_size
                        + 1
                    )
                    causal_mask = causal_mask[:, None, :, :]
                    window_mask = window_mask[:, None, :, :]
                sliding_mask = causal_mask & window_mask

                if attention_mask is not None:
                    attention_mask = attention_mask & sliding_mask
                else:
                    attention_mask = sliding_mask

                # The absolute-position mask handles causality itself.
                is_causal = False

        attention_bias = None
        if self.softcap is not None:
            scale = (
                self.scaling
                if self.scaling is not None
                else self.head_dim ** -0.5
            )
            batch_size, query_length, _, _ = q.shape
            key_length = k.shape[1]
            grouped_q = q.reshape(
                batch_size,
                query_length,
                self.num_kv_heads,
                self.num_kv_groups,
                self.head_dim,
            )
            scores = jnp.einsum(
                'btkgh,bskh->bkgts',
                grouped_q,
                k,
            ) * scale
            capped_scores = self.softcap * jnp.tanh(scores / self.softcap)
            attention_bias = (capped_scores - scores).reshape(
                batch_size,
                self.num_heads,
                query_length,
                key_length,
            )

        # Apply attention using requested kernel via Attention.apply entry point!
        out = self.apply(
            query=q,
            key=k,
            value=v,
            kernel=kernel,
            bias=attention_bias,
            mask=attention_mask,
            scale=self.scaling,
            is_causal=is_causal,
            segment_ids=segment_ids,
        )

        # Output projection from (Batch, SeqLen, Heads, HeadDim) directly to (Batch, SeqLen, HiddenSize)
        out = self.o_proj(out, out_sharding=out_sharding)

        if kv_cache is not None:
            return out, (k_cache, v_cache)

        return out, None


class JointAttention(nn.Module):
    """Attend jointly over two independently projected token streams.

    Each stream owns its query, key, value, and output projections. Projected
    tokens are concatenated along the sequence axis, passed through one shared
    attention operation, and split back into their original streams before the
    output projections. Architectural normalization, modulation, and residual
    policies remain outside this module.

    Inputs use ``[batch, sequence, hidden]`` layout. ``attention_mask`` and
    ``attention_bias`` describe the concatenated sequence of length
    ``sequence1 + sequence2``.
    """

    def __init__(
        self,
        hidden_size1: int,
        hidden_size2: int,
        num_heads: int,
        head_dim: int,
        *,
        pos_emb: nn.Module | None = None,
        bias: bool = False,
        use_qkv_norm: bool = False,
        qkv_norm_eps: float = 1e-5,
        dtype: DType | str | None = None,
        rngs: nn.Rngs | None = None,
        q_axis_names: AxisNames | None = None,
        k_axis_names: AxisNames | None = None,
        v_axis_names: AxisNames | None = None,
        o_axis_names: AxisNames | None = None,
        q_bias: bool | None = None,
        k_bias: bool | None = None,
        v_bias: bool | None = None,
        o_bias: bool | None = None,
        scaling: float | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
    ) -> None:
        if hidden_size1 <= 0 or hidden_size2 <= 0:
            raise ValueError('hidden sizes must be positive')
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError('num_heads and head_dim must be positive')
        if qkv_norm_eps <= 0:
            raise ValueError('qkv_norm_eps must be positive')
        if rngs is None:
            raise ValueError(
                'A rngs must be provided to initialize JointAttention'
            )

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pos_emb = pos_emb
        self.use_qkv_norm = use_qkv_norm
        self.qkv_norm_eps = qkv_norm_eps
        self.scaling = scaling

        def projection_bias(override: bool | None) -> bool:
            return bias if override is None else override

        projection_options = {
            'dtype': dtype,
            'rngs': rngs,
            'shard_mode': shard_mode,
            'quant': quant,
            'dot_general': dot_general,
        }

        self.q_proj_1 = nn.Linear(
            hidden_size1,
            (num_heads, head_dim),
            bias=projection_bias(q_bias),
            axis_names=q_axis_names,
            **projection_options,
        )
        self.k_proj_1 = nn.Linear(
            hidden_size1,
            (num_heads, head_dim),
            bias=projection_bias(k_bias),
            axis_names=k_axis_names,
            **projection_options,
        )
        self.v_proj_1 = nn.Linear(
            hidden_size1,
            (num_heads, head_dim),
            bias=projection_bias(v_bias),
            axis_names=v_axis_names,
            **projection_options,
        )
        self.o_proj_1 = nn.Linear(
            (num_heads, head_dim),
            hidden_size1,
            bias=projection_bias(o_bias),
            axis_names=o_axis_names,
            **projection_options,
        )

        self.q_proj_2 = nn.Linear(
            hidden_size2,
            (num_heads, head_dim),
            bias=projection_bias(q_bias),
            axis_names=q_axis_names,
            **projection_options,
        )
        self.k_proj_2 = nn.Linear(
            hidden_size2,
            (num_heads, head_dim),
            bias=projection_bias(k_bias),
            axis_names=k_axis_names,
            **projection_options,
        )
        self.v_proj_2 = nn.Linear(
            hidden_size2,
            (num_heads, head_dim),
            bias=projection_bias(v_bias),
            axis_names=v_axis_names,
            **projection_options,
        )
        self.o_proj_2 = nn.Linear(
            (num_heads, head_dim),
            hidden_size2,
            bias=projection_bias(o_bias),
            axis_names=o_axis_names,
            **projection_options,
        )

        if use_qkv_norm:
            norm_options = {
                'eps': qkv_norm_eps,
                'dtype': dtype,
                'axis_names': ('head_dim',),
                'shard_mode': shard_mode,
            }
            self.q_norm_1 = nn.RMSNorm(head_dim, **norm_options)
            self.k_norm_1 = nn.RMSNorm(head_dim, **norm_options)
            self.q_norm_2 = nn.RMSNorm(head_dim, **norm_options)
            self.k_norm_2 = nn.RMSNorm(head_dim, **norm_options)
        else:
            self.q_norm_1 = self.k_norm_1 = None
            self.q_norm_2 = self.k_norm_2 = None

    def _validate_inputs(
        self,
        x1: jax.Array,
        x2: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        x1 = jnp.asarray(x1)
        x2 = jnp.asarray(x2)
        if x1.ndim != 3 or x2.ndim != 3:
            raise ValueError(
                'JointAttention inputs must use [batch, sequence, hidden] '
                f'layout, got {x1.shape} and {x2.shape}'
            )
        if x1.shape[0] != x2.shape[0]:
            raise ValueError(
                'JointAttention inputs must have the same batch size, got '
                f'{x1.shape[0]} and {x2.shape[0]}'
            )
        if x1.shape[-1] != self.hidden_size1:
            raise ValueError(
                f'x1 must end in hidden_size1={self.hidden_size1}, got '
                f'{x1.shape}'
            )
        if x2.shape[-1] != self.hidden_size2:
            raise ValueError(
                f'x2 must end in hidden_size2={self.hidden_size2}, got '
                f'{x2.shape}'
            )
        return x1, x2

    @staticmethod
    def _joint_position_idx(
        position_idx: jax.Array | tuple[jax.Array, jax.Array] | None,
    ) -> jax.Array | None:
        if position_idx is None or not isinstance(position_idx, tuple):
            return position_idx
        if len(position_idx) != 2:
            raise ValueError('position_idx must contain exactly two arrays')
        first, second = map(jnp.asarray, position_idx)
        if first.ndim != second.ndim or first.shape[:-1] != second.shape[:-1]:
            raise ValueError(
                'per-stream position arrays must have matching leading shapes'
            )
        return jnp.concatenate((first, second), axis=-1)

    def __call__(
        self,
        x1: jax.Array,
        x2: jax.Array,
        attention_mask: jax.Array | None = None,
        attention_bias: jax.Array | None = None,
        is_causal: bool = False,
        segment_ids: SegmentIds | jax.Array | None = None,
        position_idx: jax.Array | tuple[jax.Array, jax.Array] | None = None,
        out_shardings: tuple[Sharding, Sharding] | None = None,
        kernel: str = 'dot_product',
        **kernel_kwargs: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply joint attention and return one output per input stream."""
        x1, x2 = self._validate_inputs(x1, x2)
        length1 = x1.shape[1]

        q1 = self.q_proj_1(x1)
        k1 = self.k_proj_1(x1)
        v1 = self.v_proj_1(x1)
        q2 = self.q_proj_2(x2)
        k2 = self.k_proj_2(x2)
        v2 = self.v_proj_2(x2)

        if self.q_norm_1 is not None:
            q1, k1 = self.q_norm_1(q1), self.k_norm_1(k1)
            q2, k2 = self.q_norm_2(q2), self.k_norm_2(k2)

        q = jnp.concatenate((q1, q2), axis=1)
        k = jnp.concatenate((k1, k2), axis=1)
        v = jnp.concatenate((v1, v2), axis=1)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, self._joint_position_idx(position_idx))

        out = Attention.apply(
            query=q,
            key=k,
            value=v,
            kernel=kernel,
            mask=attention_mask,
            bias=attention_bias,
            scale=self.scaling,
            is_causal=is_causal,
            segment_ids=segment_ids,
            **kernel_kwargs,
        )
        out1, out2 = jnp.split(out, (length1,), axis=1)

        if out_shardings is None:
            out_sharding1 = out_sharding2 = None
        else:
            if len(out_shardings) != 2:
                raise ValueError('out_shardings must contain exactly two values')
            out_sharding1, out_sharding2 = out_shardings
        out1 = self.o_proj_1(out1, out_sharding=out_sharding1)
        out2 = self.o_proj_2(out2, out_sharding=out_sharding2)

        return out1, out2

    def extra_repr(self) -> str:
        return (
            f'{self.hidden_size1} + {self.hidden_size2}, '
            f'heads={self.num_heads}, head_dim={self.head_dim}'
        )


__all__ = ['Attention', 'JointAttention', 'SegmentIds']
