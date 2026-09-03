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
"""Generic encoder-decoder transformer modules."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import jax
import jax.numpy as jnp
from taktiny import nn
from taktiny.nn.continuo import (
    _constrain,
    _resolve_activation,
    _validate_integer,
    _validate_probability,
)

from taktiny.nn.layers.attention import Attention
from taktiny.nn.layers.ffn import FeedForward
from taktiny.nn.utils import AxisName
from taktiny.utils.typing import (
    QuantConfig,
    Activation,
    DType,
    Sharding,
)

ModuleSpec = nn.Module | type[nn.Module]


def _instantiate_module(
    module: ModuleSpec,
    *,
    name: str,
    options: dict[str, Any],
) -> nn.Module:
    if isinstance(module, nn.Module):
        return module
    if not isinstance(module, type) or not issubclass(module, nn.Module):
        raise TypeError(f'{name} must be an nn.Module subclass or instance')
    parameters = inspect.signature(module).parameters.values()
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if not accepts_kwargs:
        accepted = {
            parameter.name
            for parameter in parameters
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        options = {
            key: value for key, value in options.items() if key in accepted
        }
    return module(**options)

def _normalize_input(
    value: jax.Array,
    *,
    name: str,
    hidden_size: int,
    batch_first: bool,
) -> tuple[jax.Array, bool]:
    value = jnp.asarray(value)
    if value.ndim not in {2, 3}:
        raise ValueError(
            f'{name} must have shape [sequence, hidden] or contain one batch axis'
        )
    if value.shape[-1] != hidden_size:
        raise ValueError(
            f'{name} trailing size must be {hidden_size}, got {value.shape[-1]}'
        )
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(f'{name} must have a floating-point dtype')

    unbatched = value.ndim == 2
    if unbatched:
        value = value[None, ...]
    elif not batch_first:
        value = jnp.swapaxes(value, 0, 1)
    return value, unbatched

def _normalize_attention_mask(
    mask: jax.Array | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    name: str,
) -> jax.Array | None:
    if mask is None:
        return None
    mask = jnp.asarray(mask)
    if mask.dtype != jnp.bool_:
        raise TypeError(f'{name} must be a boolean array')
    if mask.shape == (query_length, key_length):
        return mask[None, None, :, :]
    if mask.shape == (batch_size, query_length, key_length):
        return mask[:, None, :, :]
    target_shape = (batch_size, 1, query_length, key_length)
    try:
        return jnp.broadcast_to(mask, target_shape)
    except ValueError as error:
        raise ValueError(
            f'{name} with shape {mask.shape} cannot broadcast to {target_shape}'
        ) from error

def _normalize_padding_mask(
    mask: jax.Array | None,
    *,
    batch_size: int,
    key_length: int,
    name: str,
) -> jax.Array | None:
    if mask is None:
        return None
    mask = jnp.asarray(mask)
    if mask.dtype != jnp.bool_:
        raise TypeError(f'{name} must be a boolean array')
    if mask.shape == (key_length,) and batch_size == 1:
        mask = mask[None, :]
    if mask.shape != (batch_size, key_length):
        raise ValueError(
            f'{name} must have shape {(batch_size, key_length)}, got {mask.shape}'
        )
    # Padding masks follow the common convention: True marks ignored tokens.
    return (~mask)[:, None, None, :]

def _merge_masks(
    attention_mask: jax.Array | None,
    padding_mask: jax.Array | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    attention_name: str,
    padding_name: str,
) -> jax.Array | None:
    attention_mask = _normalize_attention_mask(
        attention_mask,
        batch_size=batch_size,
        query_length=query_length,
        key_length=key_length,
        name=attention_name,
    )
    padding_mask = _normalize_padding_mask(
        padding_mask,
        batch_size=batch_size,
        key_length=key_length,
        name=padding_name,
    )
    if attention_mask is None:
        return padding_mask
    if padding_mask is None:
        return attention_mask
    return attention_mask & padding_mask

def _apply_causal_mask(
    mask: jax.Array | None,
    is_causal: bool | jax.Array,
    *,
    query_length: int,
    key_length: int,
) -> jax.Array | None:
    if isinstance(is_causal, bool) and not is_causal:
        return mask
    query_positions = jnp.arange(query_length)[:, None]
    key_positions = jnp.arange(key_length)[None, :]
    causal = key_positions <= query_positions
    if not isinstance(is_causal, bool):
        causal = jnp.where(jnp.asarray(is_causal), causal, True)
    causal = causal[None, None, :, :]
    return causal if mask is None else mask & causal

def _attention_output(value: Any) -> jax.Array:
    return value[0] if isinstance(value, tuple) else value

class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        num_kv_heads: int | None = None,
        context_dim: int | None = None,
        apply_position_fn: Callable | None = None,
        bias: bool | list[bool] | tuple[bool] = False,
        q_norm: bool | nn.Module = False,
        k_norm: bool | nn.Module = False,
        qk_norm: bool = False,
        qk_norm_across_heads: bool | list[bool] | tuple[bool] = False,
        epsilon: float = 1e-5,
        window_size: int | None = None,
        scaling: float | None = None,
        softcap: float | None = None,
        dropout: float = 0.0,
        dtype: DType | None = None,
        rngs: nn.Rngs,
        quant: QuantConfig = None,
        axis_names: AxisName | None = None,
        dot_general: Any = None,
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
        self.use_qk_norm = qk_norm
        self.qk_norm_across_heads = qk_norm_across_heads
        self.qk_norm_eps = epsilon
        self.window_size = window_size
        self.scaling = scaling
        self.softcap = softcap
        self.dropout = dropout

        # For Grouped Query Attention (GQA)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.apply_position_fn = apply_position_fn
        q_bias = k_bias = v_bias = o_bias = False
        if isinstance(bias, (list, tuple)):
            if len(bias) != 4:
                raise ValueError('bias must contain q, k, v, and o values')
            if not all(isinstance(value, bool) for value in bias):
                raise TypeError('bias values must be booleans')
            q_bias, k_bias, v_bias, o_bias = bias
            bias = all(bias)
        elif not isinstance(bias, bool):
            raise TypeError('bias must be a boolean or four booleans')

        q_proj_axis_names = None if axis_names is None else axis_names.get('q_proj')
        k_proj_axis_names = None if axis_names is None else axis_names.get('k_proj')
        v_proj_axis_names = None if axis_names is None else axis_names.get('v_proj')
        o_proj_axis_names = None if axis_names is None else axis_names.get('o_proj')

        self.q_proj = nn.Linear(
            hidden_size, 
            (self.num_heads, self.head_dim),
            dtype=dtype, 
            bias=q_bias or bias, 
            rngs=rngs,
            axis_names=q_proj_axis_names,
            quant=quant, 
            dot_general=dot_general
        )
        self.k_proj = nn.Linear(
            self.context_dim, 
            (self.num_kv_heads, self.head_dim),
            dtype=dtype, 
            bias=k_bias or bias, 
            rngs=rngs, 
            axis_names=k_proj_axis_names, 
            quant=quant, 
            dot_general=dot_general
        )
        self.v_proj = nn.Linear(
            self.context_dim, 
            (self.num_kv_heads, self.head_dim),
            dtype=dtype, 
            bias=v_bias or bias, 
            rngs=rngs, 
            axis_names=v_proj_axis_names, 
            quant=quant, 
            dot_general=dot_general
        )
        self.o_proj = nn.Linear(
            (self.num_heads, self.head_dim), 
            hidden_size,
            dtype=dtype, 
            bias=o_bias or bias, 
            rngs=rngs, 
            axis_names=o_proj_axis_names, 
            quant=quant, 
            dot_general=dot_general
        )

        self.q_norm = q_norm if isinstance(q_norm, nn.Module) else None
        self.k_norm = k_norm if isinstance(k_norm, nn.Module) else None
        q_norm_across_heads = k_norm_across_heads = False
        if isinstance(qk_norm_across_heads, (list, tuple)):
            if len(qk_norm_across_heads) != 2:
                raise ValueError(
                    'qk_norm_across_heads must contain q and k values'
                )
            if not all(
                isinstance(value, bool) for value in qk_norm_across_heads
            ):
                raise TypeError('qk_norm_across_heads values must be booleans')
            q_norm_across_heads = qk_norm_across_heads[0]
            k_norm_across_heads = qk_norm_across_heads[1]
        elif not isinstance(qk_norm_across_heads, bool):
            raise TypeError(
                'qk_norm_across_heads must be a boolean or two booleans'
            )

        if self.q_norm is None and (qk_norm or q_norm):
            q_norm_shape = (
                (self.num_heads, self.head_dim)
                if q_norm_across_heads
                else self.head_dim
            )
            q_norm_axis_names = (
                q_proj_axis_names[1:]
                if q_norm_across_heads and q_proj_axis_names is not None
                else (
                    q_proj_axis_names[-1:]
                    if q_proj_axis_names is not None
                    else None
                )
            )
            self.q_norm = nn.RMSNorm(
                q_norm_shape,
                epsilon=epsilon,
                dtype=dtype,
                axis_names=q_norm_axis_names,
                axes=(-2, -1) if q_norm_across_heads else -1,
            )

        if self.k_norm is None and (qk_norm or k_norm):
            k_norm_shape = (
                (self.num_kv_heads, self.head_dim)
                if k_norm_across_heads
                else self.head_dim
            )
            k_norm_axis_names = (
                k_proj_axis_names[1:]
                if k_norm_across_heads and k_proj_axis_names is not None
                else (
                    k_proj_axis_names[-1:]
                    if k_proj_axis_names is not None
                    else None
                )
            )
            self.k_norm = nn.RMSNorm(
                k_norm_shape,
                epsilon=epsilon,
                dtype=dtype,
                axis_names=k_norm_axis_names,
                axes=(-2, -1) if k_norm_across_heads else -1,
            )

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

    @classmethod
    def apply(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        kernel: str = 'dot_product',
        mask: jax.Array | tp.Callable[..., jax.Array] | None = None,
        bias: jax.Array | None = None,
        scale: float | None = None,
        is_causal: bool = False,
        boundary_ids: jax.Array | None = None,
        query_offset: int | jax.Array = 0,
        respect_boundaries: bool = True,
        **kwargs: tp.Any,
    ) -> jax.Array:
        """Use JAX dot-product attention unless another kernel is selected."""
        if not isinstance(kernel, str):
            raise TypeError(
                f'kernel must be a string, got {type(kernel).__name__}'
            )
        kernel = kernel.lower()
        if kernel in ("ragged", "ragged_attention", "splash", "splash_attention", "ring", "ring_attention"):
            import warnings
            warnings.warn(f'{kernel} is unsupported right now fallback to `flash_attention`')
            kernel = 'flash'

        if kernel in ('flash', 'flash_attention'):
            if bias is not None:
                raise ValueError(
                    'boundary FlashAttention does not support additive bias'
                )
            return cls.apply_flash_attention(
                query,
                key,
                value,
                boundary_ids=boundary_ids,
                mask=mask,
                scale=scale,
                is_causal=is_causal,
                query_offset=query_offset,
                respect_boundaries=respect_boundaries,
                **kwargs,
            )

        if boundary_ids is not None or callable(mask):
            from taktiny.cosette.kernels.attention.flash_attention import (
                materialize_attention_mask,
            )

            mask = materialize_attention_mask(
                mask,
                batch_size=query.shape[0],
                num_heads=query.shape[2],
                query_length=query.shape[1],
                key_length=key.shape[1],
                boundary_ids=boundary_ids,
                is_causal=is_causal,
                query_offset=query_offset,
                respect_boundaries=respect_boundaries,
            )
            is_causal = False

        if kernel in ('dot_product', 'standard'):
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

        raise ValueError(
            f"Unknown attention kernel method: '{kernel}'. "
            "Choose 'dot_product' or 'flash'."
        )

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array | tp.Tuple[jax.Array, jax.Array] | None = None,
        attention_mask: jax.Array | tp.Callable[..., jax.Array] | None = None,
        is_causal: bool = False,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
        position_ids: jax.Array | None = None,
        cache_position: jax.Array | None = None,
        position_embedding: jax.Array | None = None,
        boundary_ids: jax.Array | None = None,
        use_sliding_window: bool | jax.Array = False,
        kernel: str = "dot_product",
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        
        context_in = context if context is not None else x

        q = self.q_proj(x)
        k = self.k_proj(
            context_in[0] if isinstance(context_in, tuple) else context_in
        )
        v = self.v_proj(
            context_in[1] if isinstance(context_in, tuple) else context_in
        )

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Apply Positional Embeddings (if provided)
        if self.apply_position_fn is not None:
            q, k = self.apply_position_fn(q, k, position_embedding)

        normalized_cache_position = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache.ndim != k.ndim or v_cache.ndim != v.ndim:
                raise ValueError(
                    'KV caches and projected updates must have equal ranks'
                )
            if k_cache.shape[0] != k.shape[0] or k_cache.shape[2:] != k.shape[2:]:
                raise ValueError(
                    'key cache shape must match key updates except for the '
                    f'sequence axis, got {k_cache.shape} and {k.shape}'
                )
            if v_cache.shape[0] != v.shape[0] or v_cache.shape[2:] != v.shape[2:]:
                raise ValueError(
                    'value cache shape must match value updates except for the '
                    f'sequence axis, got {v_cache.shape} and {v.shape}'
                )

            if cache_position is None:
                if position_ids is None:
                    raise ValueError(
                        'cache_position or position_ids is required with a KV cache'
                    )
                cache_position = position_ids

            normalized_cache_position = jnp.asarray(
                cache_position,
                dtype=jnp.int32,
            )
            expected_shape = (k.shape[0], k.shape[1])
            if normalized_cache_position.shape != expected_shape:
                raise ValueError(
                    'cache_position must have shape [batch, sequence], got '
                    f'{normalized_cache_position.shape}; expected {expected_shape}'
                )

            def update_cache_row(
                cache_row: jax.Array,
                update_row: jax.Array,
                row_positions: jax.Array,
            ) -> jax.Array:
                return cache_row.at[row_positions].set(update_row)

            k_cache = jax.vmap(update_cache_row)(
                k_cache,
                k,
                normalized_cache_position,
            )
            v_cache = jax.vmap(update_cache_row)(
                v_cache,
                v,
                normalized_cache_position,
            )
            k = k_cache
            v = v_cache

        # Sliding Window / Causal Masking
        if is_causal or self.window_size is not None:
            if self.window_size is not None:
                q_len = q.shape[1]
                k_len = k.shape[1]

                if position_ids is not None:
                    query_positions = jnp.asarray(position_ids, dtype=jnp.int32)
                elif normalized_cache_position is not None:
                    query_positions = normalized_cache_position
                else:
                    query_positions = jnp.broadcast_to(
                        jnp.arange(q_len, dtype=jnp.int32)[None, :],
                        (q.shape[0], q_len),
                    )
                if query_positions.shape != (q.shape[0], q_len):
                    raise ValueError(
                        'position_ids must have shape [batch, sequence], got '
                        f'{query_positions.shape}'
                    )

                # Cached keys use absolute positions from the start of the
                # sequence. Uncached keys belong to the same local chunk as Q.
                if kv_cache is not None:
                    key_positions = jnp.arange(k_len, dtype=jnp.int32)
                else:
                    key_positions = query_positions

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

                if isinstance(use_sliding_window, bool):
                    if use_sliding_window:
                        effective_mask = sliding_mask
                    elif is_causal:
                        effective_mask = causal_mask
                    else:
                        effective_mask = jnp.ones_like(causal_mask)
                else:
                    fallback_mask = (
                        causal_mask
                        if is_causal
                        else jnp.ones_like(causal_mask)
                    )
                    effective_mask = jnp.where(
                        jnp.asarray(use_sliding_window, dtype=jnp.bool_),
                        sliding_mask,
                        fallback_mask,
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask & effective_mask # pyright: ignore[reportOperatorIssue]
                else:
                    attention_mask = effective_mask

                # The absolute-position mask handles causality itself.
                is_causal = False

        kernel_name = kernel.lower() if isinstance(kernel, str) else kernel
        is_boundary_flash = kernel_name in {'flash', 'flash_attention'}
        if (
            kv_cache is not None
            and is_causal
            and not is_boundary_flash
            and self.window_size is None
        ):
            key_positions = jnp.arange(k.shape[1], dtype=jnp.int32)
            causal_mask = (
                key_positions[None, None, None, :]
                <= normalized_cache_position[:, None, :, None] # pyright: ignore[reportOptionalSubscript]
            )
            if attention_mask is None:
                attention_mask = causal_mask
            elif callable(attention_mask):
                raise ValueError(
                    'callable attention masks cannot be combined with cached '
                    'JAX causal attention'
                )
            else:
                attention_mask = (
                    jnp.asarray(attention_mask, dtype=jnp.bool_) & causal_mask
                )
            is_causal = False

        attention_bias = None
        if self.softcap is not None and not is_boundary_flash:
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

        # Apply either JAX dot-product or boundary FlashAttention.
        query_offset: int | jax.Array = 0
        if normalized_cache_position is not None:
            query_offset = normalized_cache_position[:, 0]

        kernel_kwargs: dict[str, tp.Any] = {}
        if is_boundary_flash and self.softcap is not None:
            kernel_kwargs['cap'] = self.softcap

        out = self.apply(
            query=q,
            key=k,
            value=v,
            kernel=kernel,
            bias=attention_bias,
            mask=attention_mask,
            scale=self.scaling,
            is_causal=is_causal,
            boundary_ids=boundary_ids,
            query_offset=query_offset,
            **kernel_kwargs,
        )

        # Output projection from (Batch, SeqLen, Heads, HeadDim) directly to (Batch, SeqLen, HiddenSize)
        out = self.o_proj(out, out_sharding=out_sharding)

        if kv_cache is not None:
            return out, (k_cache, v_cache) # pyright: ignore[reportPossiblyUnboundVariable]

        return out, None # pyright: ignore[reportReturnType]


class TransformerEncoderLayer(nn.Module):
    """A self-attention and feed-forward transformer encoder block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        dropout: float,
        activation: Activation,
        norm_first: bool,
        norm_eps: float,
        bias: bool,
        dtype: DType,
        rngs: nn.Rngs,
        quant: QuantConfig,
        dot_general: Any,
    ) -> None:
        head_dim = hidden_size // num_heads
        self.self_attention = Attention(
            hidden_size,
            num_heads,
            head_dim,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=AxisName.set_axis_names(
                q_proj=('embed', 'heads', 'head_dim'),
                k_proj=('embed', 'heads', 'head_dim'),
                v_proj=('embed', 'heads', 'head_dim'),
                o_proj=('heads', 'head_dim', 'embed'),
            ),
            quant=quant,
            dot_general=dot_general,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            activation=activation,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=AxisName.set_axis_names(
                input=('embed', 'mlp'),
                output=('mlp', 'embed'),
            ),
            quant=quant,
            dot_general=dot_general,
        )
        self.norm1 = nn.LayerNorm(
            hidden_size,
            eps=norm_eps,
            dtype=dtype,
            bias=bias,
            axis_names=('embed',),
        )
        self.norm2 = nn.LayerNorm(
            hidden_size,
            eps=norm_eps,
            dtype=dtype,
            bias=bias,
            axis_names=('embed',),
        )
        self.attention_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
        )
        self.output_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
        )
        self.norm_first = norm_first
        self.dropout = dropout

    def __call__(
        self,
        x: jax.Array,
        *,
        mask: jax.Array | None,
        is_causal: bool,
        out_sharding: jax.sharding.Sharding | None,
    ) -> jax.Array:
        mask = _apply_causal_mask(
            mask,
            is_causal,
            query_length=x.shape[1],
            key_length=x.shape[1],
        )

        attention_input = self.norm1(x) if self.norm_first else x
        attention = _attention_output(
            self.self_attention(
                attention_input,
                attention_mask=mask,
                is_causal=False,
                out_sharding=out_sharding,
            )
        )
        x = x + self.attention_dropout(
            attention,
        )
        if not self.norm_first:
            x = self.norm1(x, out_sharding=out_sharding)

        feed_input = self.norm2(x) if self.norm_first else x
        feed = self.feed_forward(
            feed_input,
            out_sharding=out_sharding,
        )
        x = x + self.output_dropout(
            feed,
            out_sharding=out_sharding,
        )
        if not self.norm_first:
            x = self.norm2(x, out_sharding=out_sharding)
        return x

class TransformerDecoderLayer(nn.Module):
    """A transformer decoder block with optional cross-attention memory."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        dropout: float,
        activation: Activation,
        norm_first: bool,
        norm_eps: float,
        bias: bool,
        dtype: DType,
        rngs: nn.Rngs,
        quant: QuantConfig,
        dot_general: Any,
    ) -> None:
        head_dim = hidden_size // num_heads
        attention_options = {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'axis_names': AxisName.set_axis_names(
                q_proj=('embed', 'heads', 'head_dim'),
                k_proj=('embed', 'heads', 'head_dim'),
                v_proj=('embed', 'heads', 'head_dim'),
                o_proj=('heads', 'head_dim', 'embed'),
            ),
            'shard_mode': shard_mode,
            'quant': quant,
            'dot_general': dot_general,
        }
        self.self_attention = Attention(**attention_options)
        self.cross_attention = Attention(**attention_options)
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            activation=activation,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=AxisName.set_axis_names(
                input=('embed', 'mlp'),
                output=('mlp', 'embed'),
            ),
            quant=quant,
            dot_general=dot_general,
        )
        norm_options = {
            'eps': norm_eps,
            'dtype': dtype,
            'bias': bias,
            'axis_names': ('embed',),
            'shard_mode': shard_mode,
        }
        self.norm1 = nn.LayerNorm(hidden_size, **norm_options)
        self.norm2 = nn.LayerNorm(hidden_size, **norm_options)
        self.norm3 = nn.LayerNorm(hidden_size, **norm_options)
        self.self_attention_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
        )
        self.cross_attention_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
        )
        self.output_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
        )
        self.norm_first = norm_first
        self.dropout = dropout

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array | None,
        *,
        self_mask: jax.Array | None,
        memory_mask: jax.Array | None,
        self_is_causal: bool,
        memory_is_causal: bool,
        out_sharding: jax.sharding.Sharding | None,
    ) -> jax.Array:
        self_mask = _apply_causal_mask(
            self_mask,
            self_is_causal,
            query_length=x.shape[1],
            key_length=x.shape[1],
        )
        if memory is not None:
            memory_mask = _apply_causal_mask(
                memory_mask,
                memory_is_causal,
                query_length=x.shape[1],
                key_length=memory.shape[1],
            )

        attention_input = self.norm1(x) if self.norm_first else x
        self_attention = _attention_output(
            self.self_attention(
                attention_input,
                attention_mask=self_mask,
                is_causal=False,
                out_sharding=out_sharding,
            )
        )
        x = x + self.self_attention_dropout(
            self_attention,
        )
        if not self.norm_first:
            x = self.norm1(x, out_sharding=out_sharding)

        if memory is not None:
            cross_input = self.norm2(x) if self.norm_first else x
            cross_attention = _attention_output(
                self.cross_attention(
                    cross_input,
                    context=memory,
                    attention_mask=memory_mask,
                    is_causal=False,
                    out_sharding=out_sharding,
                )
            )
            x = x + self.cross_attention_dropout(
                cross_attention,
            )
            if not self.norm_first:
                x = self.norm2(x, out_sharding=out_sharding)

        feed_input = self.norm3(x) if self.norm_first else x
        feed = self.feed_forward(
            feed_input,
            out_sharding=out_sharding,
        )
        x = x + self.output_dropout(
            feed,
            out_sharding=out_sharding,
        )
        if not self.norm_first:
            x = self.norm3(x, out_sharding=out_sharding)
        return x

class TransformerEncoder(nn.Module):
    """Apply encoder layers to batch-first hidden states.

    Each supplied layer must follow ``TransformerEncoderLayer.__call__``.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        norm: nn.Module | None = None,
    ) -> None:
        if not layers:
            raise ValueError('layers must contain at least one encoder layer')
        self.layers = nn.List(layers)
        if norm is not None and not isinstance(norm, nn.Module):
            raise TypeError('norm must be an nn.Module or None')
        self.norm = norm

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(
        self,
        source: jax.Array,
        mask: jax.Array | None = None,
        is_causal: bool = False,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        mask = _apply_causal_mask(
            mask,
            is_causal,
            query_length=source.shape[1],
            key_length=source.shape[1],
        )
        output = source
        for layer in self.layers:
            output = layer(
                output,
                mask=mask,
                is_causal=False,
                out_sharding=out_sharding,
            )
        if self.norm is not None:
            output = self.norm(output)
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        return f'layers={len(self.layers)}'

class TransformerDecoder(nn.Module):
    """Apply decoder layers to batch-first hidden states.

    Each supplied layer must follow ``TransformerDecoderLayer.__call__``.
    Passing ``memory=None`` omits every cross-attention branch.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        norm: nn.Module | None = None,
    ) -> None:
        if not layers:
            raise ValueError('layers must contain at least one decoder layer')
        self.layers = nn.List(layers)
        if norm is not None and not isinstance(norm, nn.Module):
            raise TypeError('norm must be an nn.Module or None')
        self.norm = norm

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(
        self,
        target: jax.Array,
        memory: jax.Array | None = None,
        target_mask: jax.Array | None = None,
        memory_mask: jax.Array | None = None,
        target_is_causal: bool = False,
        memory_is_causal: bool = False,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        target_mask = _apply_causal_mask(
            target_mask,
            target_is_causal,
            query_length=target.shape[1],
            key_length=target.shape[1],
        )
        if memory is not None:
            memory_mask = _apply_causal_mask(
                memory_mask,
                memory_is_causal,
                query_length=target.shape[1],
                key_length=memory.shape[1],
            )
        output = target
        for layer in self.layers:
            output = layer(
                output,
                memory,
                self_mask=target_mask,
                memory_mask=memory_mask,
                self_is_causal=False,
                memory_is_causal=False,
                out_sharding=out_sharding,
            )
        if self.norm is not None:
            output = self.norm(output)
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        return f'layers={len(self.layers)}'

class Transformer(nn.Module):
    """
    Apply a conventional encoder-decoder transformer.

    Inputs contain precomputed embeddings; positional information must be
    added by the caller. Attention masks use JAX semantics where ``True``
    permits attention. Key-padding masks use the common inverse convention
    where ``True`` marks a token that must be ignored. Passing ``target=None``
    runs only the encoder; passing ``source=None`` runs only the decoder.
    Custom components must follow the public ``TransformerEncoder`` and
    ``TransformerDecoder`` call contracts.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        activation: Activation = 'relu',
        *,
        norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        custom_encoder: nn.Module | None = None,
        custom_decoder: nn.Module | None = None,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        quant: QuantConfig = None,
        dot_general: Any = None,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.num_heads = _validate_integer(num_heads, 'num_heads')
        self.num_encoder_layers = _validate_integer(
            num_encoder_layers,
            'num_encoder_layers',
        )
        self.num_decoder_layers = _validate_integer(
            num_decoder_layers,
            'num_decoder_layers',
        )
        self.intermediate_size = _validate_integer(
            intermediate_size,
            'intermediate_size',
        )
        if hidden_size % num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        if not isinstance(batch_first, bool):
            raise TypeError('batch_first must be a boolean')
        if not isinstance(norm_first, bool):
            raise TypeError('norm_first must be a boolean')
        if custom_encoder is not None and not isinstance(custom_encoder, nn.Module):
            raise TypeError('custom_encoder must be an nn.Module or None')
        if custom_decoder is not None and not isinstance(custom_decoder, nn.Module):
            raise TypeError('custom_decoder must be an nn.Module or None')
        # Validate the activation before creating any parameters.
        activation_function = _resolve_activation(activation)
        self.activation_name = getattr(
            activation_function,
            '__name__',
            type(activation_function).__name__,
        )
        self.dropout = _validate_probability(dropout, 'dropout')
        self.batch_first = batch_first
        self.norm_first = norm_first

        layer_options = {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'intermediate_size': intermediate_size,
            'dropout': self.dropout,
            'activation': activation,
            'norm_first': norm_first,
            'norm_eps': norm_eps,
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'shard_mode': shard_mode,
            'quant': quant,
            'dot_general': dot_general,
        }
        if custom_encoder is None:
            encoder_norm = nn.LayerNorm(
                hidden_size,
                eps=norm_eps,
                dtype=dtype,
                bias=bias,
                axis_names=('embed',),
            )
            self.encoder = TransformerEncoder(
                [
                    TransformerEncoderLayer(**layer_options)
                    for _ in range(num_encoder_layers)
                ],
                encoder_norm,
            )
        else:
            self.encoder = custom_encoder

        if custom_decoder is None:
            decoder_norm = nn.LayerNorm(
                hidden_size,
                eps=norm_eps,
                dtype=dtype,
                bias=bias,
                axis_names=('embed',),
            )
            self.decoder = TransformerDecoder(
                [
                    TransformerDecoderLayer(**layer_options)
                    for _ in range(num_decoder_layers)
                ],
                decoder_norm,
            )
        else:
            self.decoder = custom_decoder

    def __call__(
        self,
        source: jax.Array | None,
        target: jax.Array | None = None,
        source_mask: jax.Array | None = None,
        target_mask: jax.Array | None = None,
        memory_mask: jax.Array | None = None,
        source_key_padding_mask: jax.Array | None = None,
        target_key_padding_mask: jax.Array | None = None,
        memory_key_padding_mask: jax.Array | None = None,
        source_is_causal: bool = False,
        target_is_causal: bool = False,
        memory_is_causal: bool = False,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        if source is None and target is None:
            raise ValueError('source and target cannot both be None')

        source_unbatched = None
        if source is not None:
            source, source_unbatched = _normalize_input(
                source,
                name='source',
                hidden_size=self.hidden_size,
                batch_first=self.batch_first,
            )
        elif source_mask is not None or source_key_padding_mask is not None:
            raise ValueError('source masks require a source input')

        target_unbatched = None
        if target is not None:
            target, target_unbatched = _normalize_input(
                target,
                name='target',
                hidden_size=self.hidden_size,
                batch_first=self.batch_first,
            )
        elif any(
            value is not None
            for value in (
                target_mask,
                memory_mask,
                target_key_padding_mask,
                memory_key_padding_mask,
            )
        ):
            raise ValueError('target and memory masks require a target input')

        if source is not None and target is not None:
            if source_unbatched != target_unbatched:
                raise ValueError(
                    'source and target must both be batched or unbatched'
                )
            if source.shape[0] != target.shape[0]:
                raise ValueError('source and target batch sizes must match')
        memory = None
        if source is not None:
            batch_size, source_length = source.shape[:2]
            source_attention_mask = _merge_masks(
                source_mask,
                source_key_padding_mask,
                batch_size=batch_size,
                query_length=source_length,
                key_length=source_length,
                attention_name='source_mask',
                padding_name='source_key_padding_mask',
            )
            source_attention_mask = _apply_causal_mask(
                source_attention_mask,
                source_is_causal,
                query_length=source_length,
                key_length=source_length,
            )
            memory = self.encoder(
                source,
                mask=source_attention_mask,
                is_causal=False,
                out_sharding=None,
            )

        if target is None:
            output = memory
            unbatched = source_unbatched
        else:
            batch_size, target_length = target.shape[:2]
            target_attention_mask = _merge_masks(
                target_mask,
                target_key_padding_mask,
                batch_size=batch_size,
                query_length=target_length,
                key_length=target_length,
                attention_name='target_mask',
                padding_name='target_key_padding_mask',
            )
            target_attention_mask = _apply_causal_mask(
                target_attention_mask,
                target_is_causal,
                query_length=target_length,
                key_length=target_length,
            )

            memory_attention_mask = None
            if memory is not None:
                source_length = memory.shape[1]
                memory_attention_mask = _merge_masks(
                    memory_mask,
                    memory_key_padding_mask,
                    batch_size=batch_size,
                    query_length=target_length,
                    key_length=source_length,
                    attention_name='memory_mask',
                    padding_name='memory_key_padding_mask',
                )
                memory_attention_mask = _apply_causal_mask(
                    memory_attention_mask,
                    memory_is_causal,
                    query_length=target_length,
                    key_length=source_length,
                )
            elif memory_mask is not None or memory_key_padding_mask is not None:
                raise ValueError('memory masks require a source input')

            output = self.decoder(
                target,
                memory,
                target_mask=target_attention_mask,
                memory_mask=memory_attention_mask,
                target_is_causal=False,
                memory_is_causal=False,
                out_sharding=None,
            )
            unbatched = target_unbatched

        if unbatched:
            output = output[0]
        elif not self.batch_first:
            output = jnp.swapaxes(output, 0, 1)
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        return (
            f'hidden_size={self.hidden_size}, heads={self.num_heads}, '
            f'encoder_layers={self.num_encoder_layers}, '
            f'decoder_layers={self.num_decoder_layers}'
        )


__all__ = [
    'Transformer',
    'TransformerDecoder',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerEncoderLayer',
]
