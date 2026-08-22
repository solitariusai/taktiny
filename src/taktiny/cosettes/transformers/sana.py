"""Sana transformer architecture components."""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from taktiny.cosettes import layers as ly
from taktiny import nn
from taktiny.cosettes.transformers.ordinario import (
    ConditionalTransformerLayer,
)
from taktiny.maestro.config import ModelConfig


class SanaLinearAttention(ly.AttentionLegacy):
    """Sana's ReLU feature-map linear self-attention."""

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array | tuple[jax.Array, jax.Array] | None = None,
        attention_mask: jax.Array | None = None,
        is_causal: bool = False,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
        position_idx: jax.Array | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
        kernel: str = 'linear',
        query: jax.Array | None = None,
        key: jax.Array | None = None,
        value: jax.Array | None = None,
    ) -> tuple[jax.Array, None]:
        del attention_mask, position_idx, kernel
        if context is not None:
            raise ValueError('SanaLinearAttention only supports self-attention')
        if is_causal:
            raise ValueError('SanaLinearAttention is non-causal')
        if kv_cache is not None:
            raise ValueError('SanaLinearAttention does not use a KV cache')

        q = self.q_proj(x) if query is None else query
        k = self.k_proj(x) if key is None else key
        v = self.v_proj(x) if value is None else value
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        input_dtype = q.dtype
        q = jax.nn.relu(q).transpose(0, 2, 3, 1).astype(jnp.float32)
        k = jax.nn.relu(k).transpose(0, 2, 1, 3).astype(jnp.float32)
        v = v.transpose(0, 2, 3, 1).astype(jnp.float32)
        ones = jnp.ones((*v.shape[:2], 1, v.shape[-1]), dtype=v.dtype)
        value_with_normalizer = jnp.concatenate((v, ones), axis=2)
        statistics = jnp.matmul(value_with_normalizer, k)
        output = jnp.matmul(statistics, q)
        output = output[:, :, :-1, :] / (output[:, :, -1:, :] + 1e-15)
        output = output.transpose(0, 3, 1, 2).astype(input_dtype)
        output = self.o_proj(output, out_sharding=out_sharding)
        if input_dtype == jnp.float16:
            output = jnp.clip(output, -65504, 65504)
        return output, None


class SanaTransformerLayer(ConditionalTransformerLayer):
    """Sana block with linear self-attention and a GLUMBConv FFN."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            input_layernorm=nn.LayerNorm,
            self_attention=SanaLinearAttention,
            cross_attention=ly.AttentionLegacy,
            post_attention_layernorm=nn.LayerNorm,
            mlp=ly.GLUMBConv,
        )

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array,
        temb: jax.Array,
        *,
        height: int,
        width: int,
        attention_mask: jax.Array | None = None,
        encoder_attention_mask: jax.Array | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
        cross_attention_kernel: str = 'dot_product',
        **kwargs: tp.Any,
    ) -> jax.Array:
        if kwargs:
            names = ', '.join(sorted(kwargs))
            raise TypeError(f'unexpected Sana attention arguments: {names}')
        return super().__call__(
            x,
            enc_x,
            temb,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            out_sharding=out_sharding,
            self_attention_kernel='linear',
            cross_attention_kernel=cross_attention_kernel,
            ffn_kwargs={'height': height, 'width': width},
        )


__all__ = ['SanaLinearAttention', 'SanaTransformerLayer']
