# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Chroma image-transformer layers."""

from __future__ import annotations

import typing as tp

import jax

from taktiny.cosettes import layers as ly
from taktiny import nn
from taktiny.cosettes.continuo import (
    _approximate_gelu,
    combine_joint_positions,
    flatten_modulation,
    image_transformer_dimensions,
    multi_axis_position_embedding,
    pairwise_attention_mask,
)
from taktiny.cosettes.transformers.ordinario import (
    GatedParallelTransformerLayer,
    JointTransformerLayer,
)
from taktiny.maestro.config import ModelConfig


class ChromaTransformerLayer(JointTransformerLayer):
    """Chroma dual-stream transformer layer with precomputed modulation."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        hidden_size, _, _, _ = image_transformer_dimensions(config)
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            conditioning_size=hidden_size * 6,
            context_pre_only=False,
            dual_attention=False,
            project_conditioning=False,
            context_project_conditioning=False,
            use_qkv_norm=True,
            qkv_norm_eps=1e-6,
            context_first=True,
            bias=True,
            pos_emb=multi_axis_position_embedding(config),
            activation=_approximate_gelu,
            input_layernorm=ly.AdaXNorm,
            context_input_layernorm=ly.AdaXNorm,
            joint_attention=ly.JointAttention,
            post_attention_layernorm=nn.LayerNorm,
            context_post_attention_layernorm=nn.LayerNorm,
            mlp=ly.FeedForward,
            context_mlp=ly.FeedForward,
        )

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array,
        modulation: jax.Array,
        *,
        position_idx: jax.Array | None = None,
        encoder_position_idx: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        **attention_kwargs: tp.Any,
    ) -> tuple[jax.Array, jax.Array]:
        modulation = flatten_modulation(
            modulation,
            batch_size=x.shape[0],
            groups=12,
            hidden_size=self.hidden_size,
        )
        image_modulation = modulation[:, :6 * self.hidden_size]
        text_modulation = modulation[:, 6 * self.hidden_size:]
        joint_positions = combine_joint_positions(
            encoder_position_idx,
            position_idx,
            batch_size=x.shape[0],
        )
        enc_x, x = super().__call__(
            x,
            enc_x,
            image_modulation,
            context_conditioning=text_modulation,
            position_idx=joint_positions,
            attention_mask=pairwise_attention_mask(attention_mask),
            **attention_kwargs,
        )
        return enc_x, x


class ChromaSingleTransformerLayer(GatedParallelTransformerLayer):
    """Chroma single-stream gated parallel attention-and-MLP layer."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        hidden_size, _, _, _ = image_transformer_dimensions(config)
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            conditioning_size=hidden_size * 3,
            project_conditioning=False,
            pos_emb=multi_axis_position_embedding(config),
            activation=_approximate_gelu,
            use_qkv_norm=True,
            bias=True,
            input_layernorm=ly.AdaXNorm,
            parallel_path=ly.ParallelAttentionMLP,
        )

    def __call__(
        self,
        x: jax.Array,
        modulation: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        **attention_kwargs: tp.Any,
    ) -> jax.Array:
        modulation = flatten_modulation(
            modulation,
            batch_size=x.shape[0],
            groups=3,
            hidden_size=self.hidden_size,
        )
        return super().__call__(
            x,
            None,
            modulation,
            attention_mask=pairwise_attention_mask(attention_mask),
            **attention_kwargs,
        )


__all__ = [
    'ChromaSingleTransformerLayer',
    'ChromaTransformerLayer',
]
