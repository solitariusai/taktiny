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
"""Allegro transformer architecture components."""

from __future__ import annotations

import typing as tp

import jax

from taktiny import layers as ly
from taktiny import nn
from taktiny.cosettes._continuo import _config_value
from taktiny.cosettes.transformers._ordinario import (
    ConditionalTransformerLayer,
)
from taktiny.maestro.config import ModelConfig


class AllegroTransformerLayer(ConditionalTransformerLayer):
    """Allegro's AdaLN self/cross-attention transformer block."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        attention_bias = bool(
            _config_value(config, 'attention_bias', default=False)
        )
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            ffn_dropout=_config_value(config, 'dropout', default=0.0),
            attention_bias=attention_bias,
            cross_attention_bias=attention_bias,
            input_layernorm=nn.LayerNorm,
            self_attention=ly.Attention,
            cross_attention=ly.Attention,
            post_attention_layernorm=nn.LayerNorm,
            mlp=ly.FeedForward,
        )

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array | None,
        temb: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        encoder_attention_mask: jax.Array | None = None,
        position_idx: jax.Array | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
        attention_kernel: str = 'dot_product',
        cross_attention_kernel: str = 'dot_product',
        **kwargs: tp.Any,
    ) -> jax.Array:
        if kwargs:
            names = ', '.join(sorted(kwargs))
            raise TypeError(f'unexpected Allegro attention arguments: {names}')
        return super().__call__(
            x,
            enc_x,
            temb,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            position_idx=position_idx,
            out_sharding=out_sharding,
            self_attention_kernel=attention_kernel,
            cross_attention_kernel=cross_attention_kernel,
        )


__all__ = ['AllegroTransformerLayer']
