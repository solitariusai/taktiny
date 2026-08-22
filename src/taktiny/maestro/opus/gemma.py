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
"""Gemma architectures"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.cosettes.layers import _RotaryEmbedding
from taktiny.maestro.livret import repertoire
from taktiny.cosettes.transformers.ordinario import (
    PositionEmbedding,
    PositionEmbeddings,
    TransformerCausalLM,
    TransformerModel,
)
from taktiny.cosettes.transformers.gemma import (
    Gemma2DecoderLayer,
    Gemma3DecoderLayer,
    GemmaDecoderLayer,
    GemmaRMSNorm,
    GemmaTextScaledWordEmbedding,
)
from taktiny.maestro.config import ModelConfig
from taktiny.utils.typing import LogicalRules, PathLike


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹
class GemmaModel(TransformerModel):
    _layer_type = GemmaDecoderLayer
    _token_embedding = GemmaTextScaledWordEmbedding
    _norm = GemmaRMSNorm

class Gemma(TransformerCausalLM):
    _model_type = GemmaModel
    _default_config = ModelConfig(
        vocab_size=256_000,
        hidden_size=3072,
        intermediate_size=24_576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_bidirectional_attention=None,
    )


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓   ┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫   ┏━┛
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹   ┗━╸
class Gemma2Model(GemmaModel):
    _layer_type = Gemma2DecoderLayer

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        **kwargs: tp.Any,
    ) -> None:
        if config.layer_types is None:
            config.layer_types = [
                (
                    'sliding_attention'
                    if layer_idx % 2 == 0
                    else 'full_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        super().__init__(config, rngs=rngs, **kwargs)

class Gemma2(Gemma):
    _model_type = Gemma2Model
    _default_config = ModelConfig(
        vocab_size=256_000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation='gelu_pytorch_tanh',
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        layer_types=None,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        use_bidirectional_attention=None,
    )
    _default_module_map = [
        *Gemma._default_module_map,
        ('pre_feedforward_layernorm', 'norm3'),
        ('post_feedforward_layernorm', 'norm4'),
    ]

    def _process_logits(self, logits: jax.Array) -> jax.Array:
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = jnp.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓   ┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫   ╺━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹   ┗━┛
class Gemma3TextModel(Gemma2Model):
    _layer_type = Gemma3DecoderLayer

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        **kwargs: tp.Any,
    ) -> None:
        pattern = config.sliding_window_pattern
        if not isinstance(pattern, int) or isinstance(pattern, bool):
            raise TypeError('sliding_window_pattern must be an integer')
        if pattern <= 0:
            raise ValueError('sliding_window_pattern must be positive')

        if config.layer_types is None:
            config.layer_types = [
                (
                    'sliding_attention'
                    if (index + 1) % pattern
                    else 'full_attention'
                )
                for index in range(config.num_hidden_layers)
            ]
        elif len(config.layer_types) != config.num_hidden_layers:
            raise ValueError(
                'config.layer_types must contain one entry per layer'
            )

        default_theta = config.default_theta
        global_theta = (
            config.rope_theta
            or default_theta.get('global')
        )
        local_theta = (
            config.rope_local_base_freq
            or default_theta.local
        )
        rope_parameters = ModelConfig(
            full_attention={
                'rope_type': 'default',
                'rope_theta': global_theta,
            },
            sliding_attention={
                'rope_type': 'default',
                'rope_theta': local_theta,
            },
        )
        if config.rope_parameters is not None:
            rope_parameters = rope_parameters.with_overrides(
                config.rope_parameters
            )
        if config.rope_scaling is not None:
            rope_parameters.full_attention = (
                rope_parameters.full_attention.with_overrides(
                    config.rope_scaling
                )
            )
        config.rope_parameters = rope_parameters

        super().__init__(config, rngs=rngs, **kwargs)
        head_dim = (
            config.head_dim
            or config.hidden_size // config.num_attention_heads
        )
        global_rope = config.rope_parameters.full_attention
        local_rope = config.rope_parameters.sliding_attention
        self.rotary_embedding = _RotaryEmbedding(
            head_dim,
            config.max_position_embeddings,
            base=global_rope.rope_theta,
            rope_scaling=global_rope,
        )
        self.local_rotary_embedding = _RotaryEmbedding(
            head_dim,
            config.max_position_embeddings,
            base=local_rope.rope_theta,
            rope_scaling=local_rope,
        )
        self.sliding_pattern = tuple(
            layer_type == 'sliding_attention'
            for layer_type in config.layer_types
        )

    def _position_embeddings(
        self,
        x: jax.Array,
        position_ids: jax.Array | None,
    ) -> tp.Mapping[str, PositionEmbedding]:
        return {
            'full_attention': self.rotary_embedding(x, position_ids),
            'sliding_attention': self.local_rotary_embedding(x, position_ids),
        }

    def _position_embedding_for_layer(
        self,
        position_embeddings: PositionEmbeddings,
        layer_idx: jax.Array,
    ) -> PositionEmbedding:
        if not isinstance(position_embeddings, tp.Mapping):
            cosine, sine = position_embeddings
            return cosine, sine
        use_sliding = jnp.asarray(
            self.sliding_pattern,
            dtype=jnp.bool_,
        )[layer_idx]
        local = position_embeddings['sliding_attention']
        global_ = position_embeddings['full_attention']
        return (
            jnp.where(use_sliding, local[0], global_[0]),
            jnp.where(use_sliding, local[1], global_[1]),
        )

class Gemma3(Gemma2):
    _model_type = Gemma3TextModel
    _default_module_map = [
        ('model.language_model.', 'model.'),
        *Gemma2._default_module_map,
    ]
    _default_config = ModelConfig(
        vocab_size=262_208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation='gelu_pytorch_tanh',
        max_position_embeddings=131_072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        sliding_window_pattern=6,
        layer_types=None,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        use_bidirectional_attention=False,
        default_theta={'global': 1_000_000.0, 'local': 10_000.0},
    )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        *,
        config: ModelConfig | None = None,
        local: bool = False,
        **kwargs: tp.Any,
    ) -> tp.Self:
        if config is None:
            config = ModelConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )
        text_config = vars(config).get('text_config')
        if text_config is not None:
            config = text_config
        return super().from_pretrained(
            path_or_repo,
            config=config,
            local=local,
            **kwargs,
        )

# ┏━╸┏━╸┏┳┓┏┳┓┏━┓   ╻ ╻
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫   ┗━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹     ╹
# TODO: Gemma 4

class_map = [
    ('GemmaForCausalLM', Gemma),
    ('Gemma2ForCausalLM', Gemma2),
    ('Gemma3ForCausalLM', Gemma3),
    ('Gemma3ForConditionalGeneration', Gemma3),
]

for name, cls in class_map:
    repertoire.register(name, cls)

__all__ = [
    'GemmaModel',
    'Gemma',
    'Gemma2Model',
    'Gemma2',
    'Gemma3TextModel',
    'Gemma3',
]
