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
from typing import Any
import typing as tp
import jax.numpy as jnp, jax

from taktiny.maestro._livret import repertoire
from taktiny.cosettes.common import (
    TransformerCausalLM,
    TransformerConditionalGeneration,
    DiffusionLM,
    TransformerConditionalGeneration,
    TransformerContext
)
from taktiny.cosettes.transformers.gemma import (
    GemmaTextScaledWordEmbedding,
    GemmaRMSNorm,
    GemmaDecoderLayer,
    Gemma2DecoderLayer,
    Gemma3TextScaledWordEmbedding,
    Gemma3RMSNorm,
    Gemma3DecoderLayer,
)
from taktiny import nn
from taktiny.maestro.config import ModelConfig
from taktiny.utils.typing import PathLike, LogicalRules


class Gemma(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        config.tie_word_embeddings = True
        super().__init__(
            config,
            embedding=GemmaTextScaledWordEmbedding,
            decoder=GemmaDecoderLayer,
            norm=GemmaRMSNorm,
            **kwargs
        )

    @classmethod
    def from_pretrained(
        cls, path_or_repo: Any,
        mesh: Any=None,
        sharding_rules: Any=None,
        local: bool=False,
        **kwargs: Any
    ) -> Any:
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        config.tie_word_embeddings = True
        return super().from_pretrained(
            path_or_repo,
            mesh=mesh,
            sharding_rules=sharding_rules,
            local=local,
            config=config,
            **kwargs,
        )

class Gemma2(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs: Any) -> None:
        config.tie_word_embeddings = True
        if getattr(config, 'layer_types', None) is None:
            config.layer_types = [
                (
                    'sliding_attention'
                    if (layer_idx + 1) % 2
                    else 'full_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        super().__init__(
            config,
            embedding=GemmaTextScaledWordEmbedding,
            decoder=Gemma2DecoderLayer,
            norm=GemmaRMSNorm,
            **kwargs
        )
        self.final_logit_softcapping = config.final_logit_softcapping

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        ctx: TransformerContext | None = None,
        logits_to_keep: int = 0,
    ) -> tuple[Any, ...]:
        logits, ctx = super().__call__(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ctx=ctx,
            logits_to_keep=logits_to_keep,
        )
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * jnp.tanh(logits / cap)
        return logits, ctx


class Gemma3(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        if bool(getattr(config, 'use_bidirectional_attention', False)):
            raise NotImplementedError(
                'Gemma3 bidirectional attention is not supported'
            )

        config.tie_word_embeddings      = True
        config.head_dim                 = config.head_dim or config.hidden_size // config.num_attention_heads
        config.num_key_value_heads      = config.num_key_value_heads or config.num_attention_heads
        config.rope_theta               = config.rope_theta or 1_000_000.0
        config.rope_local_base_freq     = config.rope_local_base_freq or 10_000.0
        config.query_pre_attn_scalar    = config.query_pre_attn_scalar or 256
        config.attention_bias           = config.attention_bias or False
        config.rms_norm_eps             = config.rms_norm_eps or 1e-6

        if config.layer_types is None:
            pattern = config.sliding_window_pattern or 6
            config.layer_types = [
                (
                    'sliding_attention'
                    if (layer_idx + 1) % pattern
                    else 'full_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]

        super().__init__(
            config,
            embedding=Gemma3TextScaledWordEmbedding,
            decoder=Gemma3DecoderLayer,
            norm=Gemma3RMSNorm,
            **kwargs
        )
        self.final_logit_softcapping = config.final_logit_softcapping

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        ctx: TransformerContext | None = None,
        logits_to_keep: int = 0,
    ) -> tuple[Any, ...]:
        logits, ctx = super().__call__(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ctx=ctx,
            logits_to_keep=logits_to_keep,
        )
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * jnp.tanh(logits / cap)
        return logits, ctx

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        local: bool = False,
        **kwargs: tp.Any,
    ) -> tp.Self:
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        config.tie_word_embeddings = True
        return super().from_pretrained(
            path_or_repo,
            mesh=mesh,
            sharding_rules=sharding_rules,
            local=local,
            config=config,
            **kwargs,
        )

class Gemma3ConditionalGeneration(TransformerConditionalGeneration):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=Gemma3DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

class Gemma4(TransformerConditionalGeneration):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=Gemma3DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

class Gemma4Unified(TransformerConditionalGeneration):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=Gemma3DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

class DiffusionGemma(DiffusionLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')

class_map = [
    ('GemmaForCausalLM', Gemma),
    ('Gemma2ForCausalLM', Gemma2),
    ('Gemma3ForCausalLM', Gemma3),
    ('Gemma3ForConditionalGeneration', Gemma3ConditionalGeneration),
    ('Gemma4ForConditionalGeneration', Gemma4),
    ('Gemma4UnifiedForConditionalGeneration', Gemma4Unified),
    ('DiffusionGemmaForBlockDiffusion', DiffusionGemma),
]

__all__ = []
for name, cls in class_map:
    repertoire.register(name, cls)
    __all__.append(cls.__name__)
