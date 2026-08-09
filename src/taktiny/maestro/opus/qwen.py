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
"""Qwen architectures"""

from __future__ import annotations
import typing as tp
import jax.numpy as jnp, jax

from taktiny.maestro._livret import repertoire
from taktiny.maestro.config import ModelConfig
from taktiny.transformer import (
    TransformerCausalLM,
    TransformerConditionalGeneration,
)
from taktiny.cosettes.transformers.qwen import (
    QwenDecoderLayer,
    Qwen2DecoderLayer,
    Qwen3DecoderLayer,
)
from taktiny import nn
from taktiny.utils.typing import PathLike, LogicalRules


class Qwen(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        config.num_key_value_heads = config.num_key_value_heads or config.num_attention_heads
        config.head_dim = config.head_dim or config.kv_channels or config.hidden_size // config.num_attention_heads
        config.rope_theta = config.rope_theta or config.rotary_emb_base or 10_000.0
        config.rms_norm_eps = config.rms_norm_eps or config.layer_norm_epsilon or 1e-6
        config.hidden_act = config.hidden_act or 'silu'
        config.attention_bias = False
        config.mlp_bias = not bool(config.no_bias)
        config.seq_length = config.seq_length or config.max_position_embeddings
        super().__init__(
            config,
            decoder=QwenDecoderLayer,
            norm=nn.RMSNorm,
            **kwargs
        )

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

        def split_qkv(value: tp.Any) -> tp.Any:
            return jnp.split(value, 3, axis=0)

        module_map = [
            ('transformer.wte.weight', 'model.embed_tokens.embedding'),
            ('transformer.h.', 'model.layers.'),
            ('transformer.ln_f.', 'model.norm.'),
            (
                '.attn.c_attn.weight',
                [
                    '.attn.q_proj.weight',
                    '.attn.k_proj.weight',
                    '.attn.v_proj.weight',
                ],
                split_qkv,
            ),
            (
                '.attn.c_attn.bias',
                [
                    '.attn.q_proj.bias',
                    '.attn.k_proj.bias',
                    '.attn.v_proj.bias',
                ],
                split_qkv,
            ),
            ('.attn.c_proj.', '.attn.o_proj.'),
        ]

        return cls._load_from_pretrained(
            path_or_repo,
            config,
            module_map,
            local=local,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )

class Qwen2(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=Qwen2DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs
        )

class Qwen3(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=Qwen3DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

# TODO: Qwen3MoE
class Qwen3MoE(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=Qwen2DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

# TODO: Qwen3Next
class Qwen3Next(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=Qwen2DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

# TODO: Qwen3_5MoE
class Qwen3_5MoE(TransformerConditionalGeneration):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=Qwen2DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

class_map = [
    ('QwenForCausalLM', Qwen),
    ('QWenLMHeadModel', Qwen),
    ('Qwen2ForCausalLM', Qwen2),
    ('Qwen3ForCausalLM', Qwen3),
    ('Qwen3MoeForCausalLM', Qwen3MoE),
    ('Qwen3NextForCausalLM', Qwen3Next),
    ('Qwen3_5MoeForConditionalGeneration', Qwen3_5MoE),
]

__all__ = []
for name, cls in class_map:
    repertoire.register(name, cls)
    __all__.append(cls.__name__)
