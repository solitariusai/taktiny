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
"""Llama architectures"""

from __future__ import annotations

from taktiny.maestro.livret import repertoire
from taktiny.cosettes.transformers.ordinario import (
    TransformerCausalLM,
    TransformerModel,
)
from taktiny.cosettes.transformers.llama import LlamaDecoderLayer
from taktiny.maestro.config import ModelConfig


# ╻  ╻  ┏━┓┏┳┓┏━┓
# ┃  ┃  ┣━┫┃┃┃┣━┫
# ┗━╸┗━╸╹ ╹╹ ╹╹ ╹
class LlamaModel(TransformerModel):
    _layer_type = LlamaDecoderLayer


class Llama(TransformerCausalLM):
    _model_type = LlamaModel
    _default_config = ModelConfig(
        vocab_size=32_000,
        hidden_size=4096,
        intermediate_size=11_008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act='silu',
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
    )


repertoire.register('LlamaForCausalLM', Llama)

__all__ = ['LlamaModel', 'Llama']
