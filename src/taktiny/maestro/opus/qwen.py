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

from taktiny.maestro.livret import repertoire
from taktiny.maestro.config import ModelConfig
from taktiny.cosettes.transformers.qwen import (
    Qwen2DecoderLayer,
    Qwen3DecoderLayer,
)
from taktiny.cosettes.transformers.ordinario import (
    TransformerCausalLM,
    TransformerModel,
)


# ┏━┓╻ ╻┏━╸┏┓╻   ┏━┓
# ┃┓┃┃╻┃┣╸ ┃┗┫   ┏━┛
# ┗┻┛┗┻┛┗━╸╹ ╹   ┗━╸
class Qwen2Model(TransformerModel):
    _layer_type = Qwen2DecoderLayer

    def __init__(self, config, *, rngs, **kwargs):
        if config.layer_types is None and config.use_sliding_window:
            max_window_layers = config.max_window_layers or 0
            config.layer_types = [
                (
                    'full_attention'
                    if layer_idx < max_window_layers
                    else 'sliding_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        super().__init__(config, rngs=rngs, **kwargs)


class Qwen2(TransformerCausalLM):
    _model_type = Qwen2Model
    _default_config = ModelConfig(
        vocab_size=151_936,
        hidden_size=4096,
        intermediate_size=22_016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=None,
        hidden_act='silu',
        max_position_embeddings=32_768,
        rope_theta=10_000.0,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_bias=True,
        attention_dropout=0.0,
        mlp_bias=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )

# ┏━┓╻ ╻┏━╸┏┓╻   ┏━┓
# ┃┓┃┃╻┃┣╸ ┃┗┫   ╺━┫
# ┗┻┛┗┻┛┗━╸╹ ╹   ┗━┛
class Qwen3Model(TransformerModel):
    _layer_type = Qwen3DecoderLayer


class Qwen3(TransformerCausalLM):
    _model_type = Qwen3Model
    _default_config = ModelConfig(
        vocab_size=151_936,
        hidden_size=4096,
        intermediate_size=22_016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act='silu',
        max_position_embeddings=32_768,
        rope_theta=10_000.0,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        mlp_bias=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )


repertoire.register('Qwen2ForCausalLM', Qwen2)
repertoire.register('Qwen3ForCausalLM', Qwen3)

__all__ = ['Qwen2Model', 'Qwen2', 'Qwen3Model', 'Qwen3']
