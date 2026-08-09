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
from __future__ import annotations

from taktiny.cosettes.common import (
    TransformerContext,
    TransformerCausalLM,
    TransformerConditionalGeneration,
    TransformerDecoderLayer,
    TransformerModel
)

# Llama
from taktiny.cosettes.transformers import LlamaDecoderLayer

# Qwen
from taktiny.cosettes.transformers import QwenDecoderLayer, Qwen2DecoderLayer

# Gemma
from taktiny.cosettes.transformers import (
    GemmaDecoderLayer,
    Gemma2DecoderLayer,
    Gemma3DecoderLayer
)

__all__ = [
    'TransformerContext',
    'TransformerCausalLM',
    'TransformerConditionalGeneration',
    'TransformerDecoderLayer',
    'TransformerModel',
    'LlamaDecoderLayer',
    'QwenDecoderLayer',
    'Qwen2DecoderLayer',
    'GemmaDecoderLayer',
    'Gemma2DecoderLayer',
    'Gemma3DecoderLayer',
]
