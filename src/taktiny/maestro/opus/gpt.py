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
"""GPT architectures"""

from __future__ import annotations

from taktiny.maestro._livret import repertoire
from taktiny.cosettes.common import TransformerCausalLM
from taktiny.cosettes.transformers.llama import LlamaDecoderLayer
from taktiny import nn


# TODO: GPTOSS
class GPTOSS(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

repertoire.register('GptOssForCausalLM', GPTOSS)
__all__ = [
    'GPTOSS'
]
