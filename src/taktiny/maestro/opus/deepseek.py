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
"""Deepseek architectures"""
from __future__ import annotations
import typing as tp

from taktiny.maestro._livret import repertoire
from taktiny.cosettes.common import TransformerCausalLM
from taktiny.cosettes.transformers.llama import LlamaDecoderLayer
from taktiny.maestro.config import ModelConfig
from taktiny import nn


# TODO: Deepseek
class Deepseek(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

# TODO: DeepseekV2
class DeepseekV2(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

# TODO: DeepseekV3
class DeepseekV3(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

# TODO: DeepseekV3_2
class DeepseekV3_2(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

# TODO: DeepseekV4
class DeepseekV4(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

class_map = [
    ('DeepseekForCausalLM', Deepseek),
    ('DeepseekV2ForCausalLM', DeepseekV2),
    ('DeepseekV3ForCausalLM', DeepseekV3),
    ('DeepseekV32ForCausalLM', DeepseekV3_2),
    ('DeepseekV4ForCausalLM', DeepseekV4),
]

__all__ = []
for name, cls in class_map:
    repertoire.register(name, cls)
    __all__.append(cls.__name__)
