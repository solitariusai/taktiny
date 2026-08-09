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
"""A Deep Learning framework built on JAX"""

__author__ = "Shinapri"
__version__ = "0.0.1"
__description__ = (
    "A Deep Learning framework built on JAX, featuring OOP-style modeling, "
    "full-lifecycle trainers, and native architectures spanning Transformers, Diffusion, and SSMs."
)

from taktiny.maestro._prelude import Maestro
from taktiny.maestro.config import ModelConfig
from taktiny.takt import Takt
from taktiny import nn, peft, kernels, layers
from taktiny import transforms as tt
from taktiny.trainer import (
    DatasetConfig,
    TensorBoardCallback,
    Trainer,
    TrainerCallback,
    TrainingConfig,
    WandbCallback,
)

from taktiny.maestro.opus import *

__all__ = [
    'Maestro',
    'Takt',
    'PeftConfig',
    'LoraConfig',
    'ModelConfig',
    'Trainer',
    'TrainerCallback',
    'TensorBoardCallback',
    'WandbCallback',
    'TrainingConfig',
    'DatasetConfig',
    'tt',
    'nn',
    'layers',
    'kernels',
    'peft'
]
