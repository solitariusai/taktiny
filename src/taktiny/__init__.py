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

__author__ = "Shinapri"
__version__ = "0.0.1"
__description__ = (
    "A Deep Learning library built on JAX, featuring OOP-style modeling, data pre-processing and trainers."
)

from taktiny import data as data
from taktiny import nn as nn
from taktiny import takt as takt
from taktiny import trainer as trainer
from taktiny import utils as utils
from taktiny.takt import Takt as Takt
from taktiny.utils.transforms import scan, vmap

__all__ = [
    'data',
    'nn',
    'takt',
    'trainer',
    'utils',
    'Takt',
    'vmap',
    'scan',
]
