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
"""Generic preprocessing for caller-provided data; no automatic downloads.

Use DataLoader with composable operations, or call Map/Compose/MapFields on
individual examples. Text-specific helpers also remain available from .text.
"""

from taktiny.data.loader import DataLoader, RandomAccessSource, train_validation_split
from taktiny.data.transforms import (
    ApplyTemplate,
    Batch,
    BatchMap,
    Compose,
    Filter,
    FlatMap,
    IndexMap,
    Map,
    MapFields,
    Pack,
    RandomMap,
)

__all__ = [
    'ApplyTemplate',
    'Batch',
    'BatchMap',
    'CausalLMBatch',
    'Compose',
    'DataLoader',
    'DatasetUtils',
    'Filter',
    'FlatMap',
    'IndexMap',
    'Map',
    'MapFields',
    'Pack',
    'PackSequences',
    'RandomAccessSource',
    'RandomMap',
    'tokenize',
    'train_validation_split'
]
