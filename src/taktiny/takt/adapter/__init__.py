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
"""Parameter-efficient fine-tuning adapters."""

from taktiny.takt.adapter.adalora import AdaLoRAAdapter
from taktiny.takt.adapter.base import AdapterBase
from taktiny.takt.adapter.base import AdapterBase as BaseAdapter
from taktiny.takt.adapter.dora import DoRAAdapter
from taktiny.takt.adapter.loha import LoHaAdapter
from taktiny.takt.adapter.lokr import LoKrAdapter
from taktiny.takt.adapter.lora import LoRAAdapter
from taktiny.takt.adapter.vera import VeRAAdapter

__all__ = [
    'AdaLoRAAdapter',
    'AdapterBase',
    'BaseAdapter',
    'DoRAAdapter',
    'LoHaAdapter',
    'LoKrAdapter',
    'LoRAAdapter',
    'VeRAAdapter',
]
