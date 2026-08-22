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
"""Attention kernels from MaxText (Flash, Ragged, Splash, Ring)."""

from taktiny.cosettes.kernels.attention.flash_attention import flash_attention, flash_attention_block_masked
from taktiny.cosettes.kernels.attention.ragged_attention import (
    ragged_flash_attention_kernel,
    ragged_mqa,
    ragged_mha,
    ragged_gqa,
)
from taktiny.cosettes.kernels.attention.splash_attention import (
    make_attention_reference,
    attention_reference,
    attention_reference_custom,
)
from taktiny.cosettes.kernels.attention.ring_attention import (
    is_context_parallel_ring_requested,
    build_splash_config,
)

__all__ = [
    "flash_attention",
    "flash_attention_block_masked",
    "ragged_flash_attention_kernel",
    "ragged_mqa",
    "ragged_mha",
    "ragged_gqa",
    "make_attention_reference",
    "attention_reference",
    "attention_reference_custom",
    "is_context_parallel_ring_requested",
    "build_splash_config",
]
