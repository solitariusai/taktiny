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
"""Built-in AdaLoRA adapter."""

from __future__ import annotations

from typing import Any

from taktiny.nn.modules.peft import AdaLoRALinear
from taktiny.nn.rng import Rngs
from taktiny.takt.peft.adapter import BaseAdapter
from taktiny.utils.typing import DType


class AdaLoRAAdapter(BaseAdapter):
    """Apply :class:`AdaLoRALinear` to matching modules."""

    _adapter = AdaLoRALinear

    def __init__(
        self,
        targets: str | list[str] | tuple[str, ...],
        rank: int = 8,
        alpha: float = 16.0,
        *,
        dtype: DType | None = None,
        rngs: Rngs,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            targets,
            rank,
            alpha,
            dtype=dtype,
            rngs=rngs,
            **kwargs,
        )


__all__ = ['AdaLoRAAdapter']
