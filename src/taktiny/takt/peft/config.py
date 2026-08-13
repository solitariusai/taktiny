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
"""PEFT configuration types."""

from __future__ import annotations

from dataclasses import dataclass

from taktiny.nn.rng import Rngs


class PeftConfig:
    """Base type for configurations accepted by ``Takt.apply_peft``."""


@dataclass(frozen=True)
class LoraConfig(PeftConfig):
    """Configuration for applying low-rank adapters to matching modules."""

    target_modules: str | tuple[str, ...] | list[str]
    rank: int = 8
    alpha: float = 8.0
    rngs: Rngs | None = None

    def __post_init__(self) -> None:
        targets = (
            (self.target_modules,)
            if isinstance(self.target_modules, str)
            else tuple(self.target_modules)
        )
        if not targets:
            raise ValueError('target_modules must contain at least one pattern')
        if not all(isinstance(target, str) and target for target in targets):
            raise TypeError('target_modules must contain non-empty strings')
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError('rank must be a positive integer')
        if self.alpha <= 0:
            raise ValueError('alpha must be positive')
        object.__setattr__(self, 'target_modules', targets)


__all__ = ['PeftConfig', 'LoraConfig']
