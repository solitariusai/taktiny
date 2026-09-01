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
"""Built-in VeRA adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

from taktiny.nn.base import Module, Parameter
from taktiny.nn.modules.linear import default_linear_initializer
from taktiny.nn.modules.peft import VeRALinear
from taktiny.nn.rng import Rngs
from taktiny.takt.peft.adapter import BaseAdapter
from taktiny.utils.typing import DType, Initializer


class VeRAAdapter(BaseAdapter):
    """Apply VeRA using frozen random projections shared by all targets."""

    _adapter = VeRALinear

    def __init__(
        self,
        targets: str | list[str] | tuple[str, ...],
        rank: int = 8,
        *,
        dtype: DType | None = None,
        d_initial: float = 0.1,
        rngs: Rngs,
        initializer: Initializer = default_linear_initializer,
        vera_A: Parameter | None = None,
        vera_B: Parameter | None = None,
        **kwargs: Any,
    ) -> None:
        if rngs is None:
            raise ValueError('rngs is required to initialize VeRA projections')
        if (vera_A is None) != (vera_B is None):
            raise ValueError('vera_A and vera_B must be provided together')

        self._rngs = rngs
        self._initializer = initializer
        self._projection_dtype = jnp.float32 if dtype is None else dtype
        self.vera_A = vera_A
        self.vera_B = vera_B
        if vera_A is not None:
            vera_A.trainable = False
            vera_B.trainable = False
        super().__init__(
            targets,
            rank,
            dtype=dtype,
            d_initial=d_initial,
            **kwargs,
        )
        if vera_A is not None:
            self._adapter_kwargs['vera_A'] = vera_A
            self._adapter_kwargs['vera_B'] = vera_B

    def prepare(self, targets: Sequence[tuple[str, Module]]) -> None:
        """Create projections large enough for every selected linear layer."""
        if self.vera_A is not None:
            return

        max_input = 0
        max_output = 0
        for module_path, module in targets:
            if not (
                hasattr(module, 'in_features')
                and hasattr(module, 'out_features')
            ):
                raise TypeError(
                    f'VeRA target {module_path} is not a linear module'
                )
            max_input = max(max_input, math.prod(module.in_features))
            max_output = max(max_output, math.prod(module.out_features))

        self.vera_A = Parameter(
            self._initializer(
                self._rngs(),
                (max_input, self._adapter_args[0]),
                self._projection_dtype,
            ),
            trainable=False,
        )
        self.vera_B = Parameter(
            self._initializer(
                self._rngs(),
                (self._adapter_args[0], max_output),
                self._projection_dtype,
            ),
            trainable=False,
        )
        self._adapter_kwargs['vera_A'] = self.vera_A
        self._adapter_kwargs['vera_B'] = self.vera_B


__all__ = ['VeRAAdapter']
