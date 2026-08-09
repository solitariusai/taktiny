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
"""Activation modules"""
from __future__ import annotations
from collections.abc import Callable
import jax
from taktiny import nn

class _ActivationBase(nn.Module):
    def __call__(
        self,
        x: jax.Array,
        act_fn: str | Callable[[jax.Array], jax.Array] | None = None,
    ) -> jax.Array:
        if act_fn is None:
            act_fn = self.__class__.__name__.lower()
            if 'hard' in act_fn:
                act_fn = act_fn.replace('hard', 'hard_')

        if isinstance(act_fn, str):
            act_fn = getattr(jax.nn, act_fn)

        return act_fn(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class SiLU(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class GELU(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class ReLU(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class ELU(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class Swish(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class SELU(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class SoftPlus(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class Mish(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class HardSwish(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class Sigmoid(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class SoftSign(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x, jax.nn.soft_sign)

class Tanh(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class HardTanh(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

class HardSigmoid(_ActivationBase):
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)

__all__ = [
    'SiLU',
    'GELU',
    'ReLU',
    'ELU',
    'Swish',
    'SELU',
    'SoftPlus',
    'Mish',
    'HardSwish',
    'Sigmoid',
    'SoftSign',
    'Tanh',
    'HardTanh',
    'HardSigmoid',
]
