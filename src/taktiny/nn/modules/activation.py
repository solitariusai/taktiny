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

from taktiny.nn.base import Module


class _ActivationBase(Module):
    """
    Base class for activation functions.
    """
    def __call__(
        self,
        x: jax.Array,
        act_fn: str | Callable[[jax.Array], jax.Array] | None = None,
    ) -> jax.Array:
        """Applies the activation function.

        Args:
            x (jax.Array): The input array.
            act_fn (str | Callable[[jax.Array], jax.Array] | None, optional): The activation function to apply. Defaults to None.

        Returns:
            jax.Array: The activated output array.
        """
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
    """
    SiLU (Swish) activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the SiLU activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class GELU(_ActivationBase):
    """
    GELU activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the GELU activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class ReLU(_ActivationBase):
    """
    ReLU activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the ReLU activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class ELU(_ActivationBase):
    """
    ELU activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the ELU activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class Swish(_ActivationBase):
    """
    Swish activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the Swish activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class SELU(_ActivationBase):
    """
    SELU activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the SELU activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class SoftPlus(_ActivationBase):
    """
    SoftPlus activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the SoftPlus activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class Mish(_ActivationBase):
    """
    Mish activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the Mish activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class HardSwish(_ActivationBase):
    """
    HardSwish activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the HardSwish activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class Sigmoid(_ActivationBase):
    """
    Sigmoid activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the Sigmoid activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class SoftSign(_ActivationBase):
    """
    SoftSign activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the SoftSign activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x, jax.nn.soft_sign)

class Tanh(_ActivationBase):
    """
    Tanh activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the Tanh activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class HardTanh(_ActivationBase):
    """
    HardTanh activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the HardTanh activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)

class HardSigmoid(_ActivationBase):
    """
    HardSigmoid activation function.
    """
    def __call__(self, x: jax.Array) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Applies the HardSigmoid activation function.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The activated output array.
        """
        return super().__call__(x)


__all__ = [
    'ELU',
    'GELU',
    'SELU',
    'HardSigmoid',
    'HardSwish',
    'HardTanh',
    'Mish',
    'ReLU',
    'SiLU',
    'Sigmoid',
    'SoftPlus',
    'SoftSign',
    'Swish',
    'Tanh',
]
