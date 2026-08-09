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
"""LoRA modules"""
from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jnp
from taktiny.nn.module import Module, Parameter
from taktiny.nn.rng import Rngs

class LoRALinear(Module):
    def __init__(
        self,
        base_layer: Module,
        rank: int,
        alpha: float,
        rngs: Rngs,
    ) -> None:
        self.base_layer = base_layer
        self.in_features = getattr(base_layer, 'in_features', None)
        self.out_features = getattr(base_layer, 'out_features', None)
        self.rank = rank
        self.scaling = alpha / rank

        if self.in_features is None or self.out_features is None:
            raise ValueError("Base layer must have in_features and out_features attributes.")

        # Detect if base_layer is stacked (used inside SeqStack)
        sample_param = getattr(base_layer, 'weight', None)
        if sample_param is not None:
            expected_dims = len(self.in_features) + len(self.out_features)
            is_stacked = len(sample_param.shape) > expected_dims
        else:
            is_stacked = False

        num_layers = sample_param.shape[0] if is_stacked else None

        import math
        self.in_features_flat = math.prod(self.in_features)
        self.out_features_flat = math.prod(self.out_features)

        w_key = rngs()
        if is_stacked:
            self.lora_A = Parameter(jax.random.normal(w_key, (num_layers, self.in_features_flat, self.rank), dtype=jnp.float32) * (1 / self.in_features_flat))
            self.lora_B = Parameter(jnp.zeros((num_layers, self.rank, self.out_features_flat), dtype=jnp.float32))
        else:
            self.lora_A = Parameter(jax.random.normal(w_key, (self.in_features_flat, self.rank), dtype=jnp.float32) * (1 / self.in_features_flat))
            self.lora_B = Parameter(jnp.zeros((self.rank, self.out_features_flat), dtype=jnp.float32))

    def __call__(self, x: jax.Array, *args: Any, **kwargs: Any) -> jax.Array:
        base_out = self.base_layer(x, *args, **kwargs)
        # Flatten the input's feature dimensions to match in_features_flat
        in_dims = len(self.in_features)
        x_flat = x.reshape(x.shape[:-in_dims] + (self.in_features_flat,))

        lora_out = jnp.dot(jnp.dot(x_flat, self.lora_A.value.astype(x.dtype)), self.lora_B.value.astype(x.dtype)) * self.scaling

        # Reshape the output to match base_out shape
        out_dims = self.out_features
        lora_out = lora_out.reshape(lora_out.shape[:-1] + out_dims)

        return base_out + lora_out.astype(x.dtype)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.scaling * self.rank}"


__all__ = ['LoRALinear']
