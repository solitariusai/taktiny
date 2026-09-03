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
"""Embedding modules"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import qwix
from jax.nn import initializers

from taktiny.nn.base import Module, Parameter
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import _constrain
from taktiny.utils.typing import AxisNames, DType, Initializer

default_embedding_initializer = initializers.normal(0.02)


class Embedding(Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    """

    def __init__(
        self, num_embeddings: int,
        embedding_dim: int, *,
        rngs: Rngs | None = None,
        dtype: DType = jnp.float32,
        initializer: Initializer | None = None,
        quant: QuantConfig = None,
        axis_names: AxisNames | None = None,
    ) -> None:
        """Initializes the Embedding module.

        Args:
            num_embeddings (int): Size of the dictionary of embeddings.
            embedding_dim (int): The size of each embedding vector.
            rngs (Rngs | None, optional): Random number generators for initialization. Defaults to None.
            dtype (DType, optional): The data type of the embedding weights. Defaults to jnp.float32.
            initializer (Initializer | None, optional): Initialization function for the weights. Defaults to None.
            quant (Any, optional): Quantization configuration. Defaults to None.
            axis_names (AxisNames | None, optional): Axis names for sharding. Defaults to None.
            shard_mode (optional): Mode for sharding the output. Defaults to ShardMode.AUTO.
        """
        initializer = initializer or default_embedding_initializer
        if not isinstance(num_embeddings, int) or num_embeddings <= 0:
            raise ValueError('num_embeddings must be a positive integer')
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError('embedding_dim must be a positive integer')
        if axis_names is not None:
            axis_names = tuple(axis_names)
            if len(axis_names) != 2:
                raise ValueError(
                    'axis_names must contain vocabulary and embedding axes'
                )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if rngs is None:
            raise ValueError("A rngs must be provided to initialize Embedding layer")

        key = rngs()
        embedding_array = initializer(key, (num_embeddings, embedding_dim), dtype)
        
        if quant is not None:
            from taktiny.utils.quantization import resolve_quantization_rule, quantize_embedding_weight
            rule = resolve_quantization_rule(quant, '', op_name='embedding')
            if rule is not None:
                embedding_array = quantize_embedding_weight(embedding_array, rule)
                
        self.embedding = Parameter(embedding_array)
        
        if axis_names is not None:
            self.embedding.axis_names = axis_names

    def __call__(
        self,
        indices: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Looks up the embeddings for the given indices.

        Args:
            indices (jax.Array): Tensor containing the indices to look up.
            out_sharding (jax.sharding.Sharding | None, optional): Optional sharding specification for the output. Defaults to None.

        Returns:
            jax.Array: The retrieved embeddings.
        """
        table = self.embedding.value
        if isinstance(table, qwix.QArray):
            output = qwix.dequantize(table[indices])
        else:
            output = table[indices]

        return _constrain(output, out_sharding)

    # will bring back
    # @classmethod
    # def apply_gather_reduce(
    #     cls,
    #     operand: jax.Array,
    #     indices: jax.Array,
    #     weights: jax.Array | None = None,
    #     reduce_group_size: int = 1,
    #     **kwargs: Any,
    # ) -> jax.Array:
    #     """Apply Stream Gather Reduce kernel for sparse embeddings / reductions."""
    #     from taktiny.cosette.kernels.gather_reduce_sc import sc_gather_reduce
    #     if jax.default_backend() != "tpu":
    #         gathered = operand[indices]
    #         if weights is not None:
    #             gathered = gathered * weights[..., None]
    #         return gathered
    #     return sc_gather_reduce(
    #         operand,
    #         indices,
    #         topk_weights=weights,
    #         reduce_group_size=reduce_group_size,
    #         **kwargs,
    #     )

    # @classmethod
    # def apply_ragged_gather(
    #     cls,
    #     operand: jax.Array,
    #     offsets: jax.Array,
    #     lengths: jax.Array,
    #     **kwargs: Any,
    # ) -> jax.Array:
    #     """Apply Ragged Gather kernel."""
    #     from taktiny.cosette.kernels.ragged.ragged_gather import ragged_gather
    #     return ragged_gather(operand, offsets, lengths, **kwargs)

    # @classmethod
    # def apply(
    #     cls,
    #     operand: jax.Array,
    #     indices: jax.Array,
    #     kernel: str = "gather_reduce",
    #     **kwargs: Any,
    # ) -> jax.Array:
    #     """Unified entry point for Embedding sparse gather kernels."""
    #     if kernel in ("gather_reduce", "sc_gather_reduce"):
    #         return cls.apply_gather_reduce(operand, indices, **kwargs)
    #     elif kernel in ("ragged", "ragged_gather"):
    #         return cls.apply_ragged_gather(operand, indices, **kwargs)
    #     else:
    #         return operand[indices]

    def extra_repr(self) -> str:
        return f"{self.num_embeddings} → {self.embedding_dim}"



__all__ = [
    'Embedding',
]
