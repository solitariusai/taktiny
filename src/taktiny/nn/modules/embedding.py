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
import qwix
from jax.lax import PrecisionLike
from jax.nn import initializers
from jax.sharding import PartitionSpec
from jax.typing import DTypeLike

from taktiny.nn.base import Module, Parameter
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import _constrain, _normalize_shape
from taktiny.utils.quantization import (
    quantize_embedding_weight,
    resolve_quantization_rule,
)
from taktiny.utils.spmd import with_logical_partitioning
from taktiny.utils.typing import (
    AxisNames,
    DType,
    GenericShape,
    Initializer,
    MetaData,
    QuantConfig,
)

default_embed_initializer = initializers.normal(0.02)


class Embedding(Module):
    """Stores and looks up embeddings from an N-dimensional table.

    The embedding table has shape
    ``(*num_embeddings, *embed_features)``. When ``num_embeddings`` is an
    integer, ``indices`` follows conventional embedding semantics and an input
    of shape ``(...)`` produces ``(..., *embed_features)``.

    For an N-dimensional vocabulary, the final axis of ``indices`` contains a
    coordinate for every vocabulary axis. An input of shape
    ``(..., len(num_embeddings))`` therefore produces
    ``(..., *embed_features)``.

    Args:
        num_embeddings: Size of the vocabulary axes. An integer is treated as
            a one-dimensional vocabulary.
        embed_features: Size of the embedding feature axes. An integer is
            treated as a one-dimensional feature shape.
        dtype: Data type passed to the table initializer.
        rngs: Random number generator used to initialize the table.
        initializer: Function used to initialize the table. Defaults to a
            normal distribution with standard deviation 0.02.
        quant: Optional Qwix quantization configuration for the table.
        axis_names: Optional logical names for every vocabulary and embedding
            axis.
        partition_spec: Optional partition specification for the table.
        metadata: Optional metadata attached to the embedding parameter.
        precision: Precision reserved for embedding projection operations.
        preferred_element_type: Preferred result type reserved for embedding
            projection operations.

    Attributes:
        embedding: The learnable embedding-table parameter.

    Examples:
        Look up vectors from a conventional embedding table:

        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> embedding = nn.Embedding(8, 4, rngs=nn.Rngs(0))
        >>> embedding(jnp.asarray([1, 3])).shape
        (2, 4)

        Look up feature matrices using two-dimensional coordinates:

        >>> embedding = nn.Embedding((2, 3), (4, 5), rngs=nn.Rngs(1))
        >>> coordinates = jnp.asarray([[0, 1], [1, 2]])
        >>> embedding(coordinates).shape
        (2, 4, 5)
    """

    def __init__(
        self,
        num_embeddings: GenericShape,
        embed_features: GenericShape,
        *,
        dtype: DType | None = None,
        rngs: Rngs,
        initializer: Initializer = default_embed_initializer,
        quant: QuantConfig = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.num_embeddings = _normalize_shape(
            num_embeddings,
            'num_embeddings',
        )
        self.embed_features = _normalize_shape(
            embed_features,
            'embed_features',
        )
        self.precision = precision
        self.preferred_element_type = preferred_element_type

        if axis_names is not None:
            axis_names = tuple(axis_names)

        table_shape = self.num_embeddings + self.embed_features
        if axis_names is not None or partition_spec is not None:
            initializer = with_logical_partitioning(
                initializer,
                axis_names,
                partition_spec,
            )

        embed_array = initializer(rngs(), table_shape, dtype)
        if quant is not None:
            rule = resolve_quantization_rule(quant, '', op_name='embedding')
            if rule is not None:
                embed_array = quantize_embedding_weight(
                    embed_array,
                    rule,
                    vocabulary_axis_count=len(self.num_embeddings),
                )

        self.embedding = Parameter(
            embed_array,
            axis_names=axis_names,
            partition_spec=partition_spec,
            metadata=metadata,
        )

    def __call__(
        self,
        indices: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Looks up embeddings for integer indices or N-D coordinates.

        Args:
            indices: Integer indices. For an N-dimensional vocabulary, the
                final dimension must contain one coordinate per vocabulary
                axis.
            out_sharding: Optional sharding constraint for the result.

        Returns:
            The gathered embeddings followed by ``embed_features`` axes.

        Raises:
            ValueError: If N-D coordinates do not have the required trailing
                coordinate dimension.
        """
        table = self.embedding.value
        vocabulary_rank = len(self.num_embeddings)
        if vocabulary_rank == 1:
            table_indices: Any = indices
        else:
            if indices.ndim == 0 or indices.shape[-1] != vocabulary_rank:
                raise ValueError(
                    'indices trailing dimension must match the vocabulary '
                    f'rank {vocabulary_rank}, got shape {indices.shape}'
                )
            table_indices = tuple(
                indices[..., axis]
                for axis in range(vocabulary_rank)
            )

        if isinstance(table, qwix.QArray):
            output = qwix.dequantize(table[table_indices])
        else:
            output = table[table_indices]

        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        vocabulary = '×'.join(map(str, self.num_embeddings))
        features = '×'.join(map(str, self.embed_features))
        quantized = isinstance(self.embedding.value, qwix.QArray)
        quant = ' (Qwix PTQ)' if quantized else ''
        return f'{vocabulary} ➤ {features}{quant}'


__all__ = [
    'Embedding',
]
