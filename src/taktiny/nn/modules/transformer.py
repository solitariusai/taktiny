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
"""Generic encoder-decoder Transformer modules."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import cast

import jax
import jax.numpy as jnp
from jax.lax import PrecisionLike
from jax.nn import initializers
from jax.sharding import PartitionSpec
from jax.typing import DTypeLike

from taktiny.nn.base import Module
from taktiny.nn.block import List
from taktiny.nn.modules.linear import Linear
from taktiny.nn.modules.normalization import LayerNorm
from taktiny.nn.regularization import Dropout
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import (
    _constrain,
    _validate_integer,
    _validate_probability,
)
from taktiny.utils.typing import (
    Activation,
    AxisNames,
    DotGeneral,
    DType,
    GenericShape,
    Initializer,
    MetaData,
    QuantConfig,
)

type AxisNamesMap = Mapping[str, AxisNames]
type PartitionSpecMap = Mapping[str, PartitionSpec]
type AttentionContext = jax.Array | tuple[jax.Array, jax.Array]

default_transformer_initializer = initializers.xavier_uniform()
default_transformer_bias_initializer = initializers.zeros


def _mapping_value[T](
    values: Mapping[str, T] | None,
    name: str,
) -> T | None:
    return None if values is None else values.get(name)


def _resolve_activation(
    activation: Activation
) -> tuple[Callable[[jax.Array], jax.Array], str]:
    if isinstance(activation, str):
        name = activation.lower()
        functions: dict[str, Callable[[jax.Array], jax.Array]] = {
            'gelu': jax.nn.gelu,
            'relu': jax.nn.relu,
            'silu': jax.nn.silu,
        }
        if name not in functions:
            raise ValueError(
                "activation must be 'relu', 'gelu', 'silu', or callable"
            )
        return functions[name], name

    if not callable(activation):
        raise TypeError('activation must be a string or callable')
    function = cast(Callable[[jax.Array], jax.Array], activation)
    name = str(getattr(function, '__name__', type(function).__name__))
    return function, name


def _normalize_input(
    value: jax.Array,
    *,
    name: str,
    hidden_size: int,
    batch_first: bool,
) -> tuple[jax.Array, bool]:
    value = jnp.asarray(value)
    if value.ndim not in {2, 3}:
        raise ValueError(
            f'{name} must have shape [sequence, hidden] or contain one batch axis'
        )
    if value.shape[-1] != hidden_size:
        raise ValueError(
            f'{name} trailing size must be {hidden_size}, got {value.shape[-1]}'
        )
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(f'{name} must have a floating-point dtype')

    unbatched = value.ndim == 2
    if unbatched:
        value = value[None, ...]
    elif not batch_first:
        value = jnp.swapaxes(value, 0, 1)
    return value, unbatched


def _normalize_batch_first_input(
    value: jax.Array,
    *,
    name: str,
    hidden_size: int,
) -> tuple[jax.Array, bool]:
    value = jnp.asarray(value)
    if value.ndim not in {2, 3}:
        raise ValueError(
            f'{name} must have shape [sequence, hidden] or '
            '[batch, sequence, hidden]'
        )
    if value.shape[-1] != hidden_size:
        raise ValueError(
            f'{name} trailing size must be {hidden_size}, got {value.shape[-1]}'
        )
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(f'{name} must have a floating-point dtype')
    unbatched = value.ndim == 2
    return (value[None, ...] if unbatched else value), unbatched


def _normalize_attention_mask(
    mask: jax.Array | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    name: str,
) -> jax.Array | None:
    if mask is None:
        return None
    mask = jnp.asarray(mask)
    if mask.dtype != jnp.bool_:
        raise TypeError(f'{name} must be a boolean array')
    if mask.shape == (query_length, key_length):
        mask = mask[None, None, :, :]
    elif mask.shape == (batch_size, query_length, key_length):
        mask = mask[:, None, :, :]
    target_shape = (batch_size, 1, query_length, key_length)
    try:
        return jnp.broadcast_to(mask, target_shape)
    except ValueError as error:
        raise ValueError(
            f'{name} with shape {mask.shape} cannot broadcast to {target_shape}'
        ) from error


def _normalize_attention_bias(
    bias: jax.Array | None,
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
) -> jax.Array | None:
    if bias is None:
        return None
    bias = jnp.asarray(bias)
    if not jnp.issubdtype(bias.dtype, jnp.floating):
        raise TypeError('attention_bias must have a floating-point dtype')
    target_shape = (batch_size, num_heads, query_length, key_length)
    try:
        return jnp.broadcast_to(bias, target_shape)
    except ValueError as error:
        raise ValueError(
            f'attention_bias with shape {bias.shape} cannot broadcast to '
            f'{target_shape}'
        ) from error


def _normalize_padding_mask(
    mask: jax.Array | None,
    *,
    batch_size: int,
    key_length: int,
    name: str,
) -> jax.Array | None:
    if mask is None:
        return None
    mask = jnp.asarray(mask)
    if mask.dtype != jnp.bool_:
        raise TypeError(f'{name} must be a boolean array')
    if mask.shape == (key_length,) and batch_size == 1:
        mask = mask[None, :]
    if mask.shape != (batch_size, key_length):
        raise ValueError(
            f'{name} must have shape {(batch_size, key_length)}, got {mask.shape}'
        )
    # Padding masks use the common convention where True means "ignore".
    return (~mask)[:, None, None, :]


def _merge_masks(
    attention_mask: jax.Array | None,
    padding_mask: jax.Array | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    attention_name: str,
    padding_name: str,
) -> jax.Array | None:
    attention_mask = _normalize_attention_mask(
        attention_mask,
        batch_size=batch_size,
        query_length=query_length,
        key_length=key_length,
        name=attention_name,
    )
    padding_mask = _normalize_padding_mask(
        padding_mask,
        batch_size=batch_size,
        key_length=key_length,
        name=padding_name,
    )
    if attention_mask is None:
        return padding_mask
    if padding_mask is None:
        return attention_mask
    return attention_mask & padding_mask


def _apply_causal_mask(
    mask: jax.Array | None,
    is_causal: bool,
    *,
    query_length: int,
    key_length: int,
) -> jax.Array | None:
    if not isinstance(is_causal, bool):
        raise TypeError('is_causal must be a boolean')
    if not is_causal:
        return mask
    query_positions = jnp.arange(query_length)[:, None]
    key_positions = jnp.arange(key_length)[None, :]
    causal = (key_positions <= query_positions)[None, None, :, :]
    return causal if mask is None else mask & causal


def _parse_projection_bias(
    bias: bool | Sequence[bool],
) -> tuple[bool, bool, bool, bool]:
    if isinstance(bias, bool):
        return (bias, bias, bias, bias)
    values = tuple(bias)
    if len(values) != 4 or not all(isinstance(value, bool) for value in values):
        raise ValueError('bias must be a boolean or four q, k, v, o booleans')
    return values


def _new_linear(
    in_features: GenericShape,
    out_features: GenericShape,
    *,
    bias: bool,
    dtype: DType | None,
    rngs: Rngs,
    kernel_initializer: Initializer,
    bias_initializer: Initializer,
    quant: QuantConfig,
    dot_general: DotGeneral | None,
    axis_names: AxisNames | None,
    partition_spec: PartitionSpec | None,
    kernel_metadata: MetaData | None,
    bias_metadata: MetaData | None,
    precision: PrecisionLike,
    preferred_element_type: DTypeLike | None,
) -> Linear:
    return Linear(
        in_features,
        out_features,
        bias=bias,
        dtype=dtype,
        rngs=rngs,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        quant=quant,
        dot_general=dot_general,
        axis_names=axis_names,
        partition_spec=partition_spec,
        kernel_metadata=kernel_metadata,
        bias_metadata=bias_metadata,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


class Attention(Module):
    r"""Apply scaled dot-product multi-head attention.

    Query, key, and value projections split ``hidden_size`` into
    ``num_heads`` heads. Attention is computed as
    ``softmax(Q K^T / sqrt(head_dim)) V`` and projected back to
    ``hidden_size``. Passing ``context`` enables cross-attention; otherwise
    the module performs self-attention. Boolean masks use JAX semantics:
    ``True`` permits a query-key pair and ``False`` blocks it.

    ``axis_names`` and ``partition_spec`` are optional mappings with
    ``q_proj``, ``k_proj``, ``v_proj``, and ``o_proj`` keys.

    Example:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> attention = nn.Attention(8, 2, rngs=nn.Rngs(0), dropout=0.0)
        >>> attention(jnp.ones((3, 5, 8))).shape
        (3, 5, 8)

    Reference:
        Ashish Vaswani et al., "Attention Is All You Need" (2017),
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        *,
        context_dim: int | None = None,
        dropout: float = 0.0,
        scaling: float | None = None,
        bias: bool | Sequence[bool] = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_transformer_initializer,
        bias_initializer: Initializer = default_transformer_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNamesMap | None = None,
        partition_spec: PartitionSpecMap | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.num_heads = _validate_integer(num_heads, 'num_heads')
        if head_dim is None:
            if hidden_size % num_heads:
                raise ValueError('hidden_size must be divisible by num_heads')
            head_dim = hidden_size // num_heads
        self.head_dim = _validate_integer(head_dim, 'head_dim')
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError('num_heads * head_dim must equal hidden_size')
        self.context_dim = (
            hidden_size
            if context_dim is None
            else _validate_integer(context_dim, 'context_dim')
        )
        self.dropout = _validate_probability(
            dropout,
            'dropout',
            allow_one=False,
        )
        if scaling is not None and (
            isinstance(scaling, bool)
            or not isinstance(scaling, (int, float))
            or not math.isfinite(scaling)
            or scaling <= 0
        ):
            raise ValueError('scaling must be finite and positive')
        self.scaling = (
            float(scaling) if scaling is not None else self.head_dim**-0.5
        )
        q_bias, k_bias, v_bias, o_bias = _parse_projection_bias(bias)

        self.q_proj = _new_linear(
            hidden_size,
            (num_heads, self.head_dim),
            bias=q_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_mapping_value(axis_names, 'q_proj'),
            partition_spec=_mapping_value(partition_spec, 'q_proj'),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.k_proj = _new_linear(
            self.context_dim,
            (num_heads, self.head_dim),
            bias=k_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_mapping_value(axis_names, 'k_proj'),
            partition_spec=_mapping_value(partition_spec, 'k_proj'),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.v_proj = _new_linear(
            self.context_dim,
            (num_heads, self.head_dim),
            bias=v_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_mapping_value(axis_names, 'v_proj'),
            partition_spec=_mapping_value(partition_spec, 'v_proj'),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.o_proj = _new_linear(
            (num_heads, self.head_dim),
            hidden_size,
            bias=o_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_mapping_value(axis_names, 'o_proj'),
            partition_spec=_mapping_value(partition_spec, 'o_proj'),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.attention_dropout = Dropout(self.dropout, rngs=rngs)
        self.precision = precision
        self.preferred_element_type = preferred_element_type

    def __call__(
        self,
        x: jax.Array,
        context: AttentionContext | None = None,
        attention_mask: jax.Array | None = None,
        attention_bias: jax.Array | None = None,
        is_causal: bool = False,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Apply self-attention or cross-attention to batch-first inputs."""
        query_input, unbatched = _normalize_batch_first_input(
            x,
            name='x',
            hidden_size=self.hidden_size,
        )
        key_unbatched = unbatched
        value_unbatched = unbatched
        if context is None:
            key_input = value_input = query_input
        elif isinstance(context, tuple):
            key_input, key_unbatched = _normalize_batch_first_input(
                context[0],
                name='context key',
                hidden_size=self.context_dim,
            )
            value_input, value_unbatched = _normalize_batch_first_input(
                context[1],
                name='context value',
                hidden_size=self.context_dim,
            )
        else:
            key_input, key_unbatched = _normalize_batch_first_input(
                context,
                name='context',
                hidden_size=self.context_dim,
            )
            value_input = key_input
            value_unbatched = key_unbatched

        if key_unbatched != value_unbatched:
            raise ValueError('context key and value must use the same batching')
        if unbatched != key_unbatched:
            raise ValueError('x and context must both be batched or unbatched')
        if query_input.shape[0] != key_input.shape[0]:
            raise ValueError('x and context batch sizes must match')
        if key_input.shape[:2] != value_input.shape[:2]:
            raise ValueError('context key and value leading shapes must match')

        query = self.q_proj(query_input)
        key = self.k_proj(key_input)
        value = self.v_proj(value_input)
        batch_size, query_length = query.shape[:2]
        key_length = key.shape[1]
        mask = _normalize_attention_mask(
            attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
            name='attention_mask',
        )
        mask = _apply_causal_mask(
            mask,
            is_causal,
            query_length=query_length,
            key_length=key_length,
        )
        normalized_bias = _normalize_attention_bias(
            attention_bias,
            batch_size=batch_size,
            num_heads=self.num_heads,
            query_length=query_length,
            key_length=key_length,
        )

        scores = jnp.einsum(
            'bqhd,bkhd->bhqk',
            query,
            key,
            precision=self.precision,
            preferred_element_type=self.preferred_element_type,
        ) * self.scaling
        if normalized_bias is not None:
            scores += normalized_bias
        if mask is None:
            weights = jax.nn.softmax(scores, axis=-1)
        else:
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
            weights = jax.nn.softmax(scores, axis=-1)
            weights = jnp.where(mask, weights, 0)
        weights = self.attention_dropout(weights)
        output = jnp.einsum(
            'bhqk,bkhd->bqhd',
            weights,
            value,
            precision=self.precision,
            preferred_element_type=self.preferred_element_type,
        )
        output = self.o_proj(output, out_sharding=out_sharding)
        return output[0] if unbatched else output

    def extra_repr(self) -> str:
        context = (
            ''
            if self.context_dim == self.hidden_size
            else f', context={self.context_dim}'
        )
        return (
            f'{self.hidden_size} ➤ {self.num_heads}×{self.head_dim}'
            f'{context}, dropout={self.dropout:g}'
        )


class FeedForward(Module):
    """Apply the Transformer's position-wise feed-forward network.

    Every sequence position is transformed independently by
    ``Linear(hidden, intermediate)``, an activation, dropout, and
    ``Linear(intermediate, hidden)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> feed_forward = nn.FeedForward(8, 32, rngs=nn.Rngs(0), dropout=0.0)
        >>> feed_forward(jnp.ones((2, 5, 8))).shape
        (2, 5, 8)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Activation = 'relu',
        dropout: float = 0.0,
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_transformer_initializer,
        bias_initializer: Initializer = default_transformer_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNamesMap | None = None,
        partition_spec: PartitionSpecMap | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.intermediate_size = _validate_integer(
            intermediate_size,
            'intermediate_size',
        )
        self.activation, self.activation_name = _resolve_activation(activation)
        self.dropout = _validate_probability(
            dropout,
            'dropout',
            allow_one=False,
        )
        self.input = _new_linear(
            hidden_size,
            intermediate_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_mapping_value(axis_names, 'input'),
            partition_spec=_mapping_value(partition_spec, 'input'),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.output = _new_linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_mapping_value(axis_names, 'output'),
            partition_spec=_mapping_value(partition_spec, 'output'),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.activation_dropout = Dropout(self.dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        hidden = self.activation(self.input(x))
        hidden = self.activation_dropout(hidden)
        return self.output(hidden, out_sharding=out_sharding)

    def extra_repr(self) -> str:
        return (
            f'{self.hidden_size} ➤ {self.intermediate_size} ➤ '
            f'{self.hidden_size}, activation={self.activation_name}, '
            f'dropout={self.dropout:g}'
        )


class TransformerEncoderLayer(Module):
    """Apply one Transformer encoder layer.

    The original post-normalization order is used by default. Set
    ``norm_first=True`` for the common pre-normalization variant.

    Example:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> layer = nn.TransformerEncoderLayer(8, 2, 32, rngs=nn.Rngs(0), dropout=0.0)
        >>> layer(jnp.ones((2, 5, 8))).shape
        (2, 5, 8)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        dropout: float = 0.1,
        activation: Activation = 'relu',
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_transformer_initializer,
        bias_initializer: Initializer = default_transformer_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNamesMap | None = None,
        partition_spec: PartitionSpecMap | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.num_heads = _validate_integer(num_heads, 'num_heads')
        self.intermediate_size = _validate_integer(
            intermediate_size,
            'intermediate_size',
        )
        if hidden_size % num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        if not isinstance(norm_first, bool):
            raise TypeError('norm_first must be a boolean')
        self.norm_first = norm_first
        self.dropout = _validate_probability(
            dropout,
            'dropout',
            allow_one=False,
        )
        self.self_attention = Attention(
            hidden_size,
            num_heads,
            dropout=self.dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=axis_names,
            partition_spec=partition_spec,
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            activation=activation,
            dropout=self.dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=axis_names,
            partition_spec=partition_spec,
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        norm_axis_names = _mapping_value(axis_names, 'norm')
        norm_partition_spec = _mapping_value(partition_spec, 'norm')
        self.norm1 = LayerNorm(
            hidden_size,
            epsilon=norm_eps,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=norm_axis_names,
            partition_spec=norm_partition_spec,
        )
        self.norm2 = LayerNorm(
            hidden_size,
            epsilon=norm_eps,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=norm_axis_names,
            partition_spec=norm_partition_spec,
        )
        self.attention_dropout = Dropout(self.dropout, rngs=rngs)
        self.output_dropout = Dropout(self.dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        is_causal: bool = False,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x, unbatched = _normalize_batch_first_input(
            x,
            name='x',
            hidden_size=self.hidden_size,
        )
        if self.norm_first:
            attention = self.self_attention(
                self.norm1(x),
                attention_mask=mask,
                is_causal=is_causal,
            )
            x = x + self.attention_dropout(attention)
            feed = self.feed_forward(self.norm2(x))
            x = x + self.output_dropout(feed)
        else:
            attention = self.self_attention(
                x,
                attention_mask=mask,
                is_causal=is_causal,
            )
            x = self.norm1(x + self.attention_dropout(attention))
            feed = self.feed_forward(x)
            x = self.norm2(x + self.output_dropout(feed))
        x = _constrain(x, out_sharding)
        return x[0] if unbatched else x

    def extra_repr(self) -> str:
        return (
            f'{self.hidden_size}, heads={self.num_heads}, '
            f'intermediate={self.intermediate_size}, '
            f'norm_first={self.norm_first}'
        )


class TransformerDecoderLayer(Module):
    """Apply one Transformer decoder layer.

    The layer contains masked self-attention, encoder-decoder cross-attention,
    and a position-wise feed-forward network. ``memory=None`` skips the
    cross-attention branch, allowing decoder-only use.

    Example:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> layer = nn.TransformerDecoderLayer(8, 2, 32, rngs=nn.Rngs(0), dropout=0.0)
        >>> layer(jnp.ones((2, 4, 8)), jnp.ones((2, 6, 8)), self_is_causal=True).shape
        (2, 4, 8)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        dropout: float = 0.1,
        activation: Activation = 'relu',
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_transformer_initializer,
        bias_initializer: Initializer = default_transformer_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNamesMap | None = None,
        partition_spec: PartitionSpecMap | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.num_heads = _validate_integer(num_heads, 'num_heads')
        self.intermediate_size = _validate_integer(
            intermediate_size,
            'intermediate_size',
        )
        if hidden_size % num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        if not isinstance(norm_first, bool):
            raise TypeError('norm_first must be a boolean')
        self.norm_first = norm_first
        self.dropout = _validate_probability(
            dropout,
            'dropout',
            allow_one=False,
        )
        self.self_attention = Attention(
            hidden_size,
            num_heads,
            dropout=self.dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=axis_names,
            partition_spec=partition_spec,
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.cross_attention = Attention(
            hidden_size,
            num_heads,
            dropout=self.dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=axis_names,
            partition_spec=partition_spec,
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            activation=activation,
            dropout=self.dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=axis_names,
            partition_spec=partition_spec,
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        norm_axis_names = _mapping_value(axis_names, 'norm')
        norm_partition_spec = _mapping_value(partition_spec, 'norm')
        self.norm1 = LayerNorm(
            hidden_size,
            epsilon=norm_eps,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=norm_axis_names,
            partition_spec=norm_partition_spec,
        )
        self.norm2 = LayerNorm(
            hidden_size,
            epsilon=norm_eps,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=norm_axis_names,
            partition_spec=norm_partition_spec,
        )
        self.norm3 = LayerNorm(
            hidden_size,
            epsilon=norm_eps,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=norm_axis_names,
            partition_spec=norm_partition_spec,
        )
        self.self_attention_dropout = Dropout(self.dropout, rngs=rngs)
        self.cross_attention_dropout = Dropout(self.dropout, rngs=rngs)
        self.output_dropout = Dropout(self.dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array | None = None,
        self_mask: jax.Array | None = None,
        memory_mask: jax.Array | None = None,
        self_is_causal: bool = False,
        memory_is_causal: bool = False,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x, unbatched = _normalize_batch_first_input(
            x,
            name='x',
            hidden_size=self.hidden_size,
        )
        if memory is not None:
            memory, memory_unbatched = _normalize_batch_first_input(
                memory,
                name='memory',
                hidden_size=self.hidden_size,
            )
            if memory_unbatched != unbatched:
                raise ValueError('x and memory must both be batched or unbatched')
            if x.shape[0] != memory.shape[0]:
                raise ValueError('x and memory batch sizes must match')
        elif memory_mask is not None:
            raise ValueError('memory_mask requires memory')

        if self.norm_first:
            self_attention = self.self_attention(
                self.norm1(x),
                attention_mask=self_mask,
                is_causal=self_is_causal,
            )
            x = x + self.self_attention_dropout(self_attention)
            if memory is not None:
                cross_attention = self.cross_attention(
                    self.norm2(x),
                    context=memory,
                    attention_mask=memory_mask,
                    is_causal=memory_is_causal,
                )
                x = x + self.cross_attention_dropout(cross_attention)
            feed = self.feed_forward(self.norm3(x))
            x = x + self.output_dropout(feed)
        else:
            self_attention = self.self_attention(
                x,
                attention_mask=self_mask,
                is_causal=self_is_causal,
            )
            x = self.norm1(x + self.self_attention_dropout(self_attention))
            if memory is not None:
                cross_attention = self.cross_attention(
                    x,
                    context=memory,
                    attention_mask=memory_mask,
                    is_causal=memory_is_causal,
                )
                x = self.norm2(
                    x + self.cross_attention_dropout(cross_attention)
                )
            feed = self.feed_forward(x)
            x = self.norm3(x + self.output_dropout(feed))
        x = _constrain(x, out_sharding)
        return x[0] if unbatched else x

    def extra_repr(self) -> str:
        return (
            f'{self.hidden_size}, heads={self.num_heads}, '
            f'intermediate={self.intermediate_size}, '
            f'norm_first={self.norm_first}'
        )


class TransformerEncoder(Module):
    """Apply a stack of Transformer encoder layers to batch-first inputs."""

    def __init__(
        self,
        layers: Sequence[Module],
        norm: Module | None = None,
    ) -> None:
        if not layers:
            raise ValueError('layers must contain at least one encoder layer')
        if norm is not None and not isinstance(norm, Module):
            raise TypeError('norm must be a Module or None')
        self.layers = List(layers)
        self.norm = norm

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(
        self,
        source: jax.Array,
        mask: jax.Array | None = None,
        is_causal: bool = False,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        output = source
        for layer in self.layers:
            output = layer(
                output,
                mask=mask,
                is_causal=is_causal,
                out_sharding=out_sharding,
            )
        if self.norm is not None:
            output = self.norm(output, out_sharding=out_sharding)
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        return f'layers={len(self.layers)}'


class TransformerDecoder(Module):
    """Apply a stack of Transformer decoder layers to batch-first inputs."""

    def __init__(
        self,
        layers: Sequence[Module],
        norm: Module | None = None,
    ) -> None:
        if not layers:
            raise ValueError('layers must contain at least one decoder layer')
        if norm is not None and not isinstance(norm, Module):
            raise TypeError('norm must be a Module or None')
        self.layers = List(layers)
        self.norm = norm

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(
        self,
        target: jax.Array,
        memory: jax.Array | None = None,
        target_mask: jax.Array | None = None,
        memory_mask: jax.Array | None = None,
        target_is_causal: bool = False,
        memory_is_causal: bool = False,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        output = target
        for layer in self.layers:
            output = layer(
                output,
                memory,
                self_mask=target_mask,
                memory_mask=memory_mask,
                self_is_causal=target_is_causal,
                memory_is_causal=memory_is_causal,
                out_sharding=out_sharding,
            )
        if self.norm is not None:
            output = self.norm(output, out_sharding=out_sharding)
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        return f'layers={len(self.layers)}'


class Transformer(Module):
    """Apply the encoder-decoder architecture from Attention Is All You Need.

    Inputs are embedded hidden states; token embeddings and positional
    encodings remain the caller's responsibility. By default, unbatched inputs
    use ``[sequence, hidden]`` and batched inputs use
    ``[sequence, batch, hidden]``. Set ``batch_first=True`` for
    ``[batch, sequence, hidden]``.

    Attention masks are boolean arrays where ``True`` permits attention.
    Key-padding masks use the inverse convention where ``True`` marks an
    ignored key position. The target is causal by default, matching the
    autoregressive decoder in the original paper.

    With the original ``norm_first=False`` layout, normalization occurs only
    after each residual sublayer and no extra stack-final norm is created.
    The ``norm_first=True`` variant adds a final norm after each complete
    encoder or decoder stack.

    ``axis_names`` and ``partition_spec`` accept ``q_proj``, ``k_proj``,
    ``v_proj``, ``o_proj``, ``input``, ``output``, and ``norm`` keys.

    Example:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> model = nn.Transformer(
        ...     hidden_size=8,
        ...     num_heads=2,
        ...     num_encoder_layers=1,
        ...     num_decoder_layers=1,
        ...     intermediate_size=32,
        ...     dropout=0.0,
        ...     batch_first=True,
        ...     rngs=nn.Rngs(0),
        ... )
        >>> model(jnp.ones((2, 5, 8)), jnp.ones((2, 3, 8))).shape
        (2, 3, 8)

    Reference:
        Ashish Vaswani et al., "Attention Is All You Need" (2017),
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        activation: Activation = 'relu',
        *,
        norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        custom_encoder: Module | None = None,
        custom_decoder: Module | None = None,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_transformer_initializer,
        bias_initializer: Initializer = default_transformer_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNamesMap | None = None,
        partition_spec: PartitionSpecMap | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.num_heads = _validate_integer(num_heads, 'num_heads')
        self.num_encoder_layers = _validate_integer(
            num_encoder_layers,
            'num_encoder_layers',
        )
        self.num_decoder_layers = _validate_integer(
            num_decoder_layers,
            'num_decoder_layers',
        )
        self.intermediate_size = _validate_integer(
            intermediate_size,
            'intermediate_size',
        )
        if hidden_size % num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        if not isinstance(batch_first, bool):
            raise TypeError('batch_first must be a boolean')
        if not isinstance(norm_first, bool):
            raise TypeError('norm_first must be a boolean')
        if custom_encoder is not None and not isinstance(custom_encoder, Module):
            raise TypeError('custom_encoder must be a Module or None')
        if custom_decoder is not None and not isinstance(custom_decoder, Module):
            raise TypeError('custom_decoder must be a Module or None')
        _, self.activation_name = _resolve_activation(activation)
        self.dropout = _validate_probability(
            dropout,
            'dropout',
            allow_one=False,
        )
        self.batch_first = batch_first
        self.norm_first = norm_first

        norm_axis_names = _mapping_value(axis_names, 'norm')
        norm_partition_spec = _mapping_value(partition_spec, 'norm')
        if custom_encoder is None:
            encoder_norm = (
                LayerNorm(
                    hidden_size,
                    epsilon=norm_eps,
                    bias=bias,
                    dtype=dtype,
                    rngs=rngs,
                    axis_names=norm_axis_names,
                    partition_spec=norm_partition_spec,
                )
                if norm_first
                else None
            )
            encoder: Module = TransformerEncoder(
                [
                    TransformerEncoderLayer(
                        hidden_size,
                        num_heads,
                        intermediate_size,
                        dropout=self.dropout,
                        activation=activation,
                        norm_first=norm_first,
                        norm_eps=norm_eps,
                        bias=bias,
                        dtype=dtype,
                        rngs=rngs,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        quant=quant,
                        dot_general=dot_general,
                        axis_names=axis_names,
                        partition_spec=partition_spec,
                        kernel_metadata=kernel_metadata,
                        bias_metadata=bias_metadata,
                        precision=precision,
                        preferred_element_type=preferred_element_type,
                    )
                    for _ in range(num_encoder_layers)
                ],
                encoder_norm,
            )
        else:
            encoder = custom_encoder
        self.encoder = encoder

        if custom_decoder is None:
            decoder_norm = (
                LayerNorm(
                    hidden_size,
                    epsilon=norm_eps,
                    bias=bias,
                    dtype=dtype,
                    rngs=rngs,
                    axis_names=norm_axis_names,
                    partition_spec=norm_partition_spec,
                )
                if norm_first
                else None
            )
            decoder: Module = TransformerDecoder(
                [
                    TransformerDecoderLayer(
                        hidden_size,
                        num_heads,
                        intermediate_size,
                        dropout=self.dropout,
                        activation=activation,
                        norm_first=norm_first,
                        norm_eps=norm_eps,
                        bias=bias,
                        dtype=dtype,
                        rngs=rngs,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        quant=quant,
                        dot_general=dot_general,
                        axis_names=axis_names,
                        partition_spec=partition_spec,
                        kernel_metadata=kernel_metadata,
                        bias_metadata=bias_metadata,
                        precision=precision,
                        preferred_element_type=preferred_element_type,
                    )
                    for _ in range(num_decoder_layers)
                ],
                decoder_norm,
            )
        else:
            decoder = custom_decoder
        self.decoder = decoder

    def __call__(
        self,
        source: jax.Array | None,
        target: jax.Array | None = None,
        source_mask: jax.Array | None = None,
        target_mask: jax.Array | None = None,
        memory_mask: jax.Array | None = None,
        source_key_padding_mask: jax.Array | None = None,
        target_key_padding_mask: jax.Array | None = None,
        memory_key_padding_mask: jax.Array | None = None,
        source_is_causal: bool = False,
        target_is_causal: bool = True,
        memory_is_causal: bool = False,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        if source is None and target is None:
            raise ValueError('source and target cannot both be None')

        source_unbatched: bool | None = None
        if source is not None:
            source, source_unbatched = _normalize_input(
                source,
                name='source',
                hidden_size=self.hidden_size,
                batch_first=self.batch_first,
            )
        elif source_mask is not None or source_key_padding_mask is not None:
            raise ValueError('source masks require a source input')

        target_unbatched: bool | None = None
        if target is not None:
            target, target_unbatched = _normalize_input(
                target,
                name='target',
                hidden_size=self.hidden_size,
                batch_first=self.batch_first,
            )
        elif any(
            value is not None
            for value in (
                target_mask,
                memory_mask,
                target_key_padding_mask,
                memory_key_padding_mask,
            )
        ):
            raise ValueError('target and memory masks require a target input')

        if source is not None and target is not None:
            if source_unbatched != target_unbatched:
                raise ValueError('source and target must use the same batching')
            if source.shape[0] != target.shape[0]:
                raise ValueError('source and target batch sizes must match')

        memory: jax.Array | None = None
        if source is not None:
            batch_size, source_length = source.shape[:2]
            source_attention_mask = _merge_masks(
                source_mask,
                source_key_padding_mask,
                batch_size=batch_size,
                query_length=source_length,
                key_length=source_length,
                attention_name='source_mask',
                padding_name='source_key_padding_mask',
            )
            memory = self.encoder(
                source,
                mask=source_attention_mask,
                is_causal=source_is_causal,
            )

        if target is None:
            if memory is None:
                raise RuntimeError('encoder did not produce an output')
            output = memory
            unbatched = source_unbatched
        else:
            batch_size, target_length = target.shape[:2]
            target_attention_mask = _merge_masks(
                target_mask,
                target_key_padding_mask,
                batch_size=batch_size,
                query_length=target_length,
                key_length=target_length,
                attention_name='target_mask',
                padding_name='target_key_padding_mask',
            )
            memory_attention_mask = None
            if memory is not None:
                source_length = memory.shape[1]
                memory_attention_mask = _merge_masks(
                    memory_mask,
                    memory_key_padding_mask,
                    batch_size=batch_size,
                    query_length=target_length,
                    key_length=source_length,
                    attention_name='memory_mask',
                    padding_name='memory_key_padding_mask',
                )
            elif memory_mask is not None or memory_key_padding_mask is not None:
                raise ValueError('memory masks require a source input')
            output = self.decoder(
                target,
                memory,
                target_mask=target_attention_mask,
                memory_mask=memory_attention_mask,
                target_is_causal=target_is_causal,
                memory_is_causal=memory_is_causal,
            )
            unbatched = target_unbatched

        if unbatched:
            output = output[0]
        elif not self.batch_first:
            output = jnp.swapaxes(output, 0, 1)
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        return (
            f'{self.hidden_size}, heads={self.num_heads}, '
            f'encoder_layers={self.num_encoder_layers}, '
            f'decoder_layers={self.num_decoder_layers}, '
            f'intermediate={self.intermediate_size}'
        )


__all__ = [
    'Attention',
    'FeedForward',
    'Transformer',
    'TransformerDecoder',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerEncoderLayer',
    'default_transformer_bias_initializer',
    'default_transformer_initializer',
]
