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
"""Generic encoder-decoder transformer modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.layers.attention import Attention
from taktiny.nn.utils import (
    _constrain,
    _resolve_activation,
    _resolve_training,
    _validate_integer,
)
from taktiny.utils.typing import Activation, DType, PRNGKey, ShardMode

def _split_key(
    key: PRNGKey | None,
    count: int,
    *,
    required: bool,
) -> tuple[PRNGKey | None, ...]:
    if not required:
        return (None,) * count
    if key is None:
        raise ValueError('a key is required when transformer dropout is enabled')
    return tuple(jax.random.split(key, count))

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
        return mask[None, None, :, :]
    if mask.shape == (batch_size, query_length, key_length):
        return mask[:, None, :, :]
    target_shape = (batch_size, 1, query_length, key_length)
    try:
        return jnp.broadcast_to(mask, target_shape)
    except ValueError as error:
        raise ValueError(
            f'{name} with shape {mask.shape} cannot broadcast to {target_shape}'
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
    # Padding masks follow the common convention: True marks ignored tokens.
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
    is_causal: bool | jax.Array,
    *,
    query_length: int,
    key_length: int,
) -> jax.Array | None:
    if isinstance(is_causal, bool) and not is_causal:
        return mask
    query_positions = jnp.arange(query_length)[:, None]
    key_positions = jnp.arange(key_length)[None, :]
    causal = key_positions <= query_positions
    if not isinstance(is_causal, bool):
        causal = jnp.where(jnp.asarray(is_causal), causal, True)
    causal = causal[None, None, :, :]
    return causal if mask is None else mask & causal

def _attention_output(value: Any) -> jax.Array:
    return value[0] if isinstance(value, tuple) else value

class _FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Activation,
        dropout: float,
        bias: bool,
        dtype: DType,
        rngs: nn.Rngs,
        shard_mode: ShardMode,
        quant: Any,
        dot_general: Any,
    ) -> None:
        self.activation = _resolve_activation(activation)
        self.activation_name = getattr(
            self.activation,
            '__name__',
            type(self.activation).__name__,
        )
        self.input = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=('embed', 'mlp'),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.dropout = nn.Dropout(dropout, shard_mode=shard_mode)
        self.output = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=('mlp', 'embed'),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        key: PRNGKey | None,
        training: bool,
        out_sharding: jax.sharding.Sharding | None,
    ) -> jax.Array:
        x = self.activation(self.input(x))
        x = self.dropout(x, key=key, training=training)
        return self.output(x, out_sharding=out_sharding)

class TransformerEncoderLayer(nn.Module):
    """A self-attention and feed-forward transformer encoder block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        dropout: float,
        activation: Activation,
        norm_first: bool,
        norm_eps: float,
        bias: bool,
        dtype: DType,
        rngs: nn.Rngs,
        shard_mode: ShardMode,
        quant: Any,
        dot_general: Any,
    ) -> None:
        head_dim = hidden_size // num_heads
        self.self_attention = Attention(
            hidden_size,
            num_heads,
            head_dim,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            q_axis_names=('embed', 'heads', 'head_dim'),
            k_axis_names=('embed', 'heads', 'head_dim'),
            v_axis_names=('embed', 'heads', 'head_dim'),
            o_axis_names=('heads', 'head_dim', 'embed'),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.feed_forward = _FeedForward(
            hidden_size,
            intermediate_size,
            activation=activation,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.norm1 = nn.LayerNorm(
            hidden_size,
            eps=norm_eps,
            dtype=dtype,
            bias=bias,
            axis_names=('embed',),
            shard_mode=shard_mode,
        )
        self.norm2 = nn.LayerNorm(
            hidden_size,
            eps=norm_eps,
            dtype=dtype,
            bias=bias,
            axis_names=('embed',),
            shard_mode=shard_mode,
        )
        self.attention_dropout = nn.Dropout(dropout, shard_mode=shard_mode)
        self.output_dropout = nn.Dropout(dropout, shard_mode=shard_mode)
        self.norm_first = norm_first
        self.dropout = dropout

    def __call__(
        self,
        x: jax.Array,
        *,
        mask: jax.Array | None,
        is_causal: bool,
        key: PRNGKey | None,
        training: bool,
        out_sharding: jax.sharding.Sharding | None,
    ) -> jax.Array:
        attention_key, feed_key, output_key = _split_key(
            key,
            3,
            required=training and 0 < self.dropout < 1,
        )
        mask = _apply_causal_mask(
            mask,
            is_causal,
            query_length=x.shape[1],
            key_length=x.shape[1],
        )

        attention_input = self.norm1(x) if self.norm_first else x
        attention = _attention_output(
            self.self_attention(
                attention_input,
                attention_mask=mask,
                is_causal=False,
                out_sharding=out_sharding,
            )
        )
        x = x + self.attention_dropout(
            attention,
            key=attention_key,
            training=training,
        )
        if not self.norm_first:
            x = self.norm1(x, out_sharding=out_sharding)

        feed_input = self.norm2(x) if self.norm_first else x
        feed = self.feed_forward(
            feed_input,
            key=feed_key,
            training=training,
            out_sharding=out_sharding,
        )
        x = x + self.output_dropout(
            feed,
            key=output_key,
            training=training,
            out_sharding=out_sharding,
        )
        if not self.norm_first:
            x = self.norm2(x, out_sharding=out_sharding)
        return x

class TransformerDecoderLayer(nn.Module):
    """A transformer decoder block with optional cross-attention memory."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        dropout: float,
        activation: Activation,
        norm_first: bool,
        norm_eps: float,
        bias: bool,
        dtype: DType,
        rngs: nn.Rngs,
        shard_mode: ShardMode,
        quant: Any,
        dot_general: Any,
    ) -> None:
        head_dim = hidden_size // num_heads
        attention_options = {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'q_axis_names': ('embed', 'heads', 'head_dim'),
            'k_axis_names': ('embed', 'heads', 'head_dim'),
            'v_axis_names': ('embed', 'heads', 'head_dim'),
            'o_axis_names': ('heads', 'head_dim', 'embed'),
            'shard_mode': shard_mode,
            'quant': quant,
            'dot_general': dot_general,
        }
        self.self_attention = Attention(**attention_options)
        self.cross_attention = Attention(**attention_options)
        self.feed_forward = _FeedForward(
            hidden_size,
            intermediate_size,
            activation=activation,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        norm_options = {
            'eps': norm_eps,
            'dtype': dtype,
            'bias': bias,
            'axis_names': ('embed',),
            'shard_mode': shard_mode,
        }
        self.norm1 = nn.LayerNorm(hidden_size, **norm_options)
        self.norm2 = nn.LayerNorm(hidden_size, **norm_options)
        self.norm3 = nn.LayerNorm(hidden_size, **norm_options)
        self.self_attention_dropout = nn.Dropout(dropout, shard_mode=shard_mode)
        self.cross_attention_dropout = nn.Dropout(dropout, shard_mode=shard_mode)
        self.output_dropout = nn.Dropout(dropout, shard_mode=shard_mode)
        self.norm_first = norm_first
        self.dropout = dropout

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array | None,
        *,
        self_mask: jax.Array | None,
        memory_mask: jax.Array | None,
        self_is_causal: bool,
        memory_is_causal: bool,
        key: PRNGKey | None,
        training: bool,
        out_sharding: jax.sharding.Sharding | None,
    ) -> jax.Array:
        self_key, cross_key, feed_key, output_key = _split_key(
            key,
            4,
            required=training and 0 < self.dropout < 1,
        )
        self_mask = _apply_causal_mask(
            self_mask,
            self_is_causal,
            query_length=x.shape[1],
            key_length=x.shape[1],
        )
        if memory is not None:
            memory_mask = _apply_causal_mask(
                memory_mask,
                memory_is_causal,
                query_length=x.shape[1],
                key_length=memory.shape[1],
            )

        attention_input = self.norm1(x) if self.norm_first else x
        self_attention = _attention_output(
            self.self_attention(
                attention_input,
                attention_mask=self_mask,
                is_causal=False,
                out_sharding=out_sharding,
            )
        )
        x = x + self.self_attention_dropout(
            self_attention,
            key=self_key,
            training=training,
        )
        if not self.norm_first:
            x = self.norm1(x, out_sharding=out_sharding)

        if memory is not None:
            cross_input = self.norm2(x) if self.norm_first else x
            cross_attention = _attention_output(
                self.cross_attention(
                    cross_input,
                    context=memory,
                    attention_mask=memory_mask,
                    is_causal=False,
                    out_sharding=out_sharding,
                )
            )
            x = x + self.cross_attention_dropout(
                cross_attention,
                key=cross_key,
                training=training,
            )
            if not self.norm_first:
                x = self.norm2(x, out_sharding=out_sharding)

        feed_input = self.norm3(x) if self.norm_first else x
        feed = self.feed_forward(
            feed_input,
            key=feed_key,
            training=training,
            out_sharding=out_sharding,
        )
        x = x + self.output_dropout(
            feed,
            key=output_key,
            training=training,
            out_sharding=out_sharding,
        )
        if not self.norm_first:
            x = self.norm3(x, out_sharding=out_sharding)
        return x


class TransformerEncoder(nn.Module):
    """Apply encoder layers to batch-first hidden states.

    Each supplied layer must follow ``TransformerEncoderLayer.__call__``.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        norm: nn.Module | None = None,
        *,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if not layers:
            raise ValueError('layers must contain at least one encoder layer')
        self.layers = nn.List(layers)
        if norm is not None and not isinstance(norm, nn.Module):
            raise TypeError('norm must be an nn.Module or None')
        self.norm = norm
        self.shard_mode = shard_mode

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(
        self,
        source: jax.Array,
        mask: jax.Array | None = None,
        is_causal: bool = False,
        *,
        key: PRNGKey | None = None,
        training: bool | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        training = _resolve_training(self.training, training)
        mask = _apply_causal_mask(
            mask,
            is_causal,
            query_length=source.shape[1],
            key_length=source.shape[1],
        )
        needs_key = training and any(
            0 < getattr(layer, 'dropout', 0) < 1
            for layer in self.layers
        )
        keys = _split_key(key, len(self.layers), required=needs_key)
        output = source
        for layer, layer_key in zip(self.layers, keys):
            output = layer(
                output,
                mask=mask,
                is_causal=False,
                key=layer_key,
                training=training,
                out_sharding=out_sharding,
            )
        if self.norm is not None:
            output = self.norm(output)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return f'layers={len(self.layers)}'


class TransformerDecoder(nn.Module):
    """Apply decoder layers to batch-first hidden states.

    Each supplied layer must follow ``TransformerDecoderLayer.__call__``.
    Passing ``memory=None`` omits every cross-attention branch.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        norm: nn.Module | None = None,
        *,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if not layers:
            raise ValueError('layers must contain at least one decoder layer')
        self.layers = nn.List(layers)
        if norm is not None and not isinstance(norm, nn.Module):
            raise TypeError('norm must be an nn.Module or None')
        self.norm = norm
        self.shard_mode = shard_mode

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
        key: PRNGKey | None = None,
        training: bool | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        training = _resolve_training(self.training, training)
        target_mask = _apply_causal_mask(
            target_mask,
            target_is_causal,
            query_length=target.shape[1],
            key_length=target.shape[1],
        )
        if memory is not None:
            memory_mask = _apply_causal_mask(
                memory_mask,
                memory_is_causal,
                query_length=target.shape[1],
                key_length=memory.shape[1],
            )
        needs_key = training and any(
            0 < getattr(layer, 'dropout', 0) < 1
            for layer in self.layers
        )
        keys = _split_key(key, len(self.layers), required=needs_key)
        output = target
        for layer, layer_key in zip(self.layers, keys):
            output = layer(
                output,
                memory,
                self_mask=target_mask,
                memory_mask=memory_mask,
                self_is_causal=False,
                memory_is_causal=False,
                key=layer_key,
                training=training,
                out_sharding=out_sharding,
            )
        if self.norm is not None:
            output = self.norm(output)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return f'layers={len(self.layers)}'


class Transformer(nn.Module):
    """
    Apply a conventional encoder-decoder transformer.

    Inputs contain precomputed embeddings; positional information must be
    added by the caller. Attention masks use JAX semantics where ``True``
    permits attention. Key-padding masks use the common inverse convention
    where ``True`` marks a token that must be ignored. Passing ``target=None``
    runs only the encoder; passing ``source=None`` runs only the decoder.
    Custom components must follow the public ``TransformerEncoder`` and
    ``TransformerDecoder`` call contracts.
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
        custom_encoder: nn.Module | None = None,
        custom_decoder: nn.Module | None = None,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
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
        if custom_encoder is not None and not isinstance(custom_encoder, nn.Module):
            raise TypeError('custom_encoder must be an nn.Module or None')
        if custom_decoder is not None and not isinstance(custom_decoder, nn.Module):
            raise TypeError('custom_decoder must be an nn.Module or None')
        # Validate the activation before creating any parameters.
        activation_function = _resolve_activation(activation)
        self.activation_name = getattr(
            activation_function,
            '__name__',
            type(activation_function).__name__,
        )
        self.dropout = nn.Dropout(dropout).p
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.shard_mode = shard_mode

        layer_options = {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'intermediate_size': intermediate_size,
            'dropout': self.dropout,
            'activation': activation,
            'norm_first': norm_first,
            'norm_eps': norm_eps,
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'shard_mode': shard_mode,
            'quant': quant,
            'dot_general': dot_general,
        }
        if custom_encoder is None:
            encoder_norm = nn.LayerNorm(
                hidden_size,
                eps=norm_eps,
                dtype=dtype,
                bias=bias,
                axis_names=('embed',),
                shard_mode=shard_mode,
            )
            self.encoder = TransformerEncoder(
                [
                    TransformerEncoderLayer(**layer_options)
                    for _ in range(num_encoder_layers)
                ],
                encoder_norm,
                shard_mode=shard_mode,
            )
        else:
            self.encoder = custom_encoder

        if custom_decoder is None:
            decoder_norm = nn.LayerNorm(
                hidden_size,
                eps=norm_eps,
                dtype=dtype,
                bias=bias,
                axis_names=('embed',),
                shard_mode=shard_mode,
            )
            self.decoder = TransformerDecoder(
                [
                    TransformerDecoderLayer(**layer_options)
                    for _ in range(num_decoder_layers)
                ],
                decoder_norm,
                shard_mode=shard_mode,
            )
        else:
            self.decoder = custom_decoder

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
        target_is_causal: bool = False,
        memory_is_causal: bool = False,
        *,
        key: PRNGKey | None = None,
        training: bool | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        if source is None and target is None:
            raise ValueError('source and target cannot both be None')

        source_unbatched = None
        if source is not None:
            source, source_unbatched = _normalize_input(
                source,
                name='source',
                hidden_size=self.hidden_size,
                batch_first=self.batch_first,
            )
        elif source_mask is not None or source_key_padding_mask is not None:
            raise ValueError('source masks require a source input')

        target_unbatched = None
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
                raise ValueError(
                    'source and target must both be batched or unbatched'
                )
            if source.shape[0] != target.shape[0]:
                raise ValueError('source and target batch sizes must match')
        training = _resolve_training(self.training, training)

        component_count = int(source is not None) + int(target is not None)
        component_keys = (
            (None,) * component_count
            if key is None
            else (
                (key,)
                if component_count == 1
                else tuple(jax.random.split(key, component_count))
            )
        )
        key_index = 0

        memory = None
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
            source_attention_mask = _apply_causal_mask(
                source_attention_mask,
                source_is_causal,
                query_length=source_length,
                key_length=source_length,
            )
            memory = self.encoder(
                source,
                mask=source_attention_mask,
                is_causal=False,
                key=component_keys[key_index],
                training=training,
                out_sharding=None,
            )
            key_index += 1

        if target is None:
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
            target_attention_mask = _apply_causal_mask(
                target_attention_mask,
                target_is_causal,
                query_length=target_length,
                key_length=target_length,
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
                memory_attention_mask = _apply_causal_mask(
                    memory_attention_mask,
                    memory_is_causal,
                    query_length=target_length,
                    key_length=source_length,
                )
            elif memory_mask is not None or memory_key_padding_mask is not None:
                raise ValueError('memory masks require a source input')

            output = self.decoder(
                target,
                memory,
                target_mask=target_attention_mask,
                memory_mask=memory_attention_mask,
                target_is_causal=False,
                memory_is_causal=False,
                key=component_keys[key_index],
                training=training,
                out_sharding=None,
            )
            unbatched = target_unbatched

        if unbatched:
            output = output[0]
        elif not self.batch_first:
            output = jnp.swapaxes(output, 0, 1)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'hidden_size={self.hidden_size}, heads={self.num_heads}, '
            f'encoder_layers={self.num_encoder_layers}, '
            f'decoder_layers={self.num_decoder_layers}'
        )

__all__ = [
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerDecoder',
    'Transformer',
]
