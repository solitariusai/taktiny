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

from collections.abc import Mapping, Sequence
import inspect
import math
from typing import Any

import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.cosettes.layers.attention import AttentionLegacy, JointAttention
from taktiny.cosettes.layers.ffn import FeedForward, GLUMBConv
from taktiny.cosettes.layers.normalization import AdaXNorm, NormType
from taktiny.nn.continuo import (
    _constrain,
    _resolve_activation,
    _validate_integer,
    _validate_probability,
)
from taktiny.utils.typing import (
    Activation,
    DType,
    ShardMode,
    Sharding,
)


ModuleSpec = nn.Module | type[nn.Module]


def _instantiate_module(
    module: ModuleSpec,
    *,
    name: str,
    options: dict[str, Any],
) -> nn.Module:
    if isinstance(module, nn.Module):
        return module
    if not isinstance(module, type) or not issubclass(module, nn.Module):
        raise TypeError(f'{name} must be an nn.Module subclass or instance')
    parameters = inspect.signature(module).parameters.values()
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if not accepts_kwargs:
        accepted = {
            parameter.name
            for parameter in parameters
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        options = {
            key: value for key, value in options.items() if key in accepted
        }
    return module(**options)

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
        self.self_attention = AttentionLegacy(
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
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            activation=activation,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            input_axis_names=('embed', 'mlp'),
            output_axis_names=('mlp', 'embed'),
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
        self.attention_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.output_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.norm_first = norm_first
        self.dropout = dropout

    def __call__(
        self,
        x: jax.Array,
        *,
        mask: jax.Array | None,
        is_causal: bool,
        out_sharding: jax.sharding.Sharding | None,
    ) -> jax.Array:
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
        )
        if not self.norm_first:
            x = self.norm1(x, out_sharding=out_sharding)

        feed_input = self.norm2(x) if self.norm_first else x
        feed = self.feed_forward(
            feed_input,
            out_sharding=out_sharding,
        )
        x = x + self.output_dropout(
            feed,
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
        self.self_attention = AttentionLegacy(**attention_options)
        self.cross_attention = AttentionLegacy(**attention_options)
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            activation=activation,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            input_axis_names=('embed', 'mlp'),
            output_axis_names=('mlp', 'embed'),
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
        self.self_attention_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.cross_attention_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.output_dropout = nn.Dropout(
            dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
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
        out_sharding: jax.sharding.Sharding | None,
    ) -> jax.Array:
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
            )
            if not self.norm_first:
                x = self.norm2(x, out_sharding=out_sharding)

        feed_input = self.norm3(x) if self.norm_first else x
        feed = self.feed_forward(
            feed_input,
            out_sharding=out_sharding,
        )
        x = x + self.output_dropout(
            feed,
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
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        mask = _apply_causal_mask(
            mask,
            is_causal,
            query_length=source.shape[1],
            key_length=source.shape[1],
        )
        output = source
        for layer in self.layers:
            output = layer(
                output,
                mask=mask,
                is_causal=False,
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
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
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
        output = target
        for layer in self.layers:
            output = layer(
                output,
                memory,
                self_mask=target_mask,
                memory_mask=memory_mask,
                self_is_causal=False,
                memory_is_causal=False,
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
        self.dropout = _validate_probability(dropout, 'dropout')
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
                out_sharding=None,
            )

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


class ConditionalTransformerLayer(nn.Module):
    """A modulated stream with optional read-only cross-attention context.

    A six-way conditioning vector modulates self-attention and feed-forward
    branches. ``encoder_hidden_states`` supplies keys and values to
    cross-attention but is not itself updated. Architecture-specific
    feed-forward arguments are passed through ``ffn_kwargs`` so the same
    residual topology can use a token MLP, convolutional FFN, or another
    compatible module.
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        head_dim: int | None = None,
        cross_num_heads: int | None = None,
        cross_head_dim: int | None = None,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        activation: Activation = 'gelu',
        norm_eps: float = 1e-6,
        norm_elementwise_affine: bool = False,
        bias: bool = True,
        attention_out_bias: bool = True,
        cross_attention_bias: bool = True,
        mlp_bias: bool = True,
        use_qkv_norm: bool = False,
        qkv_norm_across_heads: bool = False,
        qkv_norm_eps: float = 1e-5,
        pos_emb: nn.Module | None = None,
        cross_pos_emb: nn.Module | None = None,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
        input_layernorm: ModuleSpec = nn.LayerNorm,
        self_attention: ModuleSpec = AttentionLegacy,
        cross_attention: ModuleSpec | None = AttentionLegacy,
        cross_attention_layernorm: ModuleSpec | None = None,
        post_attention_layernorm: ModuleSpec = nn.LayerNorm,
        mlp: ModuleSpec = GLUMBConv,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.context_size = _validate_integer(context_size, 'context_size')
        self.num_heads = _validate_integer(num_heads, 'num_heads')
        self.intermediate_size = _validate_integer(
            intermediate_size,
            'intermediate_size',
        )
        if head_dim is None:
            if hidden_size % num_heads:
                raise ValueError(
                    'hidden_size must be divisible by num_heads when '
                    'head_dim is not provided'
                )
            head_dim = hidden_size // num_heads
        self.head_dim = _validate_integer(head_dim, 'head_dim')
        cross_num_heads = (
            num_heads if cross_num_heads is None else cross_num_heads
        )
        cross_head_dim = (
            self.head_dim if cross_head_dim is None else cross_head_dim
        )
        self.cross_num_heads = _validate_integer(
            cross_num_heads,
            'cross_num_heads',
        )
        self.cross_head_dim = _validate_integer(
            cross_head_dim,
            'cross_head_dim',
        )
        self.dropout = _validate_probability(dropout, 'dropout')
        self.ffn_dropout = _validate_probability(
            ffn_dropout,
            'ffn_dropout',
        )
        self.shard_mode = shard_mode

        norm_options = {
            'normalized_shape': hidden_size,
            'eps': norm_eps,
            'elementwise_affine': norm_elementwise_affine,
            'dtype': dtype,
            'bias': norm_elementwise_affine,
            'axis_names': ('embed',),
            'shard_mode': shard_mode,
        }
        self.norm1 = _instantiate_module(
            input_layernorm,
            name='input_layernorm',
            options=norm_options,
        )
        self.attn1 = _instantiate_module(
            self_attention,
            name='self_attention',
            options={
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'head_dim': self.head_dim,
                'pos_emb': pos_emb,
                'bias': False,
                'q_bias': bias,
                'k_bias': bias,
                'v_bias': bias,
                'o_bias': attention_out_bias,
                'use_qk_norm': use_qkv_norm,
                'qk_norm_across_heads': qkv_norm_across_heads,
                'qk_norm_eps': qkv_norm_eps,
                'dtype': dtype,
                'rngs': rngs,
                'q_axis_names': ('embed', 'heads', 'head_dim'),
                'k_axis_names': ('embed', 'heads', 'head_dim'),
                'v_axis_names': ('embed', 'heads', 'head_dim'),
                'o_axis_names': ('heads', 'head_dim', 'embed'),
                'dropout': self.dropout,
                'shard_mode': shard_mode,
                'quant': quant,
                'dot_general': dot_general,
            },
        )
        self.attn2 = (
            None
            if cross_attention is None
            else _instantiate_module(
                cross_attention,
                name='cross_attention',
                options={
                    'hidden_size': hidden_size,
                    'num_heads': self.cross_num_heads,
                    'head_dim': self.cross_head_dim,
                    'context_dim': context_size,
                    'pos_emb': cross_pos_emb,
                    'bias': False,
                    'q_bias': cross_attention_bias,
                    'k_bias': cross_attention_bias,
                    'v_bias': cross_attention_bias,
                    'o_bias': attention_out_bias,
                    'use_qk_norm': use_qkv_norm,
                    'qk_norm_across_heads': qkv_norm_across_heads,
                    'qk_norm_eps': qkv_norm_eps,
                    'dtype': dtype,
                    'rngs': rngs,
                    'q_axis_names': (
                        'embed',
                        'cross_heads',
                        'cross_head_dim',
                    ),
                    'k_axis_names': (
                        'context_embed',
                        'cross_heads',
                        'cross_head_dim',
                    ),
                    'v_axis_names': (
                        'context_embed',
                        'cross_heads',
                        'cross_head_dim',
                    ),
                    'o_axis_names': (
                        'cross_heads',
                        'cross_head_dim',
                        'embed',
                    ),
                    'dropout': self.dropout,
                    'shard_mode': shard_mode,
                    'quant': quant,
                    'dot_general': dot_general,
                },
            )
        )
        self.norm_cross = (
            None
            if cross_attention_layernorm is None
            else _instantiate_module(
                cross_attention_layernorm,
                name='cross_attention_layernorm',
                options=norm_options,
            )
        )
        self.norm2 = _instantiate_module(
            post_attention_layernorm,
            name='post_attention_layernorm',
            options=norm_options,
        )
        self.ff = _instantiate_module(
            mlp,
            name='mlp',
            options={
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size,
                'activation': activation,
                'dropout': self.ffn_dropout,
                'bias': mlp_bias,
                'norm_type': None,
                'residual_connection': False,
                'norm_eps': norm_eps,
                'dtype': dtype,
                'rngs': rngs,
                'shard_mode': shard_mode,
                'quant': quant,
                'dot_general': dot_general,
            },
        )

        table = jax.random.normal(
            rngs(),
            (6, hidden_size),
            dtype=jnp.float32,
        ) / math.sqrt(hidden_size)
        self.scale_shift_table = nn.Parameter(table)
        self.scale_shift_table.axis_names = ('modulation', 'embed')
        self.self_attention_dropout = nn.Dropout(
            self.dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.cross_attention_dropout = nn.Dropout(
            self.dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.feed_forward_dropout = nn.Dropout(
            self.ffn_dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )

    @staticmethod
    def _modulate(
        normalized: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        if shift.ndim == normalized.ndim - 1:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        elif shift.ndim != normalized.ndim:
            raise ValueError(
                'modulation must share the activation rank or omit its '
                'sequence axis'
            )
        return normalized * (1.0 + scale) + shift

    @staticmethod
    def _gate(value: jax.Array, gate: jax.Array) -> jax.Array:
        if gate.ndim == value.ndim - 1:
            gate = gate[:, None, :]
        elif gate.ndim != value.ndim:
            raise ValueError(
                'gate must share the activation rank or omit its sequence axis'
            )
        return value * gate

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | None,
        conditioning: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        encoder_attention_mask: jax.Array | None = None,
        position_idx: jax.Array | None = None,
        encoder_position_idx: jax.Array | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
        self_attention_kernel: str = 'dot_product',
        cross_attention_kernel: str = 'dot_product',
        ffn_kwargs: Mapping[str, Any] | None = None,
    ) -> jax.Array:
        hidden_states = jnp.asarray(hidden_states)
        conditioning = jnp.asarray(conditioning)
        if (
            hidden_states.ndim != 3
            or hidden_states.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                'hidden_states must have shape [batch, sequence, hidden_size]'
            )
        if encoder_hidden_states is not None:
            encoder_hidden_states = jnp.asarray(encoder_hidden_states)
            if (
                encoder_hidden_states.ndim != 3
                or encoder_hidden_states.shape[-1] != self.context_size
            ):
                raise ValueError(
                    'encoder_hidden_states must have shape '
                    '[batch, sequence, context_size]'
                )
            if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
                raise ValueError(
                    'hidden and context streams must share a batch size'
                )

        expected_flat = (hidden_states.shape[0], 6 * self.hidden_size)
        expected_grouped = (hidden_states.shape[0], 6, self.hidden_size)
        expected_token_flat = (
            hidden_states.shape[0],
            hidden_states.shape[1],
            6 * self.hidden_size,
        )
        expected_token_grouped = (
            hidden_states.shape[0],
            hidden_states.shape[1],
            6,
            self.hidden_size,
        )
        if conditioning.shape == expected_grouped:
            modulation = conditioning
        elif conditioning.shape == expected_flat:
            modulation = conditioning.reshape(expected_grouped)
        elif conditioning.shape == expected_token_grouped:
            modulation = conditioning
        elif conditioning.shape == expected_token_flat:
            modulation = conditioning.reshape(expected_token_grouped)
        else:
            raise ValueError(
                'conditioning must be batch-wise or token-wise with six '
                f'modulation groups; got '
                f'{conditioning.shape}'
            )

        modulation = modulation.astype(hidden_states.dtype)
        table = self.scale_shift_table.value.astype(hidden_states.dtype)
        table = table[None, :, :] if modulation.ndim == 3 else table[
            None, None, :, :
        ]
        modulation = modulation + table
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_feed,
            scale_feed,
            gate_feed,
        ) = tuple(jnp.take(modulation, index, axis=-2) for index in range(6))

        normalized = self._modulate(
            self.norm1(hidden_states, out_sharding=out_sharding),
            shift_attention,
            scale_attention,
        )
        attention = _attention_output(
            self.attn1(
                normalized,
                attention_mask=attention_mask,
                is_causal=False,
                position_idx=position_idx,
                out_sharding=out_sharding,
                kernel=self_attention_kernel,
            )
        )
        hidden_states = hidden_states + self._gate(
            self.self_attention_dropout(attention),
            gate_attention,
        )

        if encoder_hidden_states is not None and self.attn2 is not None:
            cross_input = (
                hidden_states
                if self.norm_cross is None
                else self.norm_cross(
                    hidden_states,
                    out_sharding=out_sharding,
                )
            )
            cross_attention = _attention_output(
                self.attn2(
                    cross_input,
                    context=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    is_causal=False,
                    position_idx=encoder_position_idx,
                    out_sharding=out_sharding,
                    kernel=cross_attention_kernel,
                )
            )
            hidden_states = hidden_states + self.cross_attention_dropout(
                cross_attention
            )

        normalized = self._modulate(
            self.norm2(hidden_states, out_sharding=out_sharding),
            shift_feed,
            scale_feed,
        )
        ffn_kwargs = {} if ffn_kwargs is None else dict(ffn_kwargs)
        if 'out_sharding' in ffn_kwargs:
            raise ValueError('ffn_kwargs must not contain out_sharding')
        feed = self.ff(
            normalized,
            out_sharding=out_sharding,
            **ffn_kwargs,
        )
        hidden_states = hidden_states + self._gate(
            self.feed_forward_dropout(feed),
            gate_feed,
        )
        return _constrain(hidden_states, out_sharding, self.shard_mode)


class JointTransformerLayer(nn.Module):
    """A conditioned two-stream transformer layer with joint attention.

    The layer follows the MMDiT data path used by diffusion transformers while
    remaining independent of a particular model configuration. Image and
    context streams own separate normalization, modulation, output projection,
    and feed-forward parameters. Their projected QKV tensors are concatenated
    only for the shared attention operation.

    Adaptive normalizers produce six modulation groups per updated stream:
    attention shift, scale, and gate followed by feed-forward shift, scale,
    and gate. ``dual_attention=True`` adds a second hidden-stream attention and
    three more modulation groups. With ``context_pre_only=True``, context is
    normalized for joint attention but is not updated or returned.

    Inputs and outputs use ``[batch, sequence, hidden]`` layout. Joint masks
    describe the concatenated hidden/context sequence and use ``True`` for
    permitted attention positions. The return value and ``out_shardings`` use
    ``(encoder_hidden_states, hidden_states)`` order.
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        context_intermediate_size: int | None = None,
        conditioning_size: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        activation: Activation = 'gelu',
        norm: NormType = 'layernorm',
        norm_eps: float = 1e-6,
        context_pre_only: bool = False,
        dual_attention: bool = False,
        bias: bool = True,
        use_qkv_norm: bool = False,
        qkv_norm_eps: float = 1e-6,
        context_first: bool = False,
        scaling: float | None = None,
        pos_emb: nn.Module | None = None,
        second_pos_emb: nn.Module | None = None,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
        project_conditioning: bool = True,
        context_project_conditioning: bool | None = None,
        input_layernorm: ModuleSpec = AdaXNorm,
        context_input_layernorm: ModuleSpec = AdaXNorm,
        joint_attention: ModuleSpec = JointAttention,
        second_attention: ModuleSpec = AttentionLegacy,
        post_attention_layernorm: ModuleSpec | None = None,
        context_post_attention_layernorm: ModuleSpec | None = None,
        mlp: ModuleSpec = FeedForward,
        context_mlp: ModuleSpec = FeedForward,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.context_size = _validate_integer(context_size, 'context_size')
        self.num_heads = _validate_integer(num_heads, 'num_heads')
        self.intermediate_size = _validate_integer(
            intermediate_size,
            'intermediate_size',
        )
        context_intermediate_size = (
            intermediate_size
            if context_intermediate_size is None
            else context_intermediate_size
        )
        self.context_intermediate_size = _validate_integer(
            context_intermediate_size,
            'context_intermediate_size',
        )
        conditioning_size = (
            hidden_size if conditioning_size is None else conditioning_size
        )
        self.conditioning_size = _validate_integer(
            conditioning_size,
            'conditioning_size',
        )
        if head_dim is None:
            if hidden_size % num_heads:
                raise ValueError(
                    'hidden_size must be divisible by num_heads when '
                    'head_dim is not provided'
                )
            head_dim = hidden_size // num_heads
        self.head_dim = _validate_integer(head_dim, 'head_dim')
        if not isinstance(context_pre_only, bool):
            raise TypeError('context_pre_only must be a boolean')
        if not isinstance(dual_attention, bool):
            raise TypeError('dual_attention must be a boolean')
        if not isinstance(context_first, bool):
            raise TypeError('context_first must be a boolean')
        if not isinstance(project_conditioning, bool):
            raise TypeError('project_conditioning must be a boolean')
        if context_project_conditioning is None:
            context_project_conditioning = project_conditioning
        if not isinstance(context_project_conditioning, bool):
            raise TypeError(
                'context_project_conditioning must be a boolean or None'
            )
        if norm not in {'layernorm', 'rmsnorm'}:
            raise ValueError("norm must be 'layernorm' or 'rmsnorm'")
        if pos_emb is not None and not isinstance(pos_emb, nn.Module):
            raise TypeError('pos_emb must be an nn.Module or None')
        if second_pos_emb is not None and not isinstance(
            second_pos_emb,
            nn.Module,
        ):
            raise TypeError('second_pos_emb must be an nn.Module or None')

        self.context_pre_only = context_pre_only
        self.dual_attention = dual_attention
        self.context_first = context_first
        self.norm_type = norm
        self.norm_eps = norm_eps
        self.dropout = _validate_probability(dropout, 'dropout')
        self.shard_mode = shard_mode
        self.project_conditioning = project_conditioning
        self.context_project_conditioning = context_project_conditioning

        hidden_modulation_groups = 9 if dual_attention else 6
        self.norm1 = _instantiate_module(
            input_layernorm,
            name='input_layernorm',
            options={
                'embedding_dim': self.conditioning_size,
                'out_dim': hidden_size * hidden_modulation_groups,
                'norm': norm,
                'eps': norm_eps,
                'activation': 'silu',
                'bias': bias,
                'project': project_conditioning,
                'dtype': dtype,
                'rngs': rngs,
                'quant': quant,
                'dot_general': dot_general,
                'axis_names': ('conditioning', 'hidden_modulation'),
                'shard_mode': shard_mode,
            },
        )
        context_modulation_groups = 2 if context_pre_only else 6
        self.norm1_context = _instantiate_module(
            context_input_layernorm,
            name='context_input_layernorm',
            options={
                'embedding_dim': self.conditioning_size,
                'out_dim': context_size * context_modulation_groups,
                'norm': norm,
                'eps': norm_eps,
                'activation': 'silu',
                'bias': bias,
                'project': context_project_conditioning,
                'dtype': dtype,
                'rngs': rngs,
                'quant': quant,
                'dot_general': dot_general,
                'axis_names': ('conditioning', 'context_modulation'),
                'shard_mode': shard_mode,
            },
        )

        self.attn = _instantiate_module(
            joint_attention,
            name='joint_attention',
            options={
                'hidden_size1': hidden_size,
                'hidden_size2': context_size,
                'num_heads': num_heads,
                'head_dim': self.head_dim,
                'pos_emb': pos_emb,
                'bias': bias,
                'use_qk_norm': use_qkv_norm,
                'qk_norm_eps': qkv_norm_eps,
                'dtype': dtype,
                'rngs': rngs,
                'q_axis_names': ('embed', 'heads', 'head_dim'),
                'k_axis_names': ('embed', 'heads', 'head_dim'),
                'v_axis_names': ('embed', 'heads', 'head_dim'),
                'o_axis_names': ('heads', 'head_dim', 'embed'),
                'scaling': scaling,
                'context_first': context_first,
                'shard_mode': shard_mode,
                'quant': quant,
                'dot_general': dot_general,
            },
        )
        if dual_attention:
            self.attn2 = _instantiate_module(
                second_attention,
                name='second_attention',
                options={
                    'hidden_size': hidden_size,
                    'num_heads': num_heads,
                    'head_dim': self.head_dim,
                    'pos_emb': second_pos_emb,
                    'bias': bias,
                    'use_qk_norm': use_qkv_norm,
                    'qk_norm_eps': qkv_norm_eps,
                    'dtype': dtype,
                    'rngs': rngs,
                    'q_axis_names': ('embed', 'heads', 'head_dim'),
                    'k_axis_names': ('embed', 'heads', 'head_dim'),
                    'v_axis_names': ('embed', 'heads', 'head_dim'),
                    'o_axis_names': ('heads', 'head_dim', 'embed'),
                    'scaling': scaling,
                    'shard_mode': shard_mode,
                    'quant': quant,
                    'dot_general': dot_general,
                },
            )
        else:
            self.attn2 = None

        self.norm2 = self._make_norm(
            hidden_size,
            norm,
            norm_eps,
            dtype,
            shard_mode,
            'embed',
            post_attention_layernorm,
        )
        self.ff = _instantiate_module(
            mlp,
            name='mlp',
            options={
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size,
                'activation': activation,
                'dropout': self.dropout,
                'bias': bias,
                'dtype': dtype,
                'rngs': rngs,
                'input_axis_names': ('embed', 'mlp'),
                'output_axis_names': ('mlp', 'embed'),
                'shard_mode': shard_mode,
                'quant': quant,
                'dot_general': dot_general,
            },
        )
        if context_pre_only:
            self.norm2_context = None
            self.ff_context = None
        else:
            self.norm2_context = self._make_norm(
                context_size,
                norm,
                norm_eps,
                dtype,
                shard_mode,
                'context_embed',
                context_post_attention_layernorm,
            )
            self.ff_context = _instantiate_module(
                context_mlp,
                name='context_mlp',
                options={
                    'hidden_size': context_size,
                    'intermediate_size': self.context_intermediate_size,
                    'activation': activation,
                    'dropout': self.dropout,
                    'bias': bias,
                    'dtype': dtype,
                    'rngs': rngs,
                    'input_axis_names': ('context_embed', 'context_mlp'),
                    'output_axis_names': ('context_mlp', 'context_embed'),
                    'shard_mode': shard_mode,
                    'quant': quant,
                    'dot_general': dot_general,
                },
            )

        self.hidden_attention_dropout = nn.Dropout(
            self.dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.context_attention_dropout = nn.Dropout(
            self.dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.dual_attention_dropout = nn.Dropout(
            self.dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.hidden_output_dropout = nn.Dropout(
            self.dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.context_output_dropout = nn.Dropout(
            self.dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )

    @staticmethod
    def _make_norm(
        hidden_size: int,
        norm: NormType,
        eps: float,
        dtype: DType,
        shard_mode: ShardMode,
        axis_name: str,
        module: ModuleSpec | None,
    ) -> nn.Module:
        options = {
            'eps': eps,
            'dtype': dtype,
            'axis_names': (axis_name,),
            'shard_mode': shard_mode,
        }
        if module is None:
            module = nn.LayerNorm if norm == 'layernorm' else nn.RMSNorm
        if isinstance(module, nn.Module):
            return module
        if not isinstance(module, type) or not issubclass(module, nn.Module):
            raise TypeError(
                'post-attention normalization must be an nn.Module subclass '
                'or instance'
            )
        if issubclass(module, nn.LayerNorm):
            return module(
                hidden_size,
                elementwise_affine=False,
                **options,
            )
        if issubclass(module, nn.RMSNorm):
            options['epsilon'] = options.pop('eps')
            return module(
                hidden_size,
                with_scale=False,
                **options,
            )
        return module(hidden_size, **options)

    @staticmethod
    def _validate_stream(
        value: jax.Array,
        *,
        name: str,
        hidden_size: int,
    ) -> jax.Array:
        value = jnp.asarray(value)
        if value.ndim != 3:
            raise ValueError(
                f'{name} must have shape [batch, sequence, hidden]'
            )
        if value.shape[-1] != hidden_size:
            raise ValueError(
                f'{name} must end in {hidden_size}, got {value.shape}'
            )
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(f'{name} must have a floating-point dtype')
        return value

    @staticmethod
    def _modulate(
        normalized: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        if shift.ndim == normalized.ndim - 1:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        elif shift.ndim != normalized.ndim:
            raise ValueError(
                'modulation rank must match the activation rank or omit '
                'its sequence axis'
            )
        return normalized * (1.0 + scale) + shift

    @staticmethod
    def _gate(value: jax.Array, gate: jax.Array) -> jax.Array:
        if gate.ndim == value.ndim - 1:
            gate = gate[:, None, :]
        elif gate.ndim != value.ndim:
            raise ValueError(
                'gate rank must match the activation rank or omit its '
                'sequence axis'
            )
        return gate * value

    @staticmethod
    def _adaptive_groups(
        module: AdaXNorm,
        value: jax.Array,
        conditioning: jax.Array,
        count: int,
        width: int,
        *,
        out_sharding: jax.sharding.Sharding | None,
        modulation_sharding: jax.sharding.Sharding | None,
        name: str,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        normalized, modulation = module(
            value,
            conditioning,
            out_sharding=out_sharding,
            modulation_sharding=modulation_sharding,
        )
        expected_feature = count * width
        valid_shapes = {
            (value.shape[0], expected_feature),
            (value.shape[0], value.shape[1], expected_feature),
        }
        if modulation.shape not in valid_shapes:
            raise ValueError(
                f'{name} modulation must have shape '
                f'[batch, {expected_feature}] or '
                f'[batch, sequence, {expected_feature}], got '
                f'{modulation.shape}'
            )
        return normalized, tuple(jnp.split(modulation, count, axis=-1))

    @staticmethod
    def _output_shardings(
        out_sharding: jax.sharding.Sharding | None,
        out_shardings: tuple[Sharding, Sharding] | None,
    ) -> tuple[jax.sharding.Sharding | None, jax.sharding.Sharding | None]:
        if out_shardings is None:
            return out_sharding, out_sharding
        if len(out_shardings) != 2:
            raise ValueError('out_shardings must contain exactly two values')
        return out_shardings

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        conditioning: jax.Array,
        *,
        context_conditioning: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        attention_bias: jax.Array | None = None,
        hidden_attention_mask: jax.Array | None = None,
        is_causal: bool = False,
        hidden_is_causal: bool = False,
        position_idx: jax.Array | None = None,
        hidden_position_idx: jax.Array | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
        out_shardings: tuple[Sharding, Sharding] | None = None,
        modulation_sharding: jax.sharding.Sharding | None = None,
        kernel: str = 'dot_product',
        **kernel_kwargs: Any,
    ) -> tuple[jax.Array | None, jax.Array]:
        hidden_states = self._validate_stream(
            hidden_states,
            name='hidden_states',
            hidden_size=self.hidden_size,
        )
        encoder_hidden_states = self._validate_stream(
            encoder_hidden_states,
            name='encoder_hidden_states',
            hidden_size=self.context_size,
        )
        if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError('hidden and context streams must share a batch size')
        conditioning = jnp.asarray(conditioning)
        valid_conditioning_shapes = {
            (hidden_states.shape[0], self.conditioning_size),
            (
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.conditioning_size,
            ),
        }
        if conditioning.shape not in valid_conditioning_shapes:
            raise ValueError(
                'conditioning must have shape [batch, conditioning] or '
                '[batch, hidden_sequence, conditioning], got '
                f'{conditioning.shape}'
            )
        if not jnp.issubdtype(conditioning.dtype, jnp.floating):
            raise TypeError('conditioning must have a floating-point dtype')
        if context_conditioning is None:
            context_conditioning = conditioning
        else:
            context_conditioning = jnp.asarray(context_conditioning)
        valid_context_conditioning_shapes = {
            (encoder_hidden_states.shape[0], self.conditioning_size),
            (
                encoder_hidden_states.shape[0],
                encoder_hidden_states.shape[1],
                self.conditioning_size,
            ),
        }
        if context_conditioning.shape not in valid_context_conditioning_shapes:
            raise ValueError(
                'context_conditioning must have shape [batch, conditioning] '
                'or [batch, context_sequence, conditioning], got '
                f'{context_conditioning.shape}'
            )
        if not jnp.issubdtype(context_conditioning.dtype, jnp.floating):
            raise TypeError(
                'context_conditioning must have a floating-point dtype'
            )

        context_sharding, hidden_sharding = self._output_shardings(
            out_sharding,
            out_shardings,
        )
        hidden_group_count = 9 if self.dual_attention else 6
        normalized_hidden_base, hidden_groups = self._adaptive_groups(
            self.norm1,
            hidden_states,
            conditioning,
            hidden_group_count,
            self.hidden_size,
            out_sharding=hidden_sharding,
            modulation_sharding=modulation_sharding,
            name='hidden',
        )
        (
            hidden_shift_attention,
            hidden_scale_attention,
            hidden_gate_attention,
            hidden_shift_feed,
            hidden_scale_feed,
            hidden_gate_feed,
            *hidden_dual_groups,
        ) = hidden_groups
        normalized_hidden = self._modulate(
            normalized_hidden_base,
            hidden_shift_attention,
            hidden_scale_attention,
        )

        context_group_count = 2 if self.context_pre_only else 6
        normalized_context, context_groups = self._adaptive_groups(
            self.norm1_context,
            encoder_hidden_states,
            context_conditioning,
            context_group_count,
            self.context_size,
            out_sharding=context_sharding,
            modulation_sharding=modulation_sharding,
            name='context',
        )
        if self.context_pre_only:
            context_scale_attention, context_shift_attention = context_groups
            normalized_context = self._modulate(
                normalized_context,
                context_shift_attention,
                context_scale_attention,
            )
        else:
            (
                context_shift_attention,
                context_scale_attention,
                context_gate_attention,
                context_shift_feed,
                context_scale_feed,
                context_gate_feed,
            ) = context_groups
            normalized_context = self._modulate(
                normalized_context,
                context_shift_attention,
                context_scale_attention,
            )

        hidden_attention, context_attention = self.attn(
            normalized_hidden,
            normalized_context,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            is_causal=is_causal,
            position_idx=position_idx,
            out_shardings=(hidden_sharding, context_sharding),
            kernel=kernel,
            **kernel_kwargs,
        )
        hidden_attention = self.hidden_attention_dropout(
            self._gate(hidden_attention, hidden_gate_attention),
            out_sharding=hidden_sharding,
        )
        hidden_states = hidden_states + hidden_attention

        if self.dual_attention:
            hidden_shift_dual, hidden_scale_dual, hidden_gate_dual = (
                hidden_dual_groups
            )
            normalized_hidden_dual = self._modulate(
                normalized_hidden_base,
                hidden_shift_dual,
                hidden_scale_dual,
            )
            dual_attention = _attention_output(
                self.attn2(
                    normalized_hidden_dual,
                    attention_mask=hidden_attention_mask,
                    is_causal=hidden_is_causal,
                    position_idx=hidden_position_idx,
                    out_sharding=hidden_sharding,
                    kernel=kernel,
                )
            )
            dual_attention = self.dual_attention_dropout(
                self._gate(dual_attention, hidden_gate_dual),
                out_sharding=hidden_sharding,
            )
            hidden_states = hidden_states + dual_attention

        normalized_hidden_feed = self._modulate(
            self.norm2(hidden_states),
            hidden_shift_feed,
            hidden_scale_feed,
        )
        hidden_feed = self.ff(
            normalized_hidden_feed,
            out_sharding=hidden_sharding,
        )
        hidden_feed = self.hidden_output_dropout(
            self._gate(hidden_feed, hidden_gate_feed),
            out_sharding=hidden_sharding,
        )
        hidden_states = hidden_states + hidden_feed

        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attention = self.context_attention_dropout(
                self._gate(context_attention, context_gate_attention),
                out_sharding=context_sharding,
            )
            encoder_hidden_states = (
                encoder_hidden_states + context_attention
            )
            normalized_context_feed = self._modulate(
                self.norm2_context(encoder_hidden_states),
                context_shift_feed,
                context_scale_feed,
            )
            context_feed = self.ff_context(
                normalized_context_feed,
                out_sharding=context_sharding,
            )
            context_feed = self.context_output_dropout(
                self._gate(context_feed, context_gate_feed),
                out_sharding=context_sharding,
            )
            encoder_hidden_states = encoder_hidden_states + context_feed
            encoder_hidden_states = _constrain(
                encoder_hidden_states,
                context_sharding,
                self.shard_mode,
            )

        hidden_states = _constrain(
            hidden_states,
            hidden_sharding,
            self.shard_mode,
        )
        return encoder_hidden_states, hidden_states

    def extra_repr(self) -> str:
        return (
            f'{self.hidden_size} + {self.context_size}, '
            f'heads={self.num_heads}, head_dim={self.head_dim}, '
            f'context_pre_only={self.context_pre_only}, '
            f'dual_attention={self.dual_attention}, '
            f'context_first={self.context_first}'
        )


class ParallelAttentionMLP(nn.Module):
    """Compute pre-output self-attention and an MLP in parallel.

    This is the single-stream path used by Chroma and LongCat Image. Attention
    deliberately has no output projection of its own: projected attention and
    activated MLP features are concatenated and reduced by one shared output
    projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        *,
        activation: Activation = 'gelu',
        pos_emb: nn.Module | None = None,
        eps: float = 1e-6,
        bias: bool = True,
        use_qkv_norm: bool = True,
        scaling: float | None = None,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.num_heads = _validate_integer(num_heads, 'num_heads')
        self.head_dim = _validate_integer(head_dim, 'head_dim')
        self.intermediate_size = _validate_integer(
            intermediate_size,
            'intermediate_size',
        )
        if hidden_size != num_heads * head_dim:
            raise ValueError('hidden_size must equal num_heads * head_dim')
        if not isinstance(use_qkv_norm, bool):
            raise TypeError('use_qkv_norm must be a boolean')
        if pos_emb is not None and not isinstance(pos_emb, nn.Module):
            raise TypeError('pos_emb must be an nn.Module or None')

        self.activation = _resolve_activation(activation)
        self.pos_emb = pos_emb
        self.scaling = scaling
        projection_options = {
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'shard_mode': shard_mode,
            'quant': quant,
            'dot_general': dot_general,
        }
        self.q_proj = nn.Linear(
            hidden_size,
            (num_heads, head_dim),
            axis_names=('embed', 'heads', 'head_dim'),
            **projection_options,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            (num_heads, head_dim),
            axis_names=('embed', 'heads', 'head_dim'),
            **projection_options,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            (num_heads, head_dim),
            axis_names=('embed', 'heads', 'head_dim'),
            **projection_options,
        )
        self.proj_mlp = nn.Linear(
            hidden_size,
            intermediate_size,
            axis_names=('embed', 'mlp'),
            **projection_options,
        )
        self.proj_out = nn.Linear(
            hidden_size + intermediate_size,
            hidden_size,
            axis_names=('parallel', 'embed'),
            **projection_options,
        )
        norm_options = {
            'epsilon': eps,
            'dtype': dtype,
            'axis_names': ('head_dim',),
            'shard_mode': shard_mode,
        }
        if use_qkv_norm:
            self.q_norm = nn.RMSNorm(head_dim, **norm_options)
            self.k_norm = nn.RMSNorm(head_dim, **norm_options)
        else:
            self.q_norm = self.k_norm = None

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        attention_bias: jax.Array | None = None,
        is_causal: bool = False,
        position_idx: jax.Array | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
        kernel: str = 'dot_product',
        **kernel_kwargs: Any,
    ) -> jax.Array:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.pos_emb is not None:
            query, key = self.pos_emb(query, key, position_idx)

        attention = AttentionLegacy.apply(
            query,
            key,
            value,
            kernel=kernel,
            mask=attention_mask,
            bias=attention_bias,
            scale=self.scaling,
            is_causal=is_causal,
            **kernel_kwargs,
        ).reshape(*x.shape[:-1], self.hidden_size)
        mlp = self.activation(self.proj_mlp(x))
        return self.proj_out(
            jnp.concatenate((attention, mlp), axis=-1),
            out_sharding=out_sharding,
        )


class GatedParallelTransformerLayer(nn.Module):
    """A modulated transformer layer with parallel attention and FF paths.

    ``parallel_path`` owns both computations and returns their combined
    projection. A single adaptive normalization produces shift, scale, and
    gate values for that path. Optional context tokens are prepended before
    computation and may be split from the result afterward.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        conditioning_size: int,
        *,
        parallel_path: ModuleSpec,
        head_dim: int | None = None,
        dropout: float = 0.0,
        activation: Activation = 'gelu',
        norm: NormType = 'layernorm',
        norm_eps: float = 1e-6,
        bias: bool = False,
        pos_emb: nn.Module | None = None,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
        project_conditioning: bool = True,
        use_qkv_norm: bool = True,
        scaling: float | None = None,
        input_layernorm: ModuleSpec = AdaXNorm,
    ) -> None:
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.num_heads = _validate_integer(num_heads, 'num_heads')
        self.intermediate_size = _validate_integer(
            intermediate_size,
            'intermediate_size',
        )
        self.conditioning_size = _validate_integer(
            conditioning_size,
            'conditioning_size',
        )
        if head_dim is None:
            if hidden_size % num_heads:
                raise ValueError(
                    'hidden_size must be divisible by num_heads when '
                    'head_dim is not provided'
                )
            head_dim = hidden_size // num_heads
        self.head_dim = _validate_integer(head_dim, 'head_dim')
        if not isinstance(project_conditioning, bool):
            raise TypeError('project_conditioning must be a boolean')
        if norm not in {'layernorm', 'rmsnorm'}:
            raise ValueError("norm must be 'layernorm' or 'rmsnorm'")
        if pos_emb is not None and not isinstance(pos_emb, nn.Module):
            raise TypeError('pos_emb must be an nn.Module or None')

        self.project_conditioning = project_conditioning
        self.dropout = _validate_probability(dropout, 'dropout')
        self.shard_mode = shard_mode
        self.norm = _instantiate_module(
            input_layernorm,
            name='input_layernorm',
            options={
                'embedding_dim': conditioning_size,
                'out_dim': hidden_size * 3,
                'norm': norm,
                'eps': norm_eps,
                'activation': 'silu',
                'bias': bias,
                'project': project_conditioning,
                'dtype': dtype,
                'rngs': rngs,
                'quant': quant,
                'dot_general': dot_general,
                'axis_names': ('conditioning', 'modulation'),
                'shard_mode': shard_mode,
            },
        )
        self.attn = _instantiate_module(
            parallel_path,
            name='parallel_path',
            options={
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'head_dim': self.head_dim,
                'intermediate_size': intermediate_size,
                'activation': activation,
                'pos_emb': pos_emb,
                'eps': norm_eps,
                'bias': bias,
                'use_qkv_norm': use_qkv_norm,
                'scaling': scaling,
                'dtype': dtype,
                'rngs': rngs,
                'shard_mode': shard_mode,
                'quant': quant,
                'dot_general': dot_general,
            },
        )
        self.output_dropout = nn.Dropout(
            self.dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | None,
        conditioning: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        attention_bias: jax.Array | None = None,
        is_causal: bool = False,
        position_idx: jax.Array | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
        modulation_sharding: jax.sharding.Sharding | None = None,
        kernel: str = 'dot_product',
        **kernel_kwargs: Any,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        hidden_states = JointTransformerLayer._validate_stream(
            hidden_states,
            name='hidden_states',
            hidden_size=self.hidden_size,
        )
        if encoder_hidden_states is not None:
            encoder_hidden_states = JointTransformerLayer._validate_stream(
                encoder_hidden_states,
                name='encoder_hidden_states',
                hidden_size=self.hidden_size,
            )
            if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    'hidden and context streams must share a batch size'
                )
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = jnp.concatenate(
                (encoder_hidden_states, hidden_states),
                axis=1,
            )
        elif split_hidden_states and text_seq_len is None:
            raise ValueError(
                'text_seq_len is required when splitting a pre-concatenated '
                'stream'
            )

        conditioning = jnp.asarray(conditioning)
        valid_shapes = {
            (hidden_states.shape[0], self.conditioning_size),
            (
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.conditioning_size,
            ),
        }
        if conditioning.shape not in valid_shapes:
            raise ValueError(
                'conditioning must have shape [batch, conditioning] or '
                '[batch, sequence, conditioning], got '
                f'{conditioning.shape}'
            )
        if not jnp.issubdtype(conditioning.dtype, jnp.floating):
            raise TypeError('conditioning must have a floating-point dtype')

        normalized, modulation = self.norm(
            hidden_states,
            conditioning,
            out_sharding=out_sharding,
            modulation_sharding=modulation_sharding,
        )
        expected_feature = 3 * self.hidden_size
        valid_modulation_shapes = {
            (hidden_states.shape[0], expected_feature),
            (
                hidden_states.shape[0],
                hidden_states.shape[1],
                expected_feature,
            ),
        }
        if modulation.shape not in valid_modulation_shapes:
            raise ValueError(
                f'modulation must end in {expected_feature} features, got '
                f'{modulation.shape}'
            )
        shift, scale, gate = jnp.split(modulation, 3, axis=-1)
        normalized = JointTransformerLayer._modulate(
            normalized,
            shift,
            scale,
        )
        output = self.attn(
            normalized,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            is_causal=is_causal,
            position_idx=position_idx,
            out_sharding=out_sharding,
            kernel=kernel,
            **kernel_kwargs,
        )
        output = self.output_dropout(
            JointTransformerLayer._gate(output, gate),
            out_sharding=out_sharding,
        )
        hidden_states = hidden_states + output
        if hidden_states.dtype == jnp.float16:
            hidden_states = jnp.clip(hidden_states, -65504, 65504)
        hidden_states = _constrain(
            hidden_states,
            out_sharding,
            self.shard_mode,
        )
        if split_hidden_states:
            return (
                hidden_states[:, :text_seq_len],
                hidden_states[:, text_seq_len:],
            )
        return hidden_states

    def extra_repr(self) -> str:
        return (
            f'{self.hidden_size}, heads={self.num_heads}, '
            f'head_dim={self.head_dim}, mlp={self.intermediate_size}'
        )


__all__ = [
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerDecoder',
    'Transformer',
    'ConditionalTransformerLayer',
    'JointTransformerLayer',
    'GatedParallelTransformerLayer',
    'ParallelAttentionMLP',
]
