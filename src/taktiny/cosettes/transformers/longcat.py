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
"""LongCat image and audio transformer layers."""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from taktiny import layers as ly
from taktiny import nn
from taktiny.cosettes._continuo import (
    _config_value,
    _approximate_gelu,
    _model_dtype,
    _shard_mode,
    combine_joint_positions,
    image_transformer_dimensions,
    multi_axis_position_embedding,
)
from taktiny.cosettes.transformers._ordinario import (
    ConditionalTransformerLayer,
    GatedParallelTransformerLayer,
    JointTransformerLayer,
)
from taktiny.maestro.config import ModelConfig
from taktiny.nn._continuo import _constrain


def _audio_dimensions(
    config: ModelConfig,
) -> tuple[int, int, int, int]:
    hidden_size = _config_value(
        config,
        'dit_dim',
        'hidden_size',
        'inner_dim',
        'dim',
    )
    num_heads = _config_value(
        config,
        'dit_heads',
        'num_attention_heads',
    )
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError('config must define a positive dit_dim or hidden_size')
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError(
            'config must define a positive dit_heads or num_attention_heads'
        )

    head_dim = _config_value(
        config,
        'head_dim',
        'attention_head_dim',
    )
    if head_dim is None:
        if hidden_size % num_heads:
            raise ValueError('LongCat Audio hidden size must divide by its heads')
        head_dim = hidden_size // num_heads
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError('head_dim must be a positive integer')
    if hidden_size != num_heads * head_dim:
        raise ValueError('hidden size must equal num_heads * head_dim')

    intermediate_size = _config_value(config, 'intermediate_size')
    if intermediate_size is None:
        intermediate_size = int(
            hidden_size * _config_value(config, 'ff_mult', default=4.0)
        )
    if not isinstance(intermediate_size, int) or intermediate_size <= 0:
        raise ValueError('intermediate_size must be a positive integer')
    return hidden_size, num_heads, head_dim, intermediate_size


def _audio_layer_config(
    config: ModelConfig,
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> ModelConfig:
    values = dict(vars(config))
    values.update(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        context_dim=hidden_size,
        norm_eps=_config_value(config, 'eps', 'norm_eps', default=1e-6),
        norm_elementwise_affine=False,
    )
    if bool(_config_value(config, 'qk_norm', default=True)):
        values['qk_norm'] = 'rms_norm_across_heads'
    else:
        values['qk_norm'] = None
        values['use_qkv_norm'] = False
    return ModelConfig(**values)


def _key_mask(mask: jax.Array | None) -> jax.Array | None:
    if mask is None:
        return None
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if mask.ndim == 2:
        return mask[:, None, None, :]
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim == 4:
        return mask
    raise ValueError('attention masks must have rank 2, 3, or 4')


def _mask_attention_output(
    value: jax.Array,
    mask: jax.Array | None,
) -> jax.Array:
    if mask is None:
        return value
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if mask.ndim != 2 or mask.shape != value.shape[:2]:
        return value
    return value * mask[..., None].astype(value.dtype)


class _LongCatAudioAdaLNProjection(nn.Module):
    """The zero-initialized local AudioDiT AdaLN projection."""

    def __init__(
        self,
        hidden_size: int,
        *,
        bias: bool,
        dtype: tp.Any,
        rngs: nn.Rngs,
        shard_mode: tp.Any,
        quant: tp.Any,
        dot_general: tp.Any,
    ) -> None:
        self.linear = nn.Linear(
            hidden_size,
            6 * hidden_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            initializer=jax.nn.initializers.zeros,
            quant=quant,
            dot_general=dot_general,
            axis_names=('embed', 'modulation_embed'),
            shard_mode=shard_mode,
        )

    def __call__(self, conditioning: jax.Array) -> jax.Array:
        return self.linear(jax.nn.silu(conditioning))


class LongCatImageTransformerLayer(JointTransformerLayer):
    """LongCat Image dual-stream transformer layer."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        hidden_size, _, _, _ = image_transformer_dimensions(config)
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            conditioning_size=hidden_size,
            context_pre_only=False,
            dual_attention=False,
            project_conditioning=True,
            context_project_conditioning=True,
            use_qkv_norm=True,
            qkv_norm_eps=1e-6,
            context_first=True,
            bias=True,
            pos_emb=multi_axis_position_embedding(config),
            activation=_approximate_gelu,
            input_layernorm=ly.AdaXNorm,
            context_input_layernorm=ly.AdaXNorm,
            joint_attention=ly.JointAttention,
            post_attention_layernorm=nn.LayerNorm,
            context_post_attention_layernorm=nn.LayerNorm,
            mlp=ly.FeedForward,
            context_mlp=ly.FeedForward,
        )

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array,
        conditioning: jax.Array,
        *,
        position_idx: jax.Array | None = None,
        encoder_position_idx: jax.Array | None = None,
        **attention_kwargs: tp.Any,
    ) -> tuple[jax.Array, jax.Array]:
        joint_positions = combine_joint_positions(
            encoder_position_idx,
            position_idx,
            batch_size=x.shape[0],
        )
        enc_x, x = super().__call__(
            x,
            enc_x,
            conditioning,
            context_conditioning=conditioning,
            position_idx=joint_positions,
            **attention_kwargs,
        )
        return enc_x, x


class LongCatImageSingleTransformerLayer(GatedParallelTransformerLayer):
    """LongCat Image text-image gated parallel transformer layer."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        hidden_size, _, _, _ = image_transformer_dimensions(config)
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            conditioning_size=hidden_size,
            project_conditioning=True,
            pos_emb=multi_axis_position_embedding(config),
            activation=_approximate_gelu,
            use_qkv_norm=True,
            bias=True,
            input_layernorm=ly.AdaXNorm,
            parallel_path=ly.ParallelAttentionMLP,
        )

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array,
        conditioning: jax.Array,
        *,
        position_idx: jax.Array | None = None,
        encoder_position_idx: jax.Array | None = None,
        **attention_kwargs: tp.Any,
    ) -> tuple[jax.Array, jax.Array]:
        joint_positions = combine_joint_positions(
            encoder_position_idx,
            position_idx,
            batch_size=x.shape[0],
        )
        enc_x, x = super().__call__(
            x,
            enc_x,
            conditioning,
            position_idx=joint_positions,
            split_hidden_states=True,
            **attention_kwargs,
        )
        return enc_x, x


class LongCatAudioTransformerLayer(ConditionalTransformerLayer):
    """LongCat AudioDiT sequential self/cross-attention transformer layer.

    For ``adaln_type='local'``, ``conditioning`` is the timestep embedding and
    this layer projects it into six modulation values. For
    ``adaln_type='global'``, ``conditioning`` is the already projected global
    six-way modulation shared by every layer; the layer adds its learned local
    offset. Text conditioning remains read-only in both modes.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        (
            hidden_size,
            num_heads,
            head_dim,
            intermediate_size,
        ) = _audio_dimensions(config)
        audio_config = _audio_layer_config(
            config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
        )
        self.adaln_type = _config_value(
            config,
            'adaln_type',
            default='global',
        )
        if self.adaln_type not in {'global', 'local'}:
            raise ValueError("adaln_type must be 'global' or 'local'")
        self.adaln_use_text_cond = bool(
            _config_value(config, 'adaln_use_text_cond', default=True)
        )
        self.use_cross_attention = bool(
            _config_value(config, 'cross_attn', default=True)
        )
        self.layer_idx = layer_idx
        self.audio_position_embedding = ly.RotaryEmbedding(
            head_dim,
            max_position_embeddings=_config_value(
                config,
                'max_position_embeddings',
                default=2048,
            ),
            base=_config_value(config, 'rope_theta', default=100_000.0),
        )
        self.prompt_position_embedding = ly.RotaryEmbedding(
            head_dim,
            max_position_embeddings=_config_value(
                config,
                'max_position_embeddings',
                default=2048,
            ),
            base=_config_value(config, 'rope_theta', default=100_000.0),
        )

        bias = bool(_config_value(config, 'bias', default=True))
        dropout = float(_config_value(config, 'dropout', default=0.0))
        eps = float(_config_value(config, 'eps', 'norm_eps', default=1e-6))
        dtype = _model_dtype(config)
        shard_mode = _shard_mode(config)
        quant = _config_value(config, 'quant')
        dot_general = _config_value(config, 'dot_general')
        cross_norm = bool(
            _config_value(config, 'cross_attn_norm', default=False)
        )
        cross_input_norm = (
            nn.LayerNorm(
                hidden_size,
                eps=eps,
                elementwise_affine=True,
                dtype=dtype,
                bias=True,
                axis_names=('embed',),
                shard_mode=shard_mode,
            )
            if cross_norm
            else None
        )

        super().__init__(
            audio_config,
            rngs=rngs,
            layer_idx=layer_idx,
            activation=_approximate_gelu,
            ffn_dropout=dropout,
            attention_bias=bias,
            attention_out_bias=bias,
            cross_attention_bias=bias,
            mlp_bias=bias,
            pos_emb=self.audio_position_embedding,
            cross_pos_emb=None,
            input_layernorm=nn.LayerNorm,
            self_attention=ly.Attention,
            cross_attention=(ly.Attention if self.use_cross_attention else None),
            cross_attention_layernorm=cross_input_norm,
            post_attention_layernorm=nn.LayerNorm,
            mlp=ly.FeedForward,
        )
        self.context_cross_attention_norm = (
            nn.LayerNorm(
                hidden_size,
                eps=eps,
                elementwise_affine=True,
                dtype=dtype,
                bias=True,
                axis_names=('context_embed',),
                shard_mode=shard_mode,
            )
            if cross_norm
            else None
        )
        if self.adaln_type == 'local':
            del self.scale_shift_table
            self.adaln_mlp = _LongCatAudioAdaLNProjection(
                hidden_size,
                bias=True,
                dtype=dtype,
                rngs=rngs,
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
            )

    def _audio_modulation(
        self,
        conditioning: jax.Array,
        encoder_hidden_states: jax.Array | None,
        encoder_attention_mask: jax.Array | None,
    ) -> tuple[jax.Array, ...]:
        conditioning = jnp.asarray(conditioning)
        batch_size = conditioning.shape[0]
        if self.adaln_type == 'local':
            if conditioning.shape != (batch_size, self.hidden_size):
                raise ValueError(
                    'local AudioDiT conditioning must have shape '
                    '[batch, hidden_size]'
                )
            if self.adaln_use_text_cond:
                if encoder_hidden_states is None:
                    raise ValueError(
                        'local text-conditioned AdaLN requires encoder states'
                    )
                if encoder_attention_mask is None:
                    text_mean = jnp.mean(encoder_hidden_states, axis=1)
                else:
                    mask = jnp.asarray(
                        encoder_attention_mask,
                        dtype=encoder_hidden_states.dtype,
                    )
                    if mask.shape != encoder_hidden_states.shape[:2]:
                        raise ValueError(
                            'encoder_attention_mask must have shape '
                            '[batch, context_sequence]'
                        )
                    denominator = jnp.maximum(mask.sum(axis=1, keepdims=True), 1)
                    text_mean = (
                        encoder_hidden_states * mask[..., None]
                    ).sum(axis=1) / denominator
                conditioning = conditioning + text_mean
            modulation = self.adaln_mlp(conditioning)
        else:
            expected_flat = (batch_size, 6 * self.hidden_size)
            expected_grouped = (batch_size, 6, self.hidden_size)
            if conditioning.shape == expected_flat:
                modulation = conditioning.reshape(expected_grouped)
            elif conditioning.shape == expected_grouped:
                modulation = conditioning
            else:
                raise ValueError(
                    'global AudioDiT conditioning must have shape '
                    f'{expected_flat} or {expected_grouped}'
                )
            modulation = modulation + self.scale_shift_table.value[None]

        modulation = modulation.reshape(batch_size, 6, self.hidden_size)
        return tuple(jnp.take(modulation, index, axis=1) for index in range(6))

    def _cross_attention(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        *,
        attention_mask: jax.Array | None,
        position_idx: jax.Array | None,
        encoder_position_idx: jax.Array | None,
        out_sharding: jax.sharding.Sharding | None,
        kernel: str,
    ) -> jax.Array:
        if self.attn2 is None:
            raise RuntimeError('cross-attention is disabled')
        query = self.attn2.q_proj(hidden_states)
        key = self.attn2.k_proj(encoder_hidden_states)
        value = self.attn2.v_proj(encoder_hidden_states)
        if self.attn2.q_norm is not None:
            query = self.attn2.q_norm(query)
        if self.attn2.k_norm is not None:
            key = self.attn2.k_norm(key)

        query, _ = self.audio_position_embedding(
            query,
            query,
            position_idx,
        )
        key, _ = self.prompt_position_embedding(
            key,
            key,
            encoder_position_idx,
        )
        output = ly.Attention.apply(
            query,
            key,
            value,
            kernel=kernel,
            mask=_key_mask(attention_mask),
            scale=self.attn2.scaling,
            is_causal=False,
        )
        return self.attn2.o_proj(output, out_sharding=out_sharding)

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array | None,
        conditioning: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        encoder_attention_mask: jax.Array | None = None,
        position_idx: jax.Array | None = None,
        encoder_position_idx: jax.Array | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
        self_attention_kernel: str = 'dot_product',
        cross_attention_kernel: str = 'dot_product',
    ) -> jax.Array:
        x = jnp.asarray(x)
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                'x must have shape [batch, sequence, hidden_size]'
            )
        if enc_x is not None:
            enc_x = jnp.asarray(enc_x)
            if (
                enc_x.ndim != 3
                or enc_x.shape[0] != x.shape[0]
                or enc_x.shape[-1] != self.context_size
            ):
                raise ValueError(
                    'enc_x must have shape [batch, sequence, context_size]'
                )

        (
            gate_attention,
            scale_attention,
            shift_attention,
            gate_feed,
            scale_feed,
            shift_feed,
        ) = self._audio_modulation(
            conditioning,
            enc_x,
            encoder_attention_mask,
        )
        gate_attention = gate_attention.astype(x.dtype)
        scale_attention = scale_attention.astype(x.dtype)
        shift_attention = shift_attention.astype(x.dtype)
        gate_feed = gate_feed.astype(x.dtype)
        scale_feed = scale_feed.astype(x.dtype)
        shift_feed = shift_feed.astype(x.dtype)

        normalized = self._modulate(
            self.norm1(x, out_sharding=out_sharding),
            shift_attention,
            scale_attention,
        )
        attention = self.attn1(
            normalized,
            attention_mask=_key_mask(attention_mask),
            is_causal=False,
            position_idx=position_idx,
            out_sharding=out_sharding,
            kernel=self_attention_kernel,
        )[0]
        attention = _mask_attention_output(attention, attention_mask)
        x = x + self._gate(
            self.self_attention_dropout(attention),
            gate_attention,
        )

        if self.use_cross_attention:
            if enc_x is None:
                raise ValueError('enc_x is required when cross-attention is enabled')
            cross_input = (
                x
                if self.norm_cross is None
                else self.norm_cross(x, out_sharding=out_sharding)
            )
            context = (
                enc_x
                if self.context_cross_attention_norm is None
                else self.context_cross_attention_norm(enc_x)
            )
            cross_attention = self._cross_attention(
                cross_input,
                context,
                attention_mask=encoder_attention_mask,
                position_idx=position_idx,
                encoder_position_idx=encoder_position_idx,
                out_sharding=out_sharding,
                kernel=cross_attention_kernel,
            )
            cross_attention = _mask_attention_output(
                cross_attention,
                attention_mask,
            )
            x = x + self.cross_attention_dropout(cross_attention)

        normalized = self._modulate(
            self.norm2(x, out_sharding=out_sharding),
            shift_feed,
            scale_feed,
        )
        feed = self.ff(normalized, out_sharding=out_sharding)
        x = x + self._gate(feed, gate_feed)
        return _constrain(x, out_sharding, self.shard_mode)


__all__ = [
    'LongCatAudioTransformerLayer',
    'LongCatImageSingleTransformerLayer',
    'LongCatImageTransformerLayer',
]
