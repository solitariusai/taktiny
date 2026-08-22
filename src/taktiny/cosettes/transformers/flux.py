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
"""FLUX.2 transformer architecture components."""

from __future__ import annotations

from collections.abc import Sequence
import typing as tp

import jax
import jax.numpy as jnp

from taktiny.cosettes import layers as ly
from taktiny import nn
from taktiny.cosettes.continuo import (
    _config_value,
    combine_joint_positions as _combine_positions,
)
from taktiny.cosettes.transformers.ordinario import (
    GatedParallelTransformerLayer,
    JointTransformerLayer,
)
from taktiny.maestro.config import ModelConfig
from taktiny.utils.typing import (
    Activation,
    AxisNames,
    DType,
    ShardMode,
    Sharding,
)


def _flux2_dimensions(config: ModelConfig) -> tuple[int, int, int, int]:
    num_heads = _config_value(config, 'num_attention_heads')
    head_dim = _config_value(config, 'attention_head_dim', 'head_dim')
    hidden_size = _config_value(config, 'hidden_size', 'inner_dim', 'dim')
    if hidden_size is None and num_heads is not None and head_dim is not None:
        hidden_size = num_heads * head_dim

    dimensions = {
        'num_attention_heads': num_heads,
        'attention_head_dim': head_dim,
        'hidden_size': hidden_size,
    }
    missing = [name for name, value in dimensions.items() if value is None]
    if missing:
        raise ValueError(
            'Missing required FLUX.2 config values: ' + ', '.join(missing)
        )
    if not all(
        isinstance(value, int) and value > 0
        for value in dimensions.values()
    ):
        raise ValueError('FLUX.2 dimensions must be positive integers')
    if hidden_size != num_heads * head_dim:
        raise ValueError(
            'hidden_size must equal num_attention_heads * attention_head_dim'
        )

    intermediate_size = _config_value(config, 'intermediate_size')
    if intermediate_size is None:
        intermediate_size = int(
            hidden_size * _config_value(config, 'mlp_ratio', default=3.0)
        )
    if not isinstance(intermediate_size, int) or intermediate_size <= 0:
        raise ValueError('intermediate_size must be a positive integer')
    return hidden_size, num_heads, head_dim, intermediate_size


def _flux2_position_embedding(config: ModelConfig) -> nn.Module | None:
    configured = _config_value(config, 'pos_emb', 'position_embedding')
    if configured is not None:
        if not isinstance(configured, nn.Module):
            raise TypeError('configured FLUX.2 position embedding must be a Module')
        return configured
    axes_dim = _config_value(config, 'axes_dims_rope')
    if axes_dim is None:
        return None
    return Flux2RotaryEmbedding(
        theta=_config_value(config, 'rope_theta', default=2000.0),
        axes_dim=axes_dim,
    )


class Flux2RotaryEmbedding(ly.MultiAxisRotaryEmbedding):
    """Apply FLUX.2 multi-axis rotary embeddings to projected Q and K."""

    def __init__(
        self,
        theta: float = 2000.0,
        axes_dim: Sequence[int] = (32, 32, 32, 32),
    ) -> None:
        super().__init__(axes_dim, theta=theta)


class Flux2Modulation(nn.Module):
    """Create shared FLUX.2 shift, scale, and gate parameters."""

    def __init__(
        self,
        hidden_size: int,
        modulation_sets: int = 2,
        *,
        bias: bool = False,
        dtype: DType | None = None,
        rngs: nn.Rngs,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: tp.Any = None,
        dot_general: tp.Any = None,
    ) -> None:
        if not isinstance(modulation_sets, int) or modulation_sets <= 0:
            raise ValueError('modulation_sets must be a positive integer')
        self.hidden_size = hidden_size
        self.modulation_sets = modulation_sets
        self.linear = nn.Linear(
            hidden_size,
            hidden_size * 3 * modulation_sets,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=axis_names,
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )

    def __call__(
        self,
        conditioning: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        return self.linear(
            jax.nn.silu(conditioning),
            out_sharding=out_sharding,
        )

    def split(
        self,
        modulation: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], ...]:
        modulation = jnp.asarray(modulation)
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        expected = self.hidden_size * 3 * self.modulation_sets
        if modulation.ndim != 3 or modulation.shape[-1] != expected:
            raise ValueError(
                f'modulation must end in {expected} features, got '
                f'{modulation.shape}'
            )
        chunks = jnp.split(modulation, 3 * self.modulation_sets, axis=-1)
        return tuple(
            tuple(chunks[3 * index:3 * (index + 1)])
            for index in range(self.modulation_sets)
        )


class Flux2FeedForward(nn.Module):
    """FLUX.2 SwiGLU FFN with its gate and value projections fused."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Activation | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        dtype: DType | None = None,
        rngs: nn.Rngs,
        input_axis_names: AxisNames | None = None,
        output_axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: tp.Any = None,
        dot_general: tp.Any = None,
    ) -> None:
        del activation
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        options = {
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'shard_mode': shard_mode,
            'quant': quant,
            'dot_general': dot_general,
        }
        self.linear_in = nn.Linear(
            hidden_size,
            intermediate_size * 2,
            axis_names=input_axis_names,
            **options,
        )
        self.dropout = nn.Dropout(
            dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.linear_out = nn.Linear(
            intermediate_size,
            hidden_size,
            axis_names=output_axis_names,
            **options,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        gate, value = jnp.split(self.linear_in(x), 2, axis=-1)
        x = self.dropout(jax.nn.silu(gate) * value)
        return self.linear_out(x, out_sharding=out_sharding)


class Flux2JointAttention(ly.JointAttention):
    """Joint attention with FLUX.2's text-then-image token order."""

    def __call__(
        self,
        x1: jax.Array,
        x2: jax.Array,
        attention_mask: jax.Array | None = None,
        attention_bias: jax.Array | None = None,
        is_causal: bool = False,
        position_idx: jax.Array | None = None,
        out_shardings: tuple[Sharding, Sharding] | None = None,
        kernel: str = 'dot_product',
        **kernel_kwargs: tp.Any,
    ) -> tuple[jax.Array, jax.Array]:
        x1, x2 = self._validate_inputs(x1, x2)
        context_length = x2.shape[1]

        q1, k1, v1 = self.q_proj_1(x1), self.k_proj_1(x1), self.v_proj_1(x1)
        q2, k2, v2 = self.q_proj_2(x2), self.k_proj_2(x2), self.v_proj_2(x2)
        if self.q_norm_1 is not None:
            q1, k1 = self.q_norm_1(q1), self.k_norm_1(k1)
            q2, k2 = self.q_norm_2(q2), self.k_norm_2(k2)

        query = jnp.concatenate((q2, q1), axis=1)
        key = jnp.concatenate((k2, k1), axis=1)
        value = jnp.concatenate((v2, v1), axis=1)
        if self.pos_emb is not None:
            query, key = self.pos_emb(query, key, position_idx)
        output = ly.AttentionLegacy.apply(
            query,
            key,
            value,
            kernel=kernel,
            mask=attention_mask,
            bias=attention_bias,
            scale=self.scaling,
            is_causal=is_causal,
            **kernel_kwargs,
        )
        context_output, image_output = jnp.split(
            output,
            (context_length,),
            axis=1,
        )

        if out_shardings is None:
            image_sharding = context_sharding = None
        elif len(out_shardings) == 2:
            image_sharding, context_sharding = out_shardings
        else:
            raise ValueError('out_shardings must contain exactly two values')
        return (
            self.o_proj_1(image_output, out_sharding=image_sharding),
            self.o_proj_2(context_output, out_sharding=context_sharding),
        )


class Flux2TransformerLayer(JointTransformerLayer):
    """FLUX.2 double-stream joint-attention transformer layer."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        hidden_size, _, _, _ = _flux2_dimensions(config)
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            conditioning_size=hidden_size * 6,
            context_pre_only=False,
            dual_attention=False,
            project_conditioning=False,
            context_project_conditioning=False,
            use_qkv_norm=True,
            qkv_norm_eps=_config_value(config, 'eps', default=1e-6),
            pos_emb=_flux2_position_embedding(config),
            input_layernorm=ly.AdaXNorm,
            context_input_layernorm=ly.AdaXNorm,
            joint_attention=Flux2JointAttention,
            post_attention_layernorm=nn.LayerNorm,
            context_post_attention_layernorm=nn.LayerNorm,
            mlp=Flux2FeedForward,
            context_mlp=Flux2FeedForward,
        )

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array,
        image_modulation: jax.Array,
        text_modulation: jax.Array,
        *,
        position_idx: jax.Array | None = None,
        encoder_position_idx: jax.Array | None = None,
        **attention_kwargs: tp.Any,
    ) -> tuple[jax.Array, jax.Array]:
        joint_positions = _combine_positions(
            encoder_position_idx,
            position_idx,
            batch_size=x.shape[0],
        )
        enc_x, x = super().__call__(
            x,
            enc_x,
            image_modulation,
            context_conditioning=text_modulation,
            position_idx=joint_positions,
            **attention_kwargs,
        )
        return enc_x, x


class Flux2ParallelSelfAttention(nn.Module):
    """Fused parallel attention and SwiGLU path used by FLUX.2."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        *,
        pos_emb: nn.Module | None = None,
        eps: float = 1e-6,
        bias: bool = False,
        dtype: DType | None = None,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: tp.Any = None,
        dot_general: tp.Any = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.pos_emb = pos_emb
        options = {
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'shard_mode': shard_mode,
            'quant': quant,
            'dot_general': dot_general,
        }
        self.to_qkv_mlp_proj = nn.Linear(
            hidden_size,
            3 * hidden_size + 2 * intermediate_size,
            axis_names=('embed', 'parallel'),
            **options,
        )
        norm_options = {
            'epsilon': eps,
            'dtype': dtype,
            'axis_names': ('head_dim',),
            'shard_mode': shard_mode,
        }
        self.norm_q = nn.RMSNorm(head_dim, **norm_options)
        self.norm_k = nn.RMSNorm(head_dim, **norm_options)
        self.to_out = nn.Linear(
            hidden_size + intermediate_size,
            hidden_size,
            axis_names=('parallel_out', 'embed'),
            **options,
        )

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
        **kernel_kwargs: tp.Any,
    ) -> jax.Array:
        projected = self.to_qkv_mlp_proj(x)
        qkv, mlp = jnp.split(projected, (3 * self.hidden_size,), axis=-1)
        query, key, value = jnp.split(qkv, 3, axis=-1)
        target_shape = (*x.shape[:-1], self.num_heads, self.head_dim)
        query = self.norm_q(query.reshape(target_shape))
        key = self.norm_k(key.reshape(target_shape))
        value = value.reshape(target_shape)
        if self.pos_emb is not None:
            query, key = self.pos_emb(query, key, position_idx)

        attention = ly.AttentionLegacy.apply(
            query,
            key,
            value,
            kernel=kernel,
            mask=attention_mask,
            bias=attention_bias,
            is_causal=is_causal,
            **kernel_kwargs,
        ).reshape(*x.shape[:-1], self.hidden_size)
        gate, value = jnp.split(mlp, 2, axis=-1)
        mlp = jax.nn.silu(gate) * value
        return self.to_out(
            jnp.concatenate((attention, mlp), axis=-1),
            out_sharding=out_sharding,
        )


class Flux2SingleTransformerLayer(GatedParallelTransformerLayer):
    """FLUX.2 concatenated-stream parallel transformer layer."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        hidden_size, _, _, _ = _flux2_dimensions(config)
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            conditioning_size=hidden_size * 3,
            project_conditioning=False,
            pos_emb=_flux2_position_embedding(config),
            input_layernorm=ly.AdaXNorm,
            parallel_path=Flux2ParallelSelfAttention,
        )

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array | None,
        modulation: jax.Array,
        *,
        position_idx: jax.Array | None = None,
        encoder_position_idx: jax.Array | None = None,
        **attention_kwargs: tp.Any,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        if enc_x is not None:
            position_idx = _combine_positions(
                encoder_position_idx,
                position_idx,
                batch_size=x.shape[0],
            )
        return super().__call__(
            x,
            enc_x,
            modulation,
            position_idx=position_idx,
            **attention_kwargs,
        )


__all__ = [
    'Flux2FeedForward',
    'Flux2JointAttention',
    'Flux2Modulation',
    'Flux2ParallelSelfAttention',
    'Flux2RotaryEmbedding',
    'Flux2SingleTransformerLayer',
    'Flux2TransformerLayer',
]
