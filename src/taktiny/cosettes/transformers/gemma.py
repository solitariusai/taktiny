# Copyright 2026 Shinapri
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from jax.nn.initializers import normal

from taktiny import nn
from taktiny.cosettes.transformers.ordinario import (
    TransformerDecoderLayer,
)
from taktiny.maestro.config import ModelConfig
from taktiny.cosettes.layers import Attention, AttentionLegacy, GateMLP, MoEFFN
from taktiny.utils.typing import Axes, AxisNames, DType, Initializer, ShardMode


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹
class GemmaTextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        initializer: Initializer = normal(0.02),
        dtype: DType = 'float32',
        rngs: nn.Rngs | None = None,
        quant: tp.Any = None,
        axis_names: AxisNames | None = None,
        shard_mode: tp.Any = ShardMode.AUTO,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            rngs=rngs,
            dtype=dtype,
            initializer=initializer,
            quant=quant,
            axis_names=axis_names,
            shard_mode=shard_mode,
        )
        self.embedding_scale = embedding_dim ** 0.5

    def __call__(self, indices: jax.Array, out_sharding: tp.Any = None) -> jax.Array:
        x = super().__call__(indices)
        x = x * self.embedding_scale
        if self.shard_mode == ShardMode.EXPLICIT and out_sharding is not None:
            x = jax.lax.with_sharding_constraint(x, out_sharding)

        return x


class GemmaRMSNorm(nn.RMSNorm):
    def __init__(
        self,
        shape: int | tp.Sequence[int] | None,
        epsilon: float = 0.00001,
        *,
        axes: Axes | None = None,
        dtype: DType | None = None,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        super().__init__(
            shape,
            epsilon,
            with_scale=False,
            axes=axes,
            dtype=dtype,
            axis_names=axis_names,
            shard_mode=shard_mode
        )
        self.weight = nn.Parameter(jnp.zeros(shape, dtype=dtype), axis_names=axis_names)

    def __call__(self, x: jax.Array, out_sharding: tp.Any = None) -> jax.Array:
        input_dtype = x.dtype
        value = x.astype(jnp.float32)
        variance = jnp.mean(
            jnp.square(value),
            axis=self.axes,
            keepdims=True,
        )
        x = value * jax.lax.rsqrt(variance + self.eps)
        x = x * (1.0 + self.weight.value.astype(value.dtype))
        x = x.astype(input_dtype)
        if self.shard_mode == ShardMode.EXPLICIT and out_sharding is not None:
            x = jax.lax.with_sharding_constraint(x, out_sharding)

        return x


class GemmaDecoderLayer(TransformerDecoderLayer):
    _norm1 = GemmaRMSNorm
    _norm2 = GemmaRMSNorm

# ┏━╸┏━╸┏┳┓┏┳┓┏━┓   ┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫   ┏━┛
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹   ┗━╸
class Gemma2DecoderLayer(GemmaDecoderLayer):
    def __init__(self, config: ModelConfig, *, rngs: nn.Rngs, layer_idx: int | None = None, **kwargs: tp.Any) -> None:
        query_pre_attn_scalar = config.query_pre_attn_scalar
        _attention_kwargs = {
            'scaling': query_pre_attn_scalar ** -0.5,
            'softcap': config.attn_logit_softcapping,
        }
        self._attention_kwargs = {
            **self._attention_kwargs,
            **_attention_kwargs,
        }
        super().__init__(config, rngs=rngs, layer_idx=layer_idx, **kwargs)

        self.norm3 = GemmaRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            axis_names=config.rmsnorm_weight_axis_names,
            shard_mode=config.shard_mode,
        )
        self.norm4 = GemmaRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            axis_names=config.rmsnorm_weight_axis_names,
            shard_mode=config.shard_mode,
        )

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        is_causal: bool = False,
        kv_cache: tp.Tuple[jax.Array, jax.Array] | None = None,
        cache_position: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        position_embedding: tuple[jax.Array, jax.Array] | None = None,
        boundary_ids: jax.Array | None = None,
        kernel: str = 'dot_product',
        out_sharding: tp.Any = None,
        **kwargs: tp.Any,
    ) -> tp.Tuple[
        jax.Array,
        tp.Tuple[jax.Array, jax.Array] | None,
    ]:
        z = x
        x, updated_cache = self.attention(
            self.norm1(x, out_sharding=out_sharding),
            attention_mask=attention_mask,
            is_causal=is_causal,
            kv_cache=kv_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embedding=position_embedding,
            boundary_ids=boundary_ids,
            use_sliding_window=self.use_sliding_window,
            kernel=kernel,
            out_sharding=out_sharding
        )
        x = self.norm2(x, out_sharding=out_sharding) + z
        z = x
        x = self.ffn(
            self.norm3(x, out_sharding=out_sharding),
            out_sharding=out_sharding
        )
        x = self.norm4(x, out_sharding=out_sharding) + z
        return x, updated_cache

# ┏━╸┏━╸┏┳┓┏┳┓┏━┓   ┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫   ╺━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹   ┗━┛
class Gemma3DecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: ModelConfig, *, rngs: nn.Rngs, layer_idx: int | None = None, **kwargs: tp.Any) -> None:
        head_dim = (
            config.head_dim
            or config.hidden_size // config.num_attention_heads
        )
        q_norm = GemmaRMSNorm(
            head_dim,
            config.rms_norm_eps,
            axis_names=config.attention_q_proj_axis_names[-1:],
            shard_mode=config.shard_mode,
        )
        k_norm = GemmaRMSNorm(
            head_dim,
            config.rms_norm_eps,
            axis_names=config.attention_k_proj_axis_names[-1:],
            shard_mode=config.shard_mode,
        )
        _attention_kwargs = {
            'q_norm': q_norm,
            'k_norm': k_norm,
        }
        self._attention_kwargs = {
            **self._attention_kwargs,
            **_attention_kwargs,
        }
        super().__init__(config, rngs=rngs, layer_idx=layer_idx, **kwargs)
        del self._attention_kwargs
__all__ = [
    'GemmaTextScaledWordEmbedding',
    'GemmaRMSNorm',
    'GemmaDecoderLayer',
    'Gemma2DecoderLayer',
    'Gemma3DecoderLayer',
]
