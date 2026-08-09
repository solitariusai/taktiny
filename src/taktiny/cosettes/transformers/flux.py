from __future__ import annotations

from typing import Any

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
# DEPRECATED / LEGACY
# TODO(refactor).
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from taktiny import nn
from taktiny.layers.attention import JointAttention, Attention
from taktiny.layers.ffn import FusedGateMLP
from taktiny.layers.positional_embedding import (
    SinusoidalPositionalEmbedding,
    rotate_half,
)


def get_1d_rotary_pos_embed(dim: int, pos: jax.Array, theta: int = 10000) -> tuple[Any, ...]:
    half_dim = dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    freqs = jnp.einsum("i,j->ij", pos, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)

class Flux2PosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: tuple[int, ...]) -> None:
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: jax.Array) -> tuple[jax.Array, jax.Array]:
        cos_out = []
        sin_out = []
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                ids[..., i].astype(jnp.float32),
                theta=self.theta,
            )
            cos_out.append(jnp.repeat(cos, 2, axis=-1))
            sin_out.append(jnp.repeat(sin, 2, axis=-1))

        freqs_cos = jnp.concatenate(cos_out, axis=-1)
        freqs_sin = jnp.concatenate(sin_out, axis=-1)
        return freqs_cos, freqs_sin

class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False, seed: nn.Rngs | None = None) -> None:
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias, seed=seed)

    def __call__(self, temb: jax.Array) -> tuple[tuple[jax.Array, jax.Array, jax.Array], ...]:
        mod = jax.nn.silu(temb)
        mod = self.linear(mod)

        # Split into mod_param_sets
        # Each set has 3 chunks: shift, scale, gate
        chunks = tuple(jnp.split(mod, self.mod_param_sets * 3, axis=-1))
        return tuple(chunks[i * 3 : (i + 1) * 3] for i in range(self.mod_param_sets))

class JointAttentionBlock(nn.Module):
    def __init__(self, config: Any, seed: nn.Rngs | None = None) -> None:
        hidden_size = config.num_attention_heads * config.attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=config.eps)
        self.norm1_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=config.eps)

        self.attn = JointAttention(
            hidden_size1=hidden_size,
            hidden_size2=hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
            use_qkv_norm=True,
            rngs=seed,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=config.eps)
        self.img_mlp = FusedGateMLP(
            hidden_size=hidden_size,
            intermediate_size=int(hidden_size * config.mlp_ratio),
            activation=jax.nn.silu,
            bias=False,
            seed=seed
        )

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=config.eps)
        self.txt_mlp = FusedGateMLP(
            hidden_size=hidden_size,
            intermediate_size=int(hidden_size * config.mlp_ratio),
            activation=jax.nn.silu,
            bias=False,
            seed=seed
        )

    def __call__(self, img: jax.Array, txt: jax.Array, mod_img: Any, mod_txt: Any, rope_cos_sin: Any) -> tuple[jax.Array, jax.Array]:
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = mod_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = mod_txt

        img_norm = self.norm1(img)
        img_norm = img_norm * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        txt_norm = self.norm1_context(txt)
        txt_norm = txt_norm * (1 + c_scale_msa[:, None, :]) + c_shift_msa[:, None, :]

        cos, sin = rope_cos_sin

        def apply_rope(q: Any, k: Any, position_idx: int | None=None) -> tuple[Any, ...]:
            q_embed = (q * cos[:, None, None, :]) + (rotate_half(q) * sin[:, None, None, :])
            k_embed = (k * cos[:, None, None, :]) + (rotate_half(k) * sin[:, None, None, :])
            return q_embed, k_embed

        self.attn.pos_emb = apply_rope

        img_attn, txt_attn = self.attn(img_norm, txt_norm)
        img = img + gate_msa[:, None, :] * img_attn
        txt = txt + c_gate_msa[:, None, :] * txt_attn

        img_norm2 = self.norm2(img)
        img_norm2 = img_norm2 * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        img = img + gate_mlp[:, None, :] * self.img_mlp(img_norm2)

        txt_norm2 = self.norm2_context(txt)
        txt_norm2 = txt_norm2 * (1 + c_scale_mlp[:, None, :]) + c_shift_mlp[:, None, :]
        txt = txt + c_gate_mlp[:, None, :] * self.txt_mlp(txt_norm2)

        return img, txt

class SingleStreamBlock(nn.Module):
    def __init__(self, config: Any, seed: nn.Rngs | None = None) -> None:
        hidden_size = config.num_attention_heads * config.attention_head_dim
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=config.eps)

        self.attn = Attention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
            use_qkv_norm=True,
            bias=False,
            seed=seed
        )

        self.mlp = FusedGateMLP(
            hidden_size=hidden_size,
            intermediate_size=int(hidden_size * config.mlp_ratio),
            activation=jax.nn.silu,
            bias=False,
            seed=seed
        )

    def __call__(self, x: jax.Array, mod: Any, rope_cos_sin: Any) -> jax.Array:
        # SingleStreamBlock uses only 1 set of modulation parameters
        shift_msa, scale_msa, gate_msa = mod[0]

        x_norm = self.norm(x)
        x_norm = x_norm * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        cos, sin = rope_cos_sin

        def apply_rope(q: Any, k: Any, position_idx: int | None=None) -> tuple[Any, ...]:
            q_embed = (q * cos[:, None, None, :]) + (rotate_half(q) * sin[:, None, None, :])
            k_embed = (k * cos[:, None, None, :]) + (rotate_half(k) * sin[:, None, None, :])
            return q_embed, k_embed

        self.attn.pos_emb = apply_rope

        attn_out, _ = self.attn(x_norm)
        mlp_out = self.mlp(x_norm)

        # Parallel residual: gate applies to both attention and MLP output sum
        x = x + gate_msa[:, None, :] * (attn_out + mlp_out)
        return x

class Flux2Transformer2DModel(nn.Module):
    def __init__(self, config: Any, seed: nn.Rngs | None = None) -> None:
        self.config = config
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        self.out_channels = config.out_channels or config.in_channels

        # Embeddings
        self.pos_embed = Flux2PosEmbed(theta=config.rope_theta, axes_dim=config.axes_dims_rope)

        self.time_in = nn.Sequential(
            [
                SinusoidalPositionalEmbedding(config.timestep_guidance_channels),
                nn.Linear(config.timestep_guidance_channels, self.inner_dim, bias=False, seed=seed),
            ]
        )

        if config.guidance_embeds:
            self.guidance_in = nn.Sequential(
                [
                    SinusoidalPositionalEmbedding(config.timestep_guidance_channels),
                    nn.Linear(config.timestep_guidance_channels, self.inner_dim, bias=False, seed=seed),
                ]
            )
        else:
            self.guidance_in = None

        self.x_embedder = nn.Linear(config.in_channels, self.inner_dim, bias=False, seed=seed)
        self.context_embedder = nn.Linear(config.joint_attention_dim, self.inner_dim, bias=False, seed=seed)

        # Shared Modulations
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False, seed=seed)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False, seed=seed)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False, seed=seed)

        # Blocks
        self.transformer_blocks = nn.SeqStack(
            JointAttentionBlock(config, seed=seed)
            for _ in range(config.num_layers)
        )
        self.single_transformer_blocks = nn.SeqStack(
            SingleStreamBlock(config, seed=seed)
            for _ in range(config.num_single_layers)
        )

        # Output
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=config.eps)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True, seed=seed)

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        timestep: jax.Array,
        guidance: jax.Array | None = None,
        txt_ids: jax.Array | None = None,
        img_ids: jax.Array | None = None
    ) -> jax.Array:

        img = self.x_embedder(hidden_states)
        txt = self.context_embedder(encoder_hidden_states)

        # Timestep / Guidance
        vec = self.time_in(timestep)
        if self.guidance_in is not None and guidance is not None:
            vec = vec + self.guidance_in(guidance)

        # Shared Modulations
        mod_img = self.double_stream_modulation_img(vec)
        mod_txt = self.double_stream_modulation_txt(vec)
        mod_single = self.single_stream_modulation(vec)

        # RoPE
        ids = jnp.concatenate([txt_ids, img_ids], axis=1)
        rope_cos_sin = self.pos_embed(ids)

        # Joint Blocks
        def apply_joint(layer: Any, carry: Any, args: Any) -> tuple[Any, ...]:
            img, txt = carry
            out_img, out_txt = layer(img, txt, mod_img, mod_txt, rope_cos_sin)
            return (out_img, out_txt), None

        (img, txt), _ = self.transformer_blocks(apply_joint, (img, txt), None)

        # Single Blocks
        x = jnp.concatenate([txt, img], axis=1)

        def apply_single(layer: Any, carry: Any, args: Any) -> tuple[Any, ...]:
            return layer(carry, mod_single, rope_cos_sin), None

        x, _ = self.single_transformer_blocks(apply_single, x, None)

        # Extract Image features
        txt_len = txt.shape[1]
        img = x[:, txt_len:]

        # Output Modulation
        (shift_out, scale_out, _) = mod_img[0]

        img_norm = self.norm_out(img)
        img_norm = img_norm * (1 + scale_out[:, None, :]) + shift_out[:, None, :]

        return self.proj_out(img_norm)
