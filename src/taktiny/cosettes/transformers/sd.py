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
"""Stable Diffusion 3 transformer architectures."""

from __future__ import annotations

from collections.abc import Sequence
import typing as tp

import jax
import jax.numpy as jnp

from taktiny.cosettes import layers as ly
from taktiny import nn
from taktiny.cosettes.continuo import _approximate_gelu
from taktiny.cosettes.transformers.ordinario import JointTransformerLayer
from taktiny.maestro.config import ModelConfig
from taktiny.nn.continuo import _constrain
from taktiny.utils.typing import DType, ShardMode


class SD3TransformerLayer(JointTransformerLayer):
    """Stable Diffusion 3 MMDiT layer."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            activation=_approximate_gelu,
            input_layernorm=ly.AdaXNorm,
            context_input_layernorm=ly.AdaXNorm,
            joint_attention=ly.JointAttention,
            second_attention=ly.AttentionLegacy,
            post_attention_layernorm=nn.LayerNorm,
            context_post_attention_layernorm=nn.LayerNorm,
            mlp=ly.FeedForward,
            context_mlp=ly.FeedForward,
        )

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array,
        temb: jax.Array,
        **attention_kwargs: tp.Any,
    ) -> tuple[jax.Array | None, jax.Array]:
        return super().__call__(x, enc_x, temb, **attention_kwargs)


class SD3PatchEmbedding(ly.PatchEmbedding):
    """Patchify an NHWC latent and add SD3's 2D sinusoidal positions.

    A configured ``pos_embed_max_size`` stores one fixed square position
    table and center-crops it for each latent grid. Without a maximum size,
    positions are generated dynamically when the input grid differs from the
    training grid.
    """

    def __init__(
        self,
        sample_size: int | Sequence[int],
        patch_size: int | Sequence[int],
        in_channels: int,
        embedding_dim: int,
        *,
        pos_embed_max_size: int | Sequence[int] | None = None,
        interpolation_scale: float = 1.0,
        pos_embed_type: str | None = 'sincos',
        bias: bool = True,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        sample_shape = self._pair(sample_size, 'sample_size')
        patch_shape = self._pair(patch_size, 'patch_size')
        if any(size % patch for size, patch in zip(sample_shape, patch_shape)):
            raise ValueError('sample_size must be divisible by patch_size')
        if interpolation_scale <= 0:
            raise ValueError('interpolation_scale must be positive')
        if pos_embed_type not in {'sincos', None}:
            raise ValueError("pos_embed_type must be 'sincos' or None")
        if pos_embed_type == 'sincos' and embedding_dim % 4:
            raise ValueError(
                'embedding_dim must be divisible by 4 for 2D sincos positions'
            )

        super().__init__(
            in_channels,
            embedding_dim,
            patch_shape,
            stride=patch_shape,
            padding=0,
            bias=bias,
            flatten=True,
            position_embedding=None,
            dtype=dtype,
            rngs=rngs,
            axis_names=('patch_h', 'patch_w', 'in', 'embed'),
            shard_mode=shard_mode,
        )

        self.sample_size = sample_shape
        self.base_grid_size = tuple(
            size // patch for size, patch in zip(sample_shape, patch_shape)
        )
        self.pos_embed_max_size = (
            None
            if pos_embed_max_size is None
            else self._pair(pos_embed_max_size, 'pos_embed_max_size')
        )
        self.interpolation_scale = float(interpolation_scale)
        self.pos_embed_type = pos_embed_type

        if pos_embed_type is None:
            self.pos_embed = None
        else:
            grid_size = self.pos_embed_max_size or self.base_grid_size
            position = self._sincos_2d(
                embedding_dim,
                grid_size,
                base_grid_size=self.base_grid_size,
                interpolation_scale=self.interpolation_scale,
            )[None, ...]
            if self.pos_embed_max_size is None:
                self.pos_embed = position
            else:
                self.pos_embed = nn.Parameter(position, trainable=False)
                self.pos_embed.axis_names = (None, 'sequence', 'embed')

    @staticmethod
    def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
        values = (value, value) if isinstance(value, int) else tuple(value)
        if len(values) != 2:
            raise ValueError(f'{name} must contain exactly two dimensions')
        for index, size in enumerate(values):
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f'{name}[{index}] must be a positive integer')
        return tp.cast(tuple[int, int], values)

    @staticmethod
    def _sincos_1d(embedding_dim: int, positions: jax.Array) -> jax.Array:
        if embedding_dim % 2:
            raise ValueError('1D position embedding dimension must be even')
        frequencies = jnp.arange(embedding_dim // 2, dtype=jnp.float32)
        frequencies = frequencies / (embedding_dim / 2.0)
        frequencies = 1.0 / (10_000.0**frequencies)
        phases = jnp.outer(positions.reshape(-1), frequencies)
        return jnp.concatenate((jnp.sin(phases), jnp.cos(phases)), axis=-1)

    @classmethod
    def _sincos_2d(
        cls,
        embedding_dim: int,
        grid_size: tuple[int, int],
        *,
        base_grid_size: tuple[int, int],
        interpolation_scale: float,
    ) -> jax.Array:
        height, width = grid_size
        base_height, base_width = base_grid_size
        grid_h = (
            jnp.arange(height, dtype=jnp.float32)
            / (height / base_height)
            / interpolation_scale
        )
        grid_w = (
            jnp.arange(width, dtype=jnp.float32)
            / (width / base_width)
            / interpolation_scale
        )
        positions_h, positions_w = jnp.meshgrid(
            grid_h,
            grid_w,
            indexing='ij',
        )
        half = embedding_dim // 2
        return jnp.concatenate(
            (
                cls._sincos_1d(half, positions_w),
                cls._sincos_1d(half, positions_h),
            ),
            axis=-1,
        )

    def _positions(self, grid_size: tuple[int, int]) -> jax.Array | None:
        if self.pos_embed is None:
            return None
        table = (
            self.pos_embed.value
            if isinstance(self.pos_embed, nn.Parameter)
            else self.pos_embed
        )

        if self.pos_embed_max_size is None:
            if grid_size == self.base_grid_size:
                return table
            return self._sincos_2d(
                self.embedding_dim,
                grid_size,
                base_grid_size=self.base_grid_size,
                interpolation_scale=self.interpolation_scale,
            )[None, ...]

        maximum_height, maximum_width = self.pos_embed_max_size
        height, width = grid_size
        if height > maximum_height or width > maximum_width:
            raise ValueError(
                f'patch grid {grid_size} exceeds positional table '
                f'{self.pos_embed_max_size}'
            )
        top = (maximum_height - height) // 2
        left = (maximum_width - width) // 2
        table = table.reshape(
            1,
            maximum_height,
            maximum_width,
            self.embedding_dim,
        )
        table = table[:, top : top + height, left : left + width, :]
        return table.reshape(1, height * width, self.embedding_dim)

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        if x.ndim not in {3, 4}:
            raise ValueError('SD3PatchEmbedding expects HWC or NHWC input')
        spatial_shape = x.shape[-3:-1]
        if any(
            size % patch
            for size, patch in zip(spatial_shape, self.patch_size)
        ):
            raise ValueError('input spatial dimensions must divide into patches')

        output = super().__call__(x)
        grid_size = tuple(
            size // patch
            for size, patch in zip(spatial_shape, self.patch_size)
        )
        position = self._positions(tp.cast(tuple[int, int], grid_size))
        if position is not None:
            if x.ndim == 3:
                position = position[0]
            output = output + position.astype(output.dtype)
        return _constrain(output, out_sharding, self.shard_mode)

__all__ = [
    'SD3PatchEmbedding',
    'SD3TransformerLayer',
]
