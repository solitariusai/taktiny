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
"""Deep-compression autoencoder used by SANA."""

from __future__ import annotations

from collections.abc import Sequence
import math
import typing as tp

import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.cosettes._continuo import (
    _config_value,
    _integer_tuple,
    _multiscales,
    _normalization,
    _pixel_shuffle,
    _pixel_unshuffle,
    _positive_int,
    _shard_mode,
    _stage_values,
)
from taktiny.cosettes.autoencoders._ordinario import Autoencoder
from taktiny.layers import GLUMBConv
from taktiny.maestro.config import ModelConfig
from taktiny.nn._continuo import _constrain, _resolve_activation
from taktiny.utils.typing import DType, ShardMode


_StageValue = str | Sequence[str]


class DCResBlock(nn.Module):
    """Residual two-convolution DC-AE block for channels-last image grids."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm_type: str = 'batch_norm',
        activation: str | None = 'relu6',
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if in_channels != out_channels:
            raise ValueError('DCResBlock requires matching input/output channels')
        self.in_channels = _positive_int(in_channels, 'in_channels')
        self.out_channels = _positive_int(out_channels, 'out_channels')
        self.activation = _resolve_activation(activation, allow_none=True)
        self.conv1 = nn.Conv(
            in_channels,
            in_channels,
            (3, 3),
            padding=1,
            dtype=dtype,
            bias=True,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.conv2 = nn.Conv(
            in_channels,
            out_channels,
            (3, 3),
            padding=1,
            dtype=dtype,
            bias=False,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.norm = _normalization(
            norm_type,
            out_channels,
            dtype=dtype,
            shard_mode=shard_mode,
        )
        self.shard_mode = shard_mode

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.norm(self.conv2(x))
        return _constrain(x + residual, out_sharding, self.shard_mode)


class _MultiscaleProjection(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        kernel_size: int,
        *,
        dtype: DType,
        rngs: nn.Rngs,
        shard_mode: ShardMode,
    ) -> None:
        qkv_channels = 3 * channels
        self.proj_in = nn.Conv(
            qkv_channels,
            qkv_channels,
            (kernel_size, kernel_size),
            padding=kernel_size // 2,
            groups=qkv_channels,
            dtype=dtype,
            bias=False,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.proj_out = nn.Conv(
            qkv_channels,
            qkv_channels,
            (1, 1),
            groups=3 * num_heads,
            dtype=dtype,
            bias=False,
            rngs=rngs,
            shard_mode=shard_mode,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.proj_out(self.proj_in(x))


class MultiscaleLinearAttention(nn.Module):
    """SANA multi-scale linear/quadratic spatial attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_attention_heads: int | None = None,
        attention_head_dim: int = 8,
        mult: float = 1.0,
        norm_type: str = 'batch_norm',
        kernel_sizes: Sequence[int] = (5,),
        eps: float = 1e-15,
        residual_connection: bool = False,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        in_channels = _positive_int(in_channels, 'in_channels')
        out_channels = _positive_int(out_channels, 'out_channels')
        attention_head_dim = _positive_int(
            attention_head_dim,
            'attention_head_dim',
        )
        if not math.isfinite(mult) or mult <= 0:
            raise ValueError('mult must be finite and positive')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')
        if num_attention_heads is None:
            num_attention_heads = int(in_channels // attention_head_dim * mult)
        num_attention_heads = _positive_int(
            num_attention_heads,
            'num_attention_heads',
        )
        kernels = tuple(kernel_sizes)
        if any(
            not isinstance(kernel, int)
            or kernel <= 0
            or kernel % 2 == 0
            for kernel in kernels
        ):
            raise ValueError('kernel_sizes must contain positive odd integers')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.eps = float(eps)
        self.residual_connection = bool(residual_connection)
        self.shard_mode = shard_mode

        self.to_q = nn.Linear(
            in_channels,
            self.inner_dim,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.to_k = nn.Linear(
            in_channels,
            self.inner_dim,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.to_v = nn.Linear(
            in_channels,
            self.inner_dim,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.to_qkv_multiscale = nn.List(
            [
                _MultiscaleProjection(
                    self.inner_dim,
                    num_attention_heads,
                    kernel,
                    dtype=dtype,
                    rngs=rngs,
                    shard_mode=shard_mode,
                )
                for kernel in kernels
            ]
        )
        self.to_out = nn.Linear(
            self.inner_dim * (1 + len(kernels)),
            out_channels,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.norm_out = _normalization(
            norm_type,
            out_channels,
            dtype=dtype,
            shard_mode=shard_mode,
        )

    def _linear_attention(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
    ) -> jax.Array:
        value = jnp.pad(value, ((0, 0), (0, 0), (0, 1), (0, 0)))
        scores = jnp.matmul(value, jnp.swapaxes(key, -1, -2))
        output = jnp.matmul(scores, query).astype(jnp.float32)
        return output[:, :, :-1] / (output[:, :, -1:] + self.eps)

    def _quadratic_attention(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
    ) -> jax.Array:
        scores = jnp.matmul(jnp.swapaxes(key, -1, -2), query)
        scores = scores.astype(jnp.float32)
        scores = scores / (jnp.sum(scores, axis=-2, keepdims=True) + self.eps)
        return jnp.matmul(value, scores.astype(value.dtype))

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        if x.ndim != 4 or x.shape[-1] != self.in_channels:
            raise ValueError(
                'MultiscaleLinearAttention expects '
                f'[batch, height, width, {self.in_channels}]'
            )
        residual = x
        batch, height, width, _ = x.shape
        original_dtype = x.dtype

        qkv = jnp.concatenate(
            (self.to_q(x), self.to_k(x), self.to_v(x)),
            axis=-1,
        )
        qkv = jnp.concatenate(
            (qkv, *(projection(qkv) for projection in self.to_qkv_multiscale)),
            axis=-1,
        )
        if height * width > self.attention_head_dim:
            qkv = qkv.astype(jnp.float32)
        qkv = qkv.reshape(
            batch,
            height * width,
            -1,
            3 * self.attention_head_dim,
        )
        qkv = jnp.transpose(qkv, (0, 2, 3, 1))
        query, key, value = jnp.split(qkv, 3, axis=2)
        query = jax.nn.relu(query)
        key = jax.nn.relu(key)

        if height * width > self.attention_head_dim:
            x = self._linear_attention(query, key, value).astype(original_dtype)
        else:
            x = self._quadratic_attention(query, key, value)
        x = jnp.transpose(x, (0, 3, 1, 2)).reshape(
            batch,
            height,
            width,
            -1,
        )
        x = self.norm_out(self.to_out(x))
        if self.residual_connection:
            x = x + residual
        return _constrain(x, out_sharding, self.shard_mode)


class EfficientViTBlock(nn.Module):
    """Attention and convolutional feed-forward block used by DC-AE."""

    def __init__(
        self,
        channels: int,
        *,
        mult: float = 1.0,
        attention_head_dim: int = 32,
        qkv_multiscales: Sequence[int] = (5,),
        norm_type: str = 'batch_norm',
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.attn = MultiscaleLinearAttention(
            channels,
            channels,
            attention_head_dim=attention_head_dim,
            mult=mult,
            norm_type=norm_type,
            kernel_sizes=qkv_multiscales,
            residual_connection=True,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.conv_out = GLUMBConv(
            channels,
            4 * channels,
            norm_type='rms_norm',
            norm_bias=True,
            residual_connection=True,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        return self.conv_out(self.attn(x), out_sharding=out_sharding)


def _block(
    block_type: str,
    channels: int,
    *,
    attention_head_dim: int,
    norm_type: str,
    activation: str,
    qkv_multiscales: Sequence[int],
    dtype: DType,
    rngs: nn.Rngs,
    shard_mode: ShardMode,
) -> nn.Module:
    normalized = block_type.lower().replace('-', '').replace('_', '')
    if normalized in {'resblock', 'residualblock'}:
        return DCResBlock(
            channels,
            channels,
            norm_type=norm_type,
            activation=activation,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )
    if normalized in {'efficientvitblock', 'efficientvit'}:
        return EfficientViTBlock(
            channels,
            attention_head_dim=attention_head_dim,
            qkv_multiscales=qkv_multiscales,
            norm_type=norm_type,
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )
    raise ValueError(f'unsupported DC-AE block type: {block_type!r}')


class DCDownBlock(nn.Module):
    """Downsample by strided convolution or convolution plus pixel unshuffle."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        pixel_unshuffle: bool = False,
        shortcut: bool = True,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.factor = 2
        self.pixel_unshuffle = bool(pixel_unshuffle)
        self.shortcut = bool(shortcut)
        numerator = in_channels * self.factor**2
        if shortcut and numerator % out_channels:
            raise ValueError(
                'shortcut channels require in_channels * 4 to be divisible '
                'by out_channels'
            )
        self.group_size = numerator // out_channels
        conv_channels = out_channels
        stride = 2
        if self.pixel_unshuffle:
            if out_channels % self.factor**2:
                raise ValueError(
                    'pixel-unshuffle output channels must be divisible by four'
                )
            conv_channels //= self.factor**2
            stride = 1
        self.conv = nn.Conv(
            in_channels,
            conv_channels,
            (3, 3),
            stride=stride,
            padding=1,
            dtype=dtype,
            bias=True,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.shard_mode = shard_mode

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        output = self.conv(x)
        if self.pixel_unshuffle:
            output = _pixel_unshuffle(output, self.factor)
        if self.shortcut:
            shortcut = _pixel_unshuffle(x, self.factor)
            shortcut = shortcut.reshape(
                *shortcut.shape[:-1],
                -1,
                self.group_size,
            ).mean(axis=-1)
            output = output + shortcut
        return _constrain(output, out_sharding, self.shard_mode)


class DCUpBlock(nn.Module):
    """Upsample by interpolation or convolution plus pixel shuffle."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = 'nearest',
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.factor = 2
        self.interpolate = bool(interpolate)
        self.shortcut = bool(shortcut)
        numerator = out_channels * self.factor**2
        if shortcut and numerator % in_channels:
            raise ValueError(
                'shortcut channels require out_channels * 4 to be divisible '
                'by in_channels'
            )
        self.repeats = numerator // in_channels
        conv_channels = out_channels if interpolate else numerator
        self.resize = (
            nn.Upsample(
                scale_factor=(2, 2),
                method=interpolation_mode,
                shard_mode=shard_mode,
            )
            if interpolate
            else None
        )
        self.conv = nn.Conv(
            in_channels,
            conv_channels,
            (3, 3),
            padding=1,
            dtype=dtype,
            bias=True,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.shard_mode = shard_mode

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        if self.interpolate:
            output = self.conv(self.resize(x))
        else:
            output = _pixel_shuffle(self.conv(x), self.factor)
        if self.shortcut:
            shortcut = jnp.repeat(x, self.repeats, axis=-1)
            output = output + _pixel_shuffle(shortcut, self.factor)
        return _constrain(output, out_sharding, self.shard_mode)


class DCEncoder(nn.Module):
    """DC-AE image encoder using channels-last arrays."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        *,
        attention_head_dim: int,
        block_types: _StageValue,
        block_out_channels: Sequence[int],
        layers_per_block: Sequence[int],
        qkv_multiscales: Sequence[Sequence[int]],
        downsample_block_type: str,
        out_shortcut: bool,
        dtype: DType,
        rngs: nn.Rngs,
        shard_mode: ShardMode,
    ) -> None:
        channels = _integer_tuple(block_out_channels, 'block_out_channels')
        if any(value == 0 for value in channels):
            raise ValueError('block_out_channels values must be positive')
        layers = _integer_tuple(layers_per_block, 'layers_per_block')
        if len(layers) != len(channels):
            raise ValueError('layers_per_block must match block_out_channels')
        block_types = _stage_values(block_types, len(channels), 'block_types')
        scales = _multiscales(qkv_multiscales, len(channels), 'qkv_multiscales')
        downsample = downsample_block_type.lower().replace('-', '_')
        if downsample == 'conv':
            downsample = 'stride'
        if downsample not in {'pixel_unshuffle', 'stride'}:
            raise ValueError(
                "downsample_block_type must be 'pixel_unshuffle', 'stride', "
                "or 'Conv'"
            )
        if layers[0] == 0 and len(channels) < 2:
            raise ValueError('a zero-layer first stage requires a second stage')

        first_channels = channels[0] if layers[0] > 0 else channels[1]
        self.conv_in = (
            nn.Conv(
                in_channels,
                first_channels,
                (3, 3),
                padding=1,
                dtype=dtype,
                bias=True,
                rngs=rngs,
                shard_mode=shard_mode,
            )
            if layers[0] > 0
            else DCDownBlock(
                in_channels,
                first_channels,
                pixel_unshuffle=downsample == 'pixel_unshuffle',
                shortcut=False,
                dtype=dtype,
                rngs=rngs,
                shard_mode=shard_mode,
            )
        )

        stages: list[nn.Module] = []
        for index, (out_channels, num_layers) in enumerate(zip(channels, layers)):
            modules = [
                _block(
                    block_types[index],
                    out_channels,
                    attention_head_dim=attention_head_dim,
                    norm_type='rms_norm',
                    activation='silu',
                    qkv_multiscales=scales[index],
                    dtype=dtype,
                    rngs=rngs,
                    shard_mode=shard_mode,
                )
                for _ in range(num_layers)
            ]
            if index < len(channels) - 1 and num_layers > 0:
                modules.append(
                    DCDownBlock(
                        out_channels,
                        channels[index + 1],
                        pixel_unshuffle=downsample == 'pixel_unshuffle',
                        shortcut=True,
                        dtype=dtype,
                        rngs=rngs,
                        shard_mode=shard_mode,
                    )
                )
            stages.append(nn.Sequential(modules))
        self.down_blocks = nn.List(stages)
        self.conv_out = nn.Conv(
            channels[-1],
            latent_channels,
            (3, 3),
            padding=1,
            dtype=dtype,
            bias=True,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.out_shortcut = bool(out_shortcut)
        if self.out_shortcut:
            if channels[-1] % latent_channels:
                raise ValueError(
                    'encoder shortcut requires final channels to be divisible '
                    'by latent_channels'
                )
            self.out_shortcut_average_group_size = channels[-1] // latent_channels
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.shard_mode = shard_mode

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        if x.ndim != 4 or x.shape[-1] != self.in_channels:
            raise ValueError(
                f'DCEncoder expects [batch, height, width, {self.in_channels}]'
            )
        x = self.conv_in(x)
        for stage in self.down_blocks:
            x = stage(x)
        output = self.conv_out(x)
        if self.out_shortcut:
            shortcut = x.reshape(
                *x.shape[:-1],
                self.latent_channels,
                self.out_shortcut_average_group_size,
            ).mean(axis=-1)
            output = output + shortcut
        return _constrain(output, out_sharding, self.shard_mode)


class DCDecoder(nn.Module):
    """DC-AE latent decoder using channels-last arrays."""

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        *,
        attention_head_dim: int,
        block_types: _StageValue,
        block_out_channels: Sequence[int],
        layers_per_block: Sequence[int],
        qkv_multiscales: Sequence[Sequence[int]],
        norm_types: _StageValue,
        activations: _StageValue,
        upsample_block_type: str,
        in_shortcut: bool,
        conv_activation: str,
        dtype: DType,
        rngs: nn.Rngs,
        shard_mode: ShardMode,
    ) -> None:
        channels = _integer_tuple(block_out_channels, 'block_out_channels')
        if any(value == 0 for value in channels):
            raise ValueError('block_out_channels values must be positive')
        layers = _integer_tuple(layers_per_block, 'layers_per_block')
        if len(layers) != len(channels):
            raise ValueError('layers_per_block must match block_out_channels')
        count = len(channels)
        block_types = _stage_values(block_types, count, 'block_types')
        norm_types = _stage_values(norm_types, count, 'norm_types')
        activations = _stage_values(activations, count, 'activations')
        scales = _multiscales(qkv_multiscales, count, 'qkv_multiscales')
        upsample = upsample_block_type.lower().replace('-', '_')
        if upsample not in {'pixel_shuffle', 'interpolate'}:
            raise ValueError(
                "upsample_block_type must be 'pixel_shuffle' or 'interpolate'"
            )
        if layers[0] == 0 and count < 2:
            raise ValueError('a zero-layer first stage requires a second stage')

        self.conv_in = nn.Conv(
            latent_channels,
            channels[-1],
            (3, 3),
            padding=1,
            dtype=dtype,
            bias=True,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.in_shortcut = bool(in_shortcut)
        if self.in_shortcut:
            if channels[-1] % latent_channels:
                raise ValueError(
                    'decoder shortcut requires final channels to be divisible '
                    'by latent_channels'
                )
            self.in_shortcut_repeats = channels[-1] // latent_channels

        stages: list[nn.Module | None] = [None] * count
        for index in reversed(range(count)):
            out_channels_at_stage = channels[index]
            modules: list[nn.Module] = []
            if index < count - 1 and layers[index] > 0:
                modules.append(
                    DCUpBlock(
                        channels[index + 1],
                        out_channels_at_stage,
                        interpolate=upsample == 'interpolate',
                        shortcut=True,
                        dtype=dtype,
                        rngs=rngs,
                        shard_mode=shard_mode,
                    )
                )
            modules.extend(
                _block(
                    block_types[index],
                    out_channels_at_stage,
                    attention_head_dim=attention_head_dim,
                    norm_type=norm_types[index],
                    activation=activations[index],
                    qkv_multiscales=scales[index],
                    dtype=dtype,
                    rngs=rngs,
                    shard_mode=shard_mode,
                )
                for _ in range(layers[index])
            )
            stages[index] = nn.Sequential(modules)
        self.up_blocks = nn.List(tp.cast(list[nn.Module], stages))

        final_channels = channels[0] if layers[0] > 0 else channels[1]
        self.norm_out = nn.RMSNorm(
            final_channels,
            eps=1e-5,
            dtype=jnp.float32,
            bias=True,
            axis_names=('embed',),
            shard_mode=shard_mode,
        )
        self.conv_activation = _resolve_activation(conv_activation)
        self.conv_out = (
            nn.Conv(
                final_channels,
                out_channels,
                (3, 3),
                padding=1,
                dtype=dtype,
                bias=True,
                rngs=rngs,
                shard_mode=shard_mode,
            )
            if layers[0] > 0
            else DCUpBlock(
                final_channels,
                out_channels,
                interpolate=upsample == 'interpolate',
                shortcut=False,
                dtype=dtype,
                rngs=rngs,
                shard_mode=shard_mode,
            )
        )
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.shard_mode = shard_mode

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        if x.ndim != 4 or x.shape[-1] != self.latent_channels:
            raise ValueError(
                'DCDecoder expects [batch, height, width, '
                f'{self.latent_channels}]'
            )
        output = self.conv_in(x)
        if self.in_shortcut:
            output = output + jnp.repeat(
                x,
                self.in_shortcut_repeats,
                axis=-1,
            )
        for stage in reversed(self.up_blocks):
            output = stage(output)
        output = self.conv_activation(self.norm_out(output))
        output = self.conv_out(output)
        return _constrain(output, out_sharding, self.shard_mode)


class AutoencoderDC(Autoencoder):
    """Deep-compression autoencoder introduced by DCAE and used by SANA.

    Inputs and outputs use ``[batch, height, width, channels]`` layout. The
    module consumes arrays only, making encoding, decoding, and reconstruction
    directly usable in compiled training code.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: tp.Any = None,
    ) -> None:
        del mesh, sharding_rules
        if not isinstance(config, ModelConfig):
            raise TypeError('config must be a ModelConfig')
        self.config = config
        dtype = _config_value(
            config,
            'torch_dtype',
            'dtype',
            default='float32',
        )
        shard_mode = _shard_mode(config)
        in_channels = _positive_int(
            _config_value(config, 'in_channels', default=3),
            'in_channels',
        )
        latent_channels = _positive_int(
            _config_value(config, 'latent_channels', default=32),
            'latent_channels',
        )
        attention_head_dim = _positive_int(
            _config_value(config, 'attention_head_dim', default=32),
            'attention_head_dim',
        )
        encoder_channels = _config_value(
            config,
            'encoder_block_out_channels',
            default=(128, 256, 512, 512, 1024, 1024),
        )
        decoder_channels = _config_value(
            config,
            'decoder_block_out_channels',
            default=(128, 256, 512, 512, 1024, 1024),
        )
        encoder_layers = _config_value(
            config,
            'encoder_layers_per_block',
            default=(2, 2, 2, 3, 3, 3),
        )
        decoder_layers = _config_value(
            config,
            'decoder_layers_per_block',
            default=(3, 3, 3, 3, 3, 3),
        )
        encoder_stage_count = len(tuple(encoder_channels))
        decoder_stage_count = len(tuple(decoder_channels))
        encoder_default_scales = tuple(
            () if index < encoder_stage_count // 2 else (5,)
            for index in range(encoder_stage_count)
        )
        decoder_default_scales = tuple(
            () if index < decoder_stage_count // 2 else (5,)
            for index in range(decoder_stage_count)
        )
        encoder_scales = _config_value(
            config,
            'encoder_qkv_multiscales',
            default=encoder_default_scales,
        )
        decoder_scales = _config_value(
            config,
            'decoder_qkv_multiscales',
            default=decoder_default_scales,
        )

        encoder = DCEncoder(
            in_channels,
            latent_channels,
            attention_head_dim=attention_head_dim,
            block_types=_config_value(
                config,
                'encoder_block_types',
                default='ResBlock',
            ),
            block_out_channels=encoder_channels,
            layers_per_block=encoder_layers,
            qkv_multiscales=encoder_scales,
            downsample_block_type=_config_value(
                config,
                'downsample_block_type',
                default='pixel_unshuffle',
            ),
            out_shortcut=bool(
                _config_value(config, 'encoder_out_shortcut', default=True)
            ),
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        decoder = DCDecoder(
            in_channels,
            latent_channels,
            attention_head_dim=attention_head_dim,
            block_types=_config_value(
                config,
                'decoder_block_types',
                default='ResBlock',
            ),
            block_out_channels=decoder_channels,
            layers_per_block=decoder_layers,
            qkv_multiscales=decoder_scales,
            norm_types=_config_value(
                config,
                'decoder_norm_types',
                default='rms_norm',
            ),
            activations=_config_value(
                config,
                'decoder_act_fns',
                default='silu',
            ),
            upsample_block_type=_config_value(
                config,
                'upsample_block_type',
                default='pixel_shuffle',
            ),
            in_shortcut=bool(
                _config_value(config, 'decoder_in_shortcut', default=True)
            ),
            conv_activation=_config_value(
                config,
                'decoder_conv_act_fn',
                default='relu',
            ),
            dtype=dtype,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        scaling_factor = _config_value(config, 'scaling_factor', default=1.0)
        super().__init__(
            encoder,
            decoder,
            scaling_factor=scaling_factor,
            spatial_compression_ratio=2 ** (encoder_stage_count - 1),
            temporal_compression_ratio=1,
        )
        self.config = config
        self.in_channels = in_channels
        self.latent_channels = latent_channels

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: tp.Any,
        config: ModelConfig | None = None,
        *,
        local: bool = False,
        subfolder: str | None = None,
        **kwargs: tp.Any,
    ) -> AutoencoderDC:
        """Load a Diffusers-format DC-AE checkpoint directly."""
        if config is None:
            config = ModelConfig.load_config(
                path_or_repo,
                subfolder=subfolder,
                local=local,
            )
        if config is None:
            raise ValueError(f'unable to load AutoencoderDC config from {path_or_repo}')
        return super().from_pretrained(
            path_or_repo,
            config,
            local=local,
            subfolder=subfolder,
            weights_filename='diffusion_pytorch_model.safetensors',
            **kwargs,
        )


__all__ = [
    'AutoencoderDC',
    'DCDecoder',
    'DCDownBlock',
    'DCEncoder',
    'DCResBlock',
    'DCUpBlock',
    'EfficientViTBlock',
    'MultiscaleLinearAttention',
]
