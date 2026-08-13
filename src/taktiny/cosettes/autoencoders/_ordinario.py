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
"""Common array-to-array autoencoder model contract."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.cosettes._overture import PretrainedModel


class Autoencoder(PretrainedModel):
    """Compose an encoder and decoder without imposing a latent distribution.

    The model boundary is deliberately array-only. Text processing, image I/O,
    schedulers, and sampling loops belong outside this module, so ``encode``,
    ``decode``, and reconstruction can be differentiated and compiled normally.

    ``encode`` and ``decode`` operate in the autoencoder's native latent scale.
    Use :meth:`scale_latents` before a diffusion denoiser and
    :meth:`unscale_latents` before decoding when ``scaling_factor`` is not one.

    Args:
        encoder: Initialized module mapping samples to latent arrays.
        decoder: Initialized module mapping latent arrays to samples.
        scaling_factor: Latent multiplier expected by the denoising model.
        spatial_compression_ratio: Spatial sample-to-latent reduction ratio.
        temporal_compression_ratio: Temporal reduction ratio for video models.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        scaling_factor: float = 1.0,
        spatial_compression_ratio: int = 1,
        temporal_compression_ratio: int = 1,
    ) -> None:
        if not isinstance(encoder, nn.Module):
            raise TypeError('encoder must be an initialized nn.Module')
        if not isinstance(decoder, nn.Module):
            raise TypeError('decoder must be an initialized nn.Module')
        if (
            not isinstance(scaling_factor, (int, float))
            or isinstance(scaling_factor, bool)
            or not math.isfinite(scaling_factor)
            or scaling_factor <= 0
        ):
            raise ValueError('scaling_factor must be finite and positive')
        for name, value in (
            ('spatial_compression_ratio', spatial_compression_ratio),
            ('temporal_compression_ratio', temporal_compression_ratio),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')

        self.encoder = encoder
        self.decoder = decoder
        self.scaling_factor = float(scaling_factor)
        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

    def encode(self, sample: jax.Array) -> jax.Array:
        """Encode a sample array into native-scale latent features."""
        return self.encoder(jnp.asarray(sample))

    def decode(self, latents: jax.Array) -> jax.Array:
        """Decode native-scale latent features into a sample array."""
        return self.decoder(jnp.asarray(latents))

    def scale_latents(self, latents: jax.Array) -> jax.Array:
        """Convert native autoencoder latents to denoiser scale."""
        return jnp.asarray(latents) * self.scaling_factor

    def unscale_latents(self, latents: jax.Array) -> jax.Array:
        """Convert denoiser-scale latents back to autoencoder scale."""
        return jnp.asarray(latents) / self.scaling_factor

    def __call__(self, sample: jax.Array) -> jax.Array:
        """Encode and decode a sample without implicit latent scaling."""
        return self.decode(self.encode(sample))

    def extra_repr(self) -> str:
        return (
            f'scale={self.scaling_factor:g}, '
            f'spatial_compression={self.spatial_compression_ratio}, '
            f'temporal_compression={self.temporal_compression_ratio}'
        )


__all__ = ['Autoencoder']
