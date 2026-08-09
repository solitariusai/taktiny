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
"""LEGACY: Resnet Modules"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from taktiny import nn


# class ResnetBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int | None = None,
#         time_emb_dim: int | None = None,
#         groups: int = 32,
#         eps: float = 1e-5,
#         seed: nn.Rngs | None = None
#     ) -> None:
#         self.in_channels = in_channels
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels

#         self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
#         self.conv1 = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             padding="SAME",
#             rngs=seed,
#         )

#         if time_emb_dim is not None:
#             self.time_proj = nn.Linear(time_emb_dim, out_channels, seed=seed)
#         else:
#             self.time_proj = None

#         self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
#         self.conv2 = nn.Conv2d(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding="SAME",
#             rngs=seed,
#         )

#         if self.in_channels != self.out_channels:
#             self.res_conv = nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=1,
#                 rngs=seed,
#             )
#         else:
#             self.res_conv = None

#     def __call__(self, x: jax.Array, t_emb: jax.Array | None = None) -> jax.Array:
#         # Pre-Norm architecture used by CompVis / Diffusers
#         res = x if self.res_conv is None else self.res_conv(x)

#         h = self.norm1(x)
#         h = jax.nn.silu(h)
#         h = self.conv1(h)

#         if self.time_proj is not None and t_emb is not None:
#             time_bias = self.time_proj(jax.nn.silu(t_emb))
#             if time_bias.ndim == 1:
#                 # Unbatched: time_bias is (C,), h is (H, W, C)
#                 h = h + time_bias[None, None, :]
#             else:
#                 # Batched: time_bias is (B, C), h is (B, H, W, C)
#                 h = h + time_bias[:, None, None, :]

#         h = self.norm2(h)
#         h = jax.nn.silu(h)
#         h = self.conv2(h)

#         return h + res


class Downsample2D(nn.Module):
    pass


class FirDownsample2D(nn.Module):
    pass


class FirUpsample2D(nn.Module):
    pass


class KDownsample2D(nn.Module):
    pass


class KUpsample2D(nn.Module):
    pass


# class ResnetBlock2D(nn.Module):
#     pass


class ResnetBlockCondNorm2D(nn.Module):
    pass


class Upsample2D(nn.Module):
    pass
