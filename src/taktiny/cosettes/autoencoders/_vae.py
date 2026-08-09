# # Copyright 2026 Shinapri
# # Copyright 2026 The HuggingFace Team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # DEPRECATED / LEGACY
# # TODO(refactor).
# from __future__ import annotations

# import jax
# from taktiny import nn
# from taktiny import layers
# from taktiny.cosettes.unets._unet import get_down_block, get_up_block, UNetMidBlock2D
# from taktiny.layers import SpatialNorm

# class SpatialAttention(nn.Module):
#     def __init__(self, dim: int, seed: nn.Rngs) -> None:
#         # Standard VAE attention uses 1 head with head_dim = dim
#         self.norm = nn.GroupNorm(num_groups=32, num_channels=dim)
#         self.attn = layers.Attention(
#             hidden_size=dim,
#             num_heads=1,
#             head_dim=dim,
#             bias=True,
#             seed=seed
#         )

#     def __call__(self, x: jax.Array) -> jax.Array:
#         B, H, W, C = x.shape

#         # Pre-Norm
#         h = self.norm(x)

#         # Flatten spatial dimensions
#         h = h.reshape(B, H * W, C)

#         # Attention
#         h, _ = self.attn(h)

#         # Unflatten spatial dimensions
#         h = h.reshape(B, H, W, C)

#         # Residual connection
#         return x + h

# class Encoder(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         dims: list[int],
#         latent_dim: int,
#         depths: list[int] | None = None,
#         seed: nn.Rngs | None = None
#     ) -> None:
#         self.conv_in = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding="SAME", seed=seed)

#         self.down_blocks = nn.List()
#         layers_per_block = depths[0] if depths is not None else 2
#         current_channels = dims[0]

#         for i, dim in enumerate(dims):
#             # ResNet blocks
#             for _ in range(layers_per_block):
#                 self.down_blocks.layers.append(
#                     layers.ResnetBlock2D(in_channels=current_channels, out_channels=dim, seed=seed)
#                 )
#                 current_channels = dim

#             # Downsample (except for the last block)
#             if i != len(dims) - 1:
#                 self.down_blocks.layers.append(
#                     nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding="SAME", seed=seed)
#                 )

#         # Mid Block
#         self.mid_block1 = layers.ResnetBlock2D(in_channels=current_channels, out_channels=current_channels, seed=seed)
#         self.mid_attn = SpatialAttention(dim=current_channels, seed=seed)
#         self.mid_block2 = layers.ResnetBlock2D(in_channels=current_channels, out_channels=current_channels, seed=seed)

#         # Output
#         self.norm_out = nn.GroupNorm(num_groups=32, num_channels=current_channels)
#         self.conv_out = nn.Conv2d(current_channels, latent_dim, kernel_size=3, padding="SAME", seed=seed)

#     def __call__(self, x: jax.Array) -> jax.Array:
#         x = self.conv_in(x)

#         for layer in self.down_blocks:
#             x = layer(x)

#         x = self.mid_block1(x)
#         x = self.mid_attn(x)
#         x = self.mid_block2(x)

#         x = self.norm_out(x)
#         x = jax.nn.silu(x)
#         x = self.conv_out(x)

#         return x

# class Decoder(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         dims: list[int],
#         latent_dim: int,
#         depths: list[int] | None = None,
#         seed: nn.Rngs | None = None
#     ) -> None:
#         self.conv_in = nn.Conv2d(latent_dim, dims[-1], kernel_size=3, padding="SAME", seed=seed)

#         # Mid Block
#         current_channels = dims[-1]
#         self.mid_block1 = layers.ResnetBlock2D(in_channels=current_channels, out_channels=current_channels, seed=seed)
#         self.mid_attn = SpatialAttention(dim=current_channels, seed=seed)
#         self.mid_block2 = layers.ResnetBlock2D(in_channels=current_channels, out_channels=current_channels, seed=seed)

#         self.up_blocks = nn.List()
#         layers_per_block = depths[0] if depths is not None else 2

#         reversed_dims = list(reversed(dims))

#         for i, dim in enumerate(reversed_dims):
#             # ResNet blocks (Diffusers uses layers_per_block + 1 for upsampling blocks)
#             for _ in range(layers_per_block + 1):
#                 self.up_blocks.layers.append(
#                     layers.ResnetBlock2D(in_channels=current_channels, out_channels=dim, seed=seed)
#                 )
#                 current_channels = dim

#             # Upsample (except for the last block)
#             if i != len(reversed_dims) - 1:
#                 self.up_blocks.layers.append(
#                     nn.Sequential(
#                         nn.Upsample2d(scale_factor=2),
#                         nn.Conv2d(current_channels, current_channels, kernel_size=3, padding="SAME", seed=seed)
#                     )
#                 )

#         # Output
#         self.norm_out = nn.GroupNorm(num_groups=32, num_channels=current_channels)
#         self.conv_out = nn.Conv2d(current_channels, in_channels, kernel_size=3, padding="SAME", seed=seed)

#     def __call__(self, x: jax.Array) -> jax.Array:
#         x = self.conv_in(x)

#         x = self.mid_block1(x)
#         x = self.mid_attn(x)
#         x = self.mid_block2(x)

#         for layer in self.up_blocks:
#             x = layer(x)

#         x = self.norm_out(x)
#         x = jax.nn.silu(x)
#         x = self.conv_out(x)

#         return x

# class Autoencoder(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         dims: list[int],
#         latent_dim: int,
#         depths: list[int] | None = None,
#         seed: nn.Rngs | None = None
#     ) -> None:
#         self.encoder = Encoder(
#             in_channels=in_channels,
#             dims=dims,
#             latent_dim=latent_dim,
#             depths=depths,
#             seed=seed
#         )
#         self.decoder = Decoder(
#             in_channels=in_channels,
#             dims=dims,
#             latent_dim=latent_dim,
#             depths=depths,
#             seed=seed
#         )

#     def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
#         latent = self.encoder(x)
#         reconstructed = self.decoder(latent)
#         return reconstructed, latent

# class Encoder(nn.Module):
#     r"""
#     The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

#     Args:
#         in_channels (`int`, *optional*, defaults to 3):
#             The number of input channels.
#         out_channels (`int`, *optional*, defaults to 3):
#             The number of output channels.
#         down_block_types (`tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
#             The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
#             options.
#         block_out_channels (`tuple[int, ...]`, *optional*, defaults to `(64,)`):
#             The number of output channels for each block.
#         layers_per_block (`int`, *optional*, defaults to 2):
#             The number of layers per block.
#         norm_num_groups (`int`, *optional*, defaults to 32):
#             The number of groups for normalization.
#         act_fn (`str`, *optional*, defaults to `"silu"`):
#             The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
#         double_z (`bool`, *optional*, defaults to `True`):
#             Whether to double the number of output channels for the last block.
#     """

#     def __init__(
#         self,
#         in_channels: int = 3,
#         out_channels: int = 3,
#         down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
#         block_out_channels: tuple[int, ...] = (64,),
#         layers_per_block: int = 2,
#         norm_num_groups: int = 32,
#         act_fn: str = "silu",
#         double_z: bool = True,
#         mid_block_add_attention: bool=True,
#         *, rngs: nn.Rngs
#     ) -> None:
#         self.layers_per_block = layers_per_block

#         self.conv_in = nn.Conv2d(
#             in_channels,
#             block_out_channels[0],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             rngs=rngs
#         )

#         self.down_blocks = nn.List([])

#         # down
#         output_channel = block_out_channels[0]
#         for i, down_block_type in enumerate(down_block_types):
#             input_channel = output_channel
#             output_channel = block_out_channels[i]
#             is_final_block = i == len(block_out_channels) - 1

#             down_block = get_down_block(
#                 down_block_type,
#                 num_layers=self.layers_per_block,
#                 in_channels=input_channel,
#                 out_channels=output_channel,
#                 add_downsample=not is_final_block,
#                 resnet_eps=1e-6,
#                 downsample_padding=0,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 attention_head_dim=output_channel,
#                 temb_channels=None,
#             )
#             self.down_blocks.append(down_block)

#         # mid
#         self.mid_block = UNetMidBlock2D(
#             in_channels=block_out_channels[-1],
#             resnet_eps=1e-6,
#             resnet_act_fn=act_fn,
#             output_scale_factor=1,
#             resnet_time_scale_shift="default",
#             attention_head_dim=block_out_channels[-1],
#             resnet_groups=norm_num_groups,
#             temb_channels=None,
#             add_attention=mid_block_add_attention,
#         )

#         # out
#         self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
#         self.conv_act = nn.SiLU()

#         conv_out_channels = 2 * out_channels if double_z else out_channels
#         self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)


#     def __call__(self, sample: jax.Array) -> jax.Array:
#         r"""The forward method of the `Encoder` class."""

#         sample = self.conv_in(sample)

#         # down
#         for down_block in self.down_blocks:
#             sample = down_block(sample)

#         # middle
#         sample = self.mid_block(sample)

#         # post-process
#         sample = self.conv_norm_out(sample)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)

#         return sample

# class Decoder(nn.Module):
#     r"""
#     The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

#     Args:
#         in_channels (`int`, *optional*, defaults to 3):
#             The number of input channels.
#         out_channels (`int`, *optional*, defaults to 3):
#             The number of output channels.
#         up_block_types (`tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
#             The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
#         block_out_channels (`tuple[int, ...]`, *optional*, defaults to `(64,)`):
#             The number of output channels for each block.
#         layers_per_block (`int`, *optional*, defaults to 2):
#             The number of layers per block.
#         norm_num_groups (`int`, *optional*, defaults to 32):
#             The number of groups for normalization.
#         act_fn (`str`, *optional*, defaults to `"silu"`):
#             The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
#         norm_type (`str`, *optional*, defaults to `"group"`):
#             The normalization type to use. Can be either `"group"` or `"spatial"`.
#     """

#     def __init__(
#         self,
#         in_channels: int = 3,
#         out_channels: int = 3,
#         up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
#         block_out_channels: tuple[int, ...] = (64,),
#         layers_per_block: int = 2,
#         norm_num_groups: int = 32,
#         act_fn: str = "silu",
#         norm_type: str = "group",  # group, spatial
#         mid_block_add_attention: bool=True,
#         *, rngs: nn.Rngs
#     ) -> None:
#         self.layers_per_block = layers_per_block

#         self.conv_in = nn.Conv2d(
#             in_channels,
#             block_out_channels[-1],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#         self.up_blocks = nn.List([])

#         temb_channels = in_channels if norm_type == "spatial" else None

#         # mid
#         self.mid_block = UNetMidBlock2D(
#             in_channels=block_out_channels[-1],
#             resnet_eps=1e-6,
#             resnet_act_fn=act_fn,
#             output_scale_factor=1,
#             resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
#             attention_head_dim=block_out_channels[-1],
#             resnet_groups=norm_num_groups,
#             temb_channels=temb_channels,
#             add_attention=mid_block_add_attention,
#         )

#         # up
#         reversed_block_out_channels = list(reversed(block_out_channels))
#         output_channel = reversed_block_out_channels[0]
#         for i, up_block_type in enumerate(up_block_types):
#             prev_output_channel = output_channel
#             output_channel = reversed_block_out_channels[i]

#             is_final_block = i == len(block_out_channels) - 1

#             up_block = get_up_block(
#                 up_block_type,
#                 num_layers=self.layers_per_block + 1,
#                 in_channels=prev_output_channel,
#                 out_channels=output_channel,
#                 prev_output_channel=prev_output_channel,
#                 add_upsample=not is_final_block,
#                 resnet_eps=1e-6,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 attention_head_dim=output_channel,
#                 temb_channels=temb_channels,
#                 resnet_time_scale_shift=norm_type,
#             )
#             self.up_blocks.append(up_block)
#             prev_output_channel = output_channel

#         # out
#         if norm_type == "spatial":
#             self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels, rngs)
#         else:
#             self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)

#         self.conv_act = nn.SiLU()
#         self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

#         self.gradient_checkpointing = False

#     def __call__(
#         self,
#         sample: jax.Array,
#         latent_embeds: jax.Array | None = None,
#     ) -> jax.Array:
#         r"""The forward method of the `Decoder` class."""

#         sample = self.conv_in(sample)

#         # middle
#         sample = self.mid_block(sample, latent_embeds)

#         # up
#         for up_block in self.up_blocks:
#             sample = up_block(sample, latent_embeds)

#         # post-process
#         if latent_embeds is None:
#             sample = self.conv_norm_out(sample)
#         else:
#             sample = self.conv_norm_out(sample, latent_embeds)

#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)

#         return sample
