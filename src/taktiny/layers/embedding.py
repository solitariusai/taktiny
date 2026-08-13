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
"""Embedding layers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.nn.initializers import zeros

from taktiny import nn
from taktiny.layers.positional_embedding import FrequencyEmbedding
from taktiny.nn._continuo import (
    _canonical_axis,
    _constrain,
    _normalize_shape,
    _resolve_activation,
)
from taktiny.nn.convolution import default_conv_initializer
from taktiny.utils.typing import (
    ArrayLike,
    Activation,
    AxisNames,
    DType,
    Initializer,
    ShardMode,
)

_PositionEmbedding = (
    ArrayLike | Callable[[tuple[int, ...]], ArrayLike] | None
)


class ProjectionEmbedding(nn.Module):
    """Compose modules into a feature-projection embedding pipeline.

    The fixed order is input projection, activation, optional output
    projection, and optional normalization. Parameterized components are
    supplied as initialized modules so the same container can represent
    timestep, text, image, or other feature projections.
    """

    def __init__(
        self,
        projection: nn.Module,
        *,
        activation: Activation | None = None,
        output_projection: nn.Module | None = None,
        norm: nn.Module | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if not isinstance(projection, nn.Module):
            raise TypeError('projection must be an initialized nn.Module')
        if output_projection is not None and not isinstance(
            output_projection,
            nn.Module,
        ):
            raise TypeError(
                'output_projection must be an initialized nn.Module or None'
            )
        if norm is not None and not isinstance(norm, nn.Module):
            raise TypeError('norm must be an initialized nn.Module or None')

        self.projection = projection
        self.activation = (
            None
            if activation is None
            else _resolve_activation(activation)
        )
        self.activation_name = (
            None
            if self.activation is None
            else getattr(
                self.activation,
                '__name__',
                type(self.activation).__name__,
            )
        )
        self.output_projection = output_projection
        self.norm = norm
        self.shard_mode = shard_mode

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        output = self.projection(x)
        if self.activation is not None:
            output = self.activation(output)
        if self.output_projection is not None:
            output = self.output_projection(output)
        if self.norm is not None:
            output = self.norm(output)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            'activation='
            f'{self.activation_name}, '
            f'output_projection={self.output_projection is not None}, '
            f'norm={self.norm is not None}'
        )


class ConditionEmbedding(nn.Module):
    """Embed named conditions and fuse their representations.

    Each branch consumes the condition with the same name. Branch outputs can
    be summed, concatenated, stacked, or passed to a custom fusion callable.
    An optional projection and normalization are applied after fusion.

    Args:
        embeddings: Mapping from condition names to initialized modules.
        fusion: ``'sum'``, ``'concat'``, ``'stack'``, or a callable receiving
            an ordered mapping of branch outputs.
        axis: Axis used by concatenation or stacking.
        projection: Optional module applied to the fused representation.
        norm: Optional final normalization module.
        shard_mode: Automatic or explicit output-sharding behavior.
    """

    def __init__(
        self,
        embeddings: Mapping[str, nn.Module],
        *,
        fusion: Literal['sum', 'concat', 'stack']
        | Callable[[Mapping[str, jax.Array]], jax.Array] = 'sum',
        axis: int = -1,
        projection: nn.Module | None = None,
        norm: nn.Module | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if not embeddings:
            raise ValueError('embeddings must contain at least one branch')
        self.embeddings = nn.Dict(embeddings)
        if isinstance(fusion, str):
            if fusion not in {'sum', 'concat', 'stack'}:
                raise ValueError(
                    "fusion must be 'sum', 'concat', 'stack', or callable"
                )
        elif not callable(fusion):
            raise TypeError(
                "fusion must be 'sum', 'concat', 'stack', or callable"
            )
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise TypeError('axis must be an integer')
        if projection is not None and not isinstance(projection, nn.Module):
            raise TypeError('projection must be an initialized nn.Module or None')
        if norm is not None and not isinstance(norm, nn.Module):
            raise TypeError('norm must be an initialized nn.Module or None')

        self.fusion = fusion
        self.axis = axis
        self.projection = projection
        self.norm = norm
        self.shard_mode = shard_mode

    def _fuse(self, outputs: Mapping[str, jax.Array]) -> jax.Array:
        if callable(self.fusion):
            return jnp.asarray(self.fusion(outputs))

        values = tuple(outputs.values())
        if self.fusion in {'sum', 'stack'}:
            shape = values[0].shape
            if any(value.shape != shape for value in values[1:]):
                raise ValueError(
                    f"fusion='{self.fusion}' requires equal branch shapes"
                )
        if self.fusion == 'sum':
            output = values[0]
            for value in values[1:]:
                output = output + value
            return output
        if self.fusion == 'stack':
            return jnp.stack(values, axis=self.axis)

        rank = values[0].ndim
        axis = _canonical_axis(self.axis, rank)
        reference = values[0].shape
        for value in values[1:]:
            if value.ndim != rank or any(
                value.shape[index] != reference[index]
                for index in range(rank)
                if index != axis
            ):
                raise ValueError(
                    "fusion='concat' requires matching non-concatenated "
                    'dimensions'
                )
        return jnp.concatenate(values, axis=axis)

    def __call__(
        self,
        conditions: Mapping[str, Any] | None = None,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
        **named_conditions: Any,
    ) -> jax.Array:
        if conditions is not None and named_conditions:
            raise ValueError(
                'pass conditions as a mapping or keyword arguments, not both'
            )
        conditions = (
            named_conditions
            if conditions is None
            else dict(conditions)
        )
        expected = set(self.embeddings.keys())
        actual = set(conditions)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            details = []
            if missing:
                details.append(f'missing {missing}')
            if unexpected:
                details.append(f'unexpected {unexpected}')
            raise ValueError(
                'condition names do not match branches: '
                + ', '.join(details)
            )

        outputs = {
            name: embedding(conditions[name])
            for name, embedding in self.embeddings.items()
        }
        output = self._fuse(outputs)
        if self.projection is not None:
            output = self.projection(output)
        if self.norm is not None:
            output = self.norm(output)
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        fusion = (
            self.fusion
            if isinstance(self.fusion, str)
            else getattr(self.fusion, '__name__', type(self.fusion).__name__)
        )
        return f'branches={len(self.embeddings)}, fusion={fusion}'


class PatchEmbedding(nn.Module):
    """Project non-overlapping or overlapping patches into token embeddings.

    Inputs use channels-last layout and may be unbatched
    ``[*spatial, channels]`` or batched ``[batch, *spatial, channels]``.
    The spatial rank is determined by ``patch_size``. An integer denotes a
    square 2D patch; pass an explicit tuple for other ranks, such as ``(4,)``
    for 1D or ``(2, 4, 4)`` for 3D patches.

    Args:
        in_channels: Number of input channels.
        embedding_dim: Size of each patch embedding.
        patch_size: Patch extent in each spatial dimension.
        stride: Patch stride. Defaults to ``patch_size``.
        padding: Padding forwarded to :class:`taktiny.nn.Conv`.
        bias: Whether the projection has a learned bias.
        flatten: Whether to flatten the spatial grid into a token axis.
        norm: Optional callable applied after projection and flattening.
        position_embedding: Optional array or callable receiving the projected
            spatial grid shape. It must return one embedding per patch and is
            added after ``norm``. A leading singleton batch axis is accepted.
        dtype: Projection parameter dtype.
        rngs: Random stream used to initialize the projection.
        initializer: Projection weight initializer.
        bias_initializer: Projection bias initializer.
        axis_names: Logical axes for the convolution weight. Its length must
            be ``spatial_rank + 2``.
        shard_mode: Automatic or explicit output-sharding behavior.
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        patch_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        dtype: DType = jnp.float32,
        *,
        bias: bool = True,
        flatten: bool = True,
        norm: nn.Module | Callable[[jax.Array], jax.Array] | None = None,
        position_embedding: _PositionEmbedding = None,
        rngs: nn.Rngs,
        initializer: Initializer = default_conv_initializer,
        bias_initializer: Initializer = zeros,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError('embedding_dim must be a positive integer')

        if isinstance(patch_size, int):
            patch_shape = _normalize_shape(
                (patch_size, patch_size),
                'patch_size',
            )
        else:
            patch_shape = _normalize_shape(patch_size, 'patch_size')

        if stride is None:
            stride_shape = patch_shape
        elif isinstance(stride, int):
            stride_shape = _normalize_shape(
                (stride,) * len(patch_shape),
                'stride',
            )
        else:
            stride_shape = _normalize_shape(stride, 'stride')
            if len(stride_shape) != len(patch_shape):
                raise ValueError(
                    'stride and patch_size must describe the same number '
                    'of spatial dimensions'
                )

        if norm is not None and not callable(norm):
            raise TypeError('norm must be callable or None')
        if (
            position_embedding is not None
            and not callable(position_embedding)
        ):
            position_embedding = jnp.asarray(position_embedding)

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.patch_size = patch_shape
        self.stride = stride_shape
        self.flatten = bool(flatten)
        self.norm = norm
        self.position_embedding = position_embedding
        self.spatial_rank = len(patch_shape)
        self.shard_mode = shard_mode
        self.projection = nn.Conv(
            in_channels,
            embedding_dim,
            kernel_size=patch_shape,
            stride=stride_shape,
            padding=padding,
            dtype=dtype,
            bias=bias,
            rngs=rngs,
            initializer=initializer,
            bias_initializer=bias_initializer,
            axis_names=axis_names,
            shard_mode=shard_mode,
        )

    def _add_position_embedding(
        self,
        output: jax.Array,
        grid_shape: tuple[int, ...],
        *,
        batched: bool,
    ) -> jax.Array:
        source = self.position_embedding
        if source is None:
            return output

        position = source(grid_shape) if callable(source) else source
        position = jnp.asarray(position, dtype=output.dtype)
        unbatched_shape = output.shape[1:] if batched else output.shape

        if batched:
            valid_shapes = {
                unbatched_shape,
                (1, *unbatched_shape),
                output.shape,
            }
            if position.shape not in valid_shapes:
                raise ValueError(
                    'position_embedding must have shape '
                    f'{unbatched_shape}, {(1, *unbatched_shape)}, or '
                    f'{output.shape}; got {position.shape}'
                )
            if position.shape == unbatched_shape:
                position = position[None, ...]
        else:
            if position.shape == (1, *unbatched_shape):
                position = position[0]
            elif position.shape != unbatched_shape:
                raise ValueError(
                    'position_embedding must have shape '
                    f'{unbatched_shape} or {(1, *unbatched_shape)}; got '
                    f'{position.shape}'
                )

        return output + position

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        batched_rank = self.spatial_rank + 2
        if x.ndim not in {batched_rank - 1, batched_rank}:
            raise ValueError(
                f'expected an unbatched rank-{batched_rank - 1} or batched '
                f'rank-{batched_rank} input, got rank {x.ndim}'
            )
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f'expected {self.in_channels} input channels, got '
                f'{x.shape[-1]}'
            )

        batched = x.ndim == batched_rank
        output = self.projection(x)
        grid_shape = tuple(
            output.shape[1:-1] if batched else output.shape[:-1]
        )

        if self.flatten:
            patch_count = math.prod(grid_shape)
            if batched:
                output = output.reshape(
                    output.shape[0],
                    patch_count,
                    self.embedding_dim,
                )
            else:
                output = output.reshape(patch_count, self.embedding_dim)

        if self.norm is not None:
            input_shape = output.shape
            output = self.norm(output)
            if output.shape != input_shape:
                raise ValueError(
                    f'norm must preserve shape {input_shape}, got '
                    f'{output.shape}'
                )

        output = self._add_position_embedding(
            output,
            grid_shape,
            batched=batched,
        )
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'{self.in_channels} -> {self.embedding_dim}, '
            f'patch={self.patch_size}, stride={self.stride}, '
            f'flatten={self.flatten}'
        )


class CombinedTimestepTextProjEmbedding(ConditionEmbedding):
    """Combine projected timestep and pooled-text conditioning.

    Args:
        embedding_dim: Size of the shared output embedding.
        pooled_projection_dim: Size of the pooled-text input.
        frequency_dim: Size of the sinusoidal timestep embedding.
        frequency_shift: Timestep frequency denominator adjustment.
        flip_sin_to_cos: Emit cosine timestep features before sine features.
        timestep_scale: Multiplier applied to timestep frequencies.
        activation: Activation used by both projection branches.
        bias: Whether the learned projections include biases.
        dtype: Projection parameter and frequency output dtype.
        rngs: Random stream used to initialize the projections.
        quant: Optional quantization rule forwarded to linear projections.
        dot_general: Optional matrix multiplication implementation.
        shard_mode: Automatic or explicit output-sharding behavior.
    """

    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        frequency_dim: int = 256,
        frequency_shift: float = 0.0,
        flip_sin_to_cos: bool = True,
        timestep_scale: float = 1.0,
        activation: Activation = 'silu',
        dtype: DType = jnp.float32,
        *,
        bias: bool = True,
        rngs: nn.Rngs,
        quant: Any = None,
        dot_general: Any = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        for name, value in (
            ('embedding_dim', embedding_dim),
            ('pooled_projection_dim', pooled_projection_dim),
            ('frequency_dim', frequency_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')

        linear_options = {
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'quant': quant,
            'dot_general': dot_general,
            'shard_mode': shard_mode,
        }
        timestep_embedding = ProjectionEmbedding(
            nn.Linear(
                frequency_dim,
                embedding_dim,
                axis_names=('frequency', 'conditioning'),
                **linear_options,
            ),
            activation=activation,
            output_projection=nn.Linear(
                embedding_dim,
                embedding_dim,
                axis_names=('conditioning', 'embed'),
                **linear_options,
            ),
            shard_mode=shard_mode,
        )
        text_embedding = ProjectionEmbedding(
            nn.Linear(
                pooled_projection_dim,
                embedding_dim,
                axis_names=('pooled_projection', 'conditioning'),
                **linear_options,
            ),
            activation=activation,
            output_projection=nn.Linear(
                embedding_dim,
                embedding_dim,
                axis_names=('conditioning', 'embed'),
                **linear_options,
            ),
            shard_mode=shard_mode,
        )
        super().__init__(
            {
                'timestep': nn.Sequential(
                    [
                        FrequencyEmbedding(
                            frequency_dim,
                            frequency_shift=frequency_shift,
                            flip_sin_to_cos=flip_sin_to_cos,
                            scale=timestep_scale,
                            dtype=dtype,
                            shard_mode=shard_mode,
                        ),
                        timestep_embedding,
                    ]
                ),
                'pooled_projection': text_embedding,
            },
            fusion='sum',
            shard_mode=shard_mode,
        )

        self.embedding_dim = embedding_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.frequency_dim = frequency_dim

    def __call__(
        self,
        timestep: jax.Array,
        pooled_projection: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        return super().__call__(
            timestep=timestep,
            pooled_projection=pooled_projection,
            out_sharding=out_sharding,
        )



__all__ = [
    'ConditionEmbedding',
    'PatchEmbedding',
    'ProjectionEmbedding',
    'CombinedTimestepTextProjEmbedding',
]
