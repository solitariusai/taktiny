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
"""Conditioned normalization layers."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias
import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.nn.linear import default_linear_initializer
from taktiny.nn.continuo import (
    _constrain,
    _normalize_shape,
    _resolve_activation,
    _validate_integer,
    _validate_positive_float,
)
from taktiny.utils.typing import (
    Activation,
    Axes,
    AxisNames,
    DType,
    Initializer,
    ShardMode,
)

NormType: TypeAlias = Literal['layernorm', 'rmsnorm']
NormModule: TypeAlias = (
    nn.LayerNorm | nn.RMSNorm | nn.GroupNorm | nn.BatchNorm
)
Normalizer: TypeAlias = (
    NormType | NormModule | Callable[[jax.Array], jax.Array]
)

# TODO: AdaXNorm normalize complex
class AdaXNorm(nn.Module):
    """
    Normalize activations and project a conditioning tensor.

    ``norm`` may be ``"layernorm"``, ``"rmsnorm"``, or any module or
    callable that maps the input activation to an equally shaped array. The
    built-in normalizers are parameter-free and may reduce over any requested
    axes. The projected modulation is deliberately left unsplit so an
    architecture can interpret it as scale, shift, gate, or another signal.
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int | Sequence[int],
        norm: Normalizer = 'layernorm',
        eps: float = 1e-6,
        *,
        axes: Axes = -1,
        activation: Activation | None = 'silu',
        bias: bool = True,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        initializer: Initializer = default_linear_initializer,
        zero_init: bool = False,
        project: bool = True,
        quant: Any = None,
        dot_general: Any = None,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.embedding_dim = _validate_integer(embedding_dim, 'embedding_dim')
        self.out_dim = _normalize_shape(out_dim, 'out_dim')

        if isinstance(norm, str):
            normalized_name = norm.lower().replace('-', '_')
            aliases = {
                'layer': 'layernorm',
                'layer_norm': 'layernorm',
                'rms': 'rmsnorm',
                'rms_norm': 'rmsnorm',
            }
            normalized_name = aliases.get(normalized_name, normalized_name)
            if normalized_name not in {'layernorm', 'rmsnorm'}:
                raise ValueError(f'unsupported norm: {norm!r}')
            self.norm_type = normalized_name
            if normalized_name == 'layernorm':
                self.normalizer = nn.LayerNorm(
                    None,
                    eps=eps,
                    elementwise_affine=False,
                    bias=False,
                    axes=axes,
                    shard_mode=shard_mode,
                )
            else:
                self.normalizer = nn.RMSNorm(
                    None,
                    epsilon=eps,
                    with_scale=False,
                    axes=axes,
                    shard_mode=shard_mode,
                )
        elif isinstance(norm, nn.Module) or callable(norm):
            self.norm_type = 'custom'
            self.normalizer = norm
        else:
            raise TypeError('norm must be a supported string, module, or callable')

        self.eps = _validate_positive_float(eps, 'eps')
        self.axes = axes
        self.activation = _resolve_activation(activation, allow_none=True)
        self.shard_mode = shard_mode
        if not isinstance(zero_init, bool):
            raise TypeError('zero_init must be a bool')
        if not isinstance(project, bool):
            raise TypeError('project must be a bool')
        self.zero_init = zero_init
        self.project = project
        if project:
            if zero_init:
                initializer = jax.nn.initializers.zeros
            self.linear = nn.Linear(
                self.embedding_dim,
                self.out_dim,
                bias=bias,
                dtype=dtype,
                rngs=rngs,
                initializer=initializer,
                quant=quant,
                dot_general=dot_general,
                axis_names=axis_names,
                shard_mode=shard_mode,
            )
        else:
            if self.out_dim != (self.embedding_dim,):
                raise ValueError(
                    'out_dim must equal embedding_dim when project=False'
                )
            self.linear = None

    def _normalize(self, x: jax.Array) -> jax.Array:
        normalized = self.normalizer(x)
        if normalized.shape != x.shape:
            raise ValueError(
                'a normalizer must preserve the input shape; '
                f'got {x.shape} -> {normalized.shape}'
            )
        return normalized

    def __call__(
        self,
        x: jax.Array,
        conditioning: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
        modulation_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.asarray(x)
        conditioning = jnp.asarray(conditioning)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            raise TypeError('x must have a floating-point or complex dtype')
        if conditioning.shape[-1:] != (self.embedding_dim,):
            raise ValueError(
                'conditioning has an incompatible trailing dimension: '
                f'expected {self.embedding_dim}, got {conditioning.shape}'
            )

        normalized = _constrain(
            self._normalize(x),
            out_sharding,
            self.shard_mode,
        )
        if self.linear is None:
            modulation = _constrain(
                conditioning,
                modulation_sharding,
                self.shard_mode,
            )
        else:
            modulation = self.linear(
                self.activation(conditioning),
                out_sharding=modulation_sharding,
            )
        return normalized, modulation

    def extra_repr(self) -> str:
        output = 'x'.join(map(str, self.out_dim))
        projection = 'projected' if self.project else 'precomputed'
        return (
            f'{self.embedding_dim} -> {output}, norm={self.norm_type}, '
            f'{projection}'
        )


class SpatialNorm(nn.Module):
    """
    Apply group normalization modulated by a spatial conditioning tensor.

    Inputs use channels-last layout ``[batch, ..., channels]``. The conditioning
    tensor is resized to the feature tensor's non-channel dimensions, then two
    pointwise projections produce multiplicative and additive modulation. This
    formulation supports sequences, images, volumes, and higher-rank spatial
    arrays with the same parameters.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        *,
        num_groups: int = 32,
        eps: float = 1e-6,
        interpolation: str = 'nearest',
        affine: bool = True,
        bias: bool = True,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        initializer: Initializer = default_linear_initializer,
        quant: Any = None,
        dot_general: Any = None,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.f_channels = _validate_integer(f_channels, 'f_channels')
        self.zq_channels = _validate_integer(zq_channels, 'zq_channels')
        self.num_groups = _validate_integer(num_groups, 'num_groups')
        if self.f_channels % self.num_groups != 0:
            raise ValueError('f_channels must be divisible by num_groups')
        if not isinstance(interpolation, str):
            raise TypeError('interpolation must be a string')
        if axis_names is not None and len(axis_names) != 2:
            raise ValueError(
                'axis_names must contain conditioning and feature axes'
            )

        self.eps = _validate_positive_float(eps, 'eps')
        self.interpolation = interpolation
        self.shard_mode = shard_mode
        norm_axis_names = (
            None if axis_names is None else (axis_names[-1],)
        )
        self.norm_layer = nn.GroupNorm(
            self.num_groups,
            self.f_channels,
            eps=self.eps,
            affine=affine,
            bias=bias,
            dtype=dtype,
            axis_names=norm_axis_names,
            shard_mode=shard_mode,
        )
        projection_options = {
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'initializer': initializer,
            'quant': quant,
            'dot_general': dot_general,
            'axis_names': axis_names,
            'shard_mode': shard_mode,
        }
        self.scale = nn.Linear(
            self.zq_channels,
            self.f_channels,
            **projection_options,
        )
        self.shift = nn.Linear(
            self.zq_channels,
            self.f_channels,
            **projection_options,
        )

    def __call__(
        self,
        features: jax.Array,
        conditioning: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
        modulation_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        features = jnp.asarray(features)
        conditioning = jnp.asarray(conditioning)
        if features.ndim < 2 or conditioning.ndim != features.ndim:
            raise ValueError(
                'features and conditioning must have equal rank and use '
                '[batch, ..., channels] layout'
            )
        if features.shape[-1] != self.f_channels:
            raise ValueError(
                f'expected {self.f_channels} feature channels, '
                f'got {features.shape[-1]}'
            )
        if conditioning.shape[-1] != self.zq_channels:
            raise ValueError(
                f'expected {self.zq_channels} conditioning channels, '
                f'got {conditioning.shape[-1]}'
            )
        if conditioning.shape[0] != features.shape[0]:
            raise ValueError('features and conditioning must share a batch size')

        target_shape = (*features.shape[:-1], self.zq_channels)
        if conditioning.shape != target_shape:
            conditioning = jax.image.resize(
                conditioning,
                shape=target_shape,
                method=self.interpolation,
            )

        normalized = self.norm_layer(features)
        scale = self.scale(
            conditioning,
            out_sharding=modulation_sharding,
        )
        shift = self.shift(
            conditioning,
            out_sharding=modulation_sharding,
        )
        output = normalized * scale + shift
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'{self.f_channels}, condition={self.zq_channels}, '
            f'groups={self.num_groups}'
        )

__all__ = [
    'NormType',
    'NormModule',
    'Normalizer',
    'AdaXNorm',
    'SpatialNorm',
]
