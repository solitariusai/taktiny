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
"""Linear modules."""
from __future__ import annotations
import typing as tp
import jax
import jax.numpy as jnp
import qwix
from jax.nn.initializers import lecun_uniform
import warnings

from taktiny.nn.module import Module, Parameter
from taktiny.nn.rng import Rngs
from taktiny.nn._continuo import _constrain, _normalize_shape
from taktiny.utils.typing import AxisNames, DType, Initializer, ShardMode

default_linear_initializer = lecun_uniform()


# Deprecated: Linear seed
class Linear(Module):
    """General linear projection with optional Qwix-quantized weights."""
    def __init__(
        self,
        in_features: int | tuple[int, ...],
        out_features: int | tuple[int, ...],
        *,
        bias: bool = True,
        dtype: tp.Optional[DType] = jnp.float32,
        rngs: Rngs | None = None,
        seed: Rngs | None = None,
        initializer: Initializer = default_linear_initializer,
        quant: tp.Any = None,
        dot_general: tp.Any = None,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if isinstance(in_features, int):
            in_features = (in_features,)
        else:
            in_features = tuple(in_features)

        if isinstance(out_features, int):
            out_features = (out_features,)
        else:
            out_features = tuple(out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.dot_general = dot_general
        self.shard_mode = shard_mode

        if rngs is None and seed is None:
            raise ValueError('A rngs must be provided to initialize Linear layer')
        if rngs is None:
            warnings.warn('seed is deprecated. use `rngs` instead')
            rngs = seed

        weight_shape = in_features + out_features
        self.weight = Parameter(
            initializer(rngs(), weight_shape, dtype)
        )
        self.weight.quantization = quant
        self.weight.input_axis_count = len(in_features)
        self.weight.quantization_batch_axis_count = 0

        if axis_names is not None:
            if len(axis_names) != len(weight_shape):
                raise ValueError(
                    f'axis_names length {len(axis_names)} must match '
                    f'weight dimensions {len(weight_shape)}'
                )
            self.weight.axis_names = axis_names

        if bias:
            self.bias = Parameter(jnp.zeros(out_features, dtype=dtype))
            if axis_names is not None:
                self.bias.axis_names = axis_names[-len(out_features):]

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        in_dims = len(self.in_features)
        x_contracting_dims = tuple(range(x.ndim - in_dims, x.ndim))
        weight_contracting_dims = tuple(range(in_dims))
        dimension_numbers = (
            (x_contracting_dims, weight_contracting_dims),
            ((), ()),
        )
        weight = self.weight.value

        explicit_out_sharding = (
            out_sharding
            if self.shard_mode == ShardMode.EXPLICIT
            else None
        )

        if isinstance(weight, qwix.QArray):
            out = qwix.dot_general(x, weight, dimension_numbers)
        elif self.dot_general is not None:
            out = self.dot_general(x, weight, dimension_numbers)
        else:
            out = jax.lax.dot_general(
                x,
                weight,
                dimension_numbers,
                out_sharding=explicit_out_sharding,
            )

        if self.has_bias:
            out += self.bias.value

        return _constrain(out, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        in_str = 'x'.join(map(str, self.in_features))
        out_str = 'x'.join(map(str, self.out_features))
        quantized = (
            isinstance(self.weight.value, qwix.QArray)
            or self.weight.quantization is not None
        )
        quant_str = ' (Qwix PTQ)' if quantized else ''
        custom_dot = ' (custom dot_general)' if self.dot_general is not None else ''
        return f'{in_str} -> {out_str}{quant_str}{custom_dot}'


class Bilinear(Module):
    """Apply a bilinear projection to two arrays with shared leading axes.

    The trailing feature shapes of ``x1`` and ``x2`` are contracted against
    the first and second input feature groups of ``weight`` respectively.
    """

    def __init__(
        self,
        in1_features: int | tuple[int, ...],
        in2_features: int | tuple[int, ...],
        out_features: int | tuple[int, ...],
        *,
        bias: bool = True,
        dtype: tp.Optional[DType] = jnp.float32,
        rngs: Rngs | None = None,
        seed: Rngs | None = None,
        initializer: Initializer = default_linear_initializer,
        quant: tp.Any = None,
        dot_general: tp.Any = None,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.in1_features = _normalize_shape(in1_features, 'in1_features')
        self.in2_features = _normalize_shape(in2_features, 'in2_features')
        self.out_features = _normalize_shape(out_features, 'out_features')
        self.has_bias = bias
        self.dot_general = dot_general
        self.shard_mode = shard_mode

        if rngs is None and seed is None:
            raise ValueError('A rngs must be provided to initialize Bilinear layer')
        if rngs is None:
            warnings.warn('seed is deprecated. use `rngs` instead')
            rngs = seed

        weight_shape = (
            self.in1_features
            + self.in2_features
            + self.out_features
        )
        self.weight = Parameter(initializer(rngs(), weight_shape, dtype))
        self.weight.quantization = quant
        self.weight.input_axis_count = (
            len(self.in1_features) + len(self.in2_features)
        )
        self.weight.quantization_batch_axis_count = 0

        if axis_names is not None:
            if len(axis_names) != len(weight_shape):
                raise ValueError(
                    f'axis_names length {len(axis_names)} must match '
                    f'weight dimensions {len(weight_shape)}'
                )
            self.weight.axis_names = axis_names

        if bias:
            self.bias = Parameter(jnp.zeros(self.out_features, dtype=dtype))
            if axis_names is not None:
                self.bias.axis_names = axis_names[-len(self.out_features):]

    def __call__(
        self,
        x1: jax.Array,
        x2: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        in1_dims = len(self.in1_features)
        in2_dims = len(self.in2_features)
        if x1.ndim < in1_dims or x2.ndim < in2_dims:
            raise ValueError('inputs have fewer axes than their feature shapes')
        if x1.shape[-in1_dims:] != self.in1_features:
            raise ValueError(
                f'x1 trailing shape must be {self.in1_features}, '
                f'got {x1.shape[-in1_dims:]}'
            )
        if x2.shape[-in2_dims:] != self.in2_features:
            raise ValueError(
                f'x2 trailing shape must be {self.in2_features}, '
                f'got {x2.shape[-in2_dims:]}'
            )

        x1_leading_shape = x1.shape[:-in1_dims]
        x2_leading_shape = x2.shape[:-in2_dims]
        if x1_leading_shape != x2_leading_shape:
            raise ValueError(
                'x1 and x2 must have identical leading shapes, got '
                f'{x1_leading_shape} and {x2_leading_shape}'
            )

        weight = self.weight.value
        first_dimension_numbers = (
            (
                tuple(range(x2.ndim - in2_dims, x2.ndim)),
                tuple(range(in1_dims, in1_dims + in2_dims)),
            ),
            ((), ()),
        )
        if isinstance(weight, qwix.QArray):
            intermediate = qwix.dot_general(
                x2,
                weight,
                first_dimension_numbers,
            )
        elif self.dot_general is not None:
            intermediate = self.dot_general(
                x2,
                weight,
                first_dimension_numbers,
            )
        else:
            intermediate = jax.lax.dot_general(
                x2,
                weight,
                first_dimension_numbers,
            )

        leading_dims = len(x1_leading_shape)
        second_dimension_numbers = (
            (
                tuple(range(leading_dims, leading_dims + in1_dims)),
                tuple(range(x1.ndim - in1_dims, x1.ndim)),
            ),
            (
                tuple(range(leading_dims)),
                tuple(range(leading_dims)),
            ),
        )
        explicit_out_sharding = (
            out_sharding
            if self.shard_mode == ShardMode.EXPLICIT
            else None
        )
        output = jax.lax.dot_general(
            intermediate,
            x1,
            second_dimension_numbers,
            out_sharding=explicit_out_sharding,
        )

        if self.has_bias:
            output += self.bias.value
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        in1 = 'x'.join(map(str, self.in1_features))
        in2 = 'x'.join(map(str, self.in2_features))
        output = 'x'.join(map(str, self.out_features))
        quantized = (
            isinstance(self.weight.value, qwix.QArray)
            or self.weight.quantization is not None
        )
        quant = ' (Qwix PTQ)' if quantized else ''
        custom_dot = ' (custom dot_general)' if self.dot_general is not None else ''
        return f'{in1}, {in2} -> {output}{quant}{custom_dot}'


__all__ = ['Linear', 'Bilinear']
