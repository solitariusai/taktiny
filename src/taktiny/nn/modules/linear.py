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
from jax.sharding import PartitionSpec

import typing as tp
from collections.abc import Sequence
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import qwix
from jax.lax import PrecisionLike
from jax.nn.initializers import lecun_uniform
from jax.typing import DTypeLike

from taktiny.nn.base import Module, Parameter
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import _constrain, _normalize_shape
from taktiny.utils.quantization import (
    quantize_linear_weight,
    resolve_quantization_rule,
)
from taktiny.utils.spmd import with_logical_partitioning
from taktiny.utils.typing import ArrayLike, AxisNames, DType, Initializer, QuantConfig

default_kernel_initializer = lecun_uniform()
default_bias_initializer = jax.nn.initializers.zeros


class DotGeneral(Protocol):
    def __call__(
        self, 
        lhs: ArrayLike,
        rhs: ArrayLike,
        dimension_numbers: tuple[
            tuple[Sequence[int], Sequence[int]], 
            tuple[Sequence[int], Sequence[int]],
        ],
        precision: PrecisionLike,
        preferred_element_type: DTypeLike | None,
        *,
        out_sharding: Any,
    ) -> jax.Array:
        ...

class Linear(Module):
    """Applies a linear transformation to the incoming data.

    .. math::
        y = x W + b
    
    Args:
        in_features (int | tuple[int, ...]): Size of each input sample. Can be a tuple for higher dimensional contraction.
        out_features (int | tuple[int, ...]): Size of each output sample.
        rngs (Rngs | None): Random number generator for weight initialization. Required if weights are initialized.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        dtype (DType | None, optional): The data type of the computation. Defaults to 'float32'.
        kernel_initializer (Initializer, optional): Initializer function for the weight matrix. Defaults to lecun_uniform.
        bias_initializer (Initializer, optional): Initializer function for the bias vector. Defaults to zeros.
        quant (QuantConfig, optional): Optional quantization configuration for the weight. Defaults to None.
        dot_general (DotGeneral | None, optional): Optional custom dot_general implementation. Defaults to None.
        axis_names (AxisNames | None, optional): Logical axis names for parameter sharding metadata. Defaults to None.
        partition_spec (PartitionSpec | None, optional): Explicit hardware sharding specification. Defaults to None.
        kernel_metadata (dict[str, Any] | Sequence[tuple[str, Any]] | None, optional): Additional metadata for the kernel parameter. Defaults to None.
        bias_metadata (dict[str, Any] | Sequence[tuple[str, Any]] | None, optional): Additional metadata for the bias parameter. Defaults to None.
        precision (PrecisionLike, optional): Numerical precision for the dot product (e.g. jax.lax.Precision). Defaults to None.
        preferred_element_type (DTypeLike | None, optional): Preferred accumulation type for the dot product. Defaults to None.
        
    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from taktiny import nn

        # Create a linear layer
        linear = nn.Linear(64, 128, rngs=nn.Rngs(0))
        
        # Apply transformation
        x = jnp.ones((32, 64))
        y = linear(x)  # shape: (32, 128)
        ```
    """
    
    def __init__(
        self,
        in_features: int | tuple[int, ...],
        out_features: int | tuple[int, ...],
        *,
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_kernel_initializer,
        bias_initializer: Initializer = default_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: dict[str, Any] | Sequence[tuple[str, Any]] | None = None,
        bias_metadata: dict[str, Any] | Sequence[tuple[str, Any]] | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
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
        self.dot_general = dot_general
        self.precision = precision
        self.preferred_element_type = preferred_element_type

        kernel_shape = in_features + out_features
        if axis_names is not None or partition_spec is not None:
            kernel_initializer = with_logical_partitioning(kernel_initializer, axis_names, partition_spec)
            
        kernel_array = kernel_initializer(rngs(), kernel_shape, dtype)

        if quant is not None:
            rule = resolve_quantization_rule(
                quant, 
                '', 
                op_name='dot_general'
            )
            if rule is not None:
                kernel_array = quantize_linear_weight(
                    kernel_array,
                    rule,
                    input_axis_count=len(in_features),
                    batch_axis_count=0,
                )

        self.kernel = Parameter(
            kernel_array, 
            axis_names=axis_names, 
            partition_spec=partition_spec,
            metadata=kernel_metadata
        )

        self.bias = None
        if bias:
            bias_axis_names = None
            bias_partition_spec = None
            if axis_names is not None:
                bias_axis_names = axis_names[-len(out_features):]
                
            if partition_spec is not None:
                bias_partition_spec = partition_spec[-len(out_features):]
                
            if bias_axis_names is not None or bias_partition_spec is not None:
                bias_initializer = with_logical_partitioning(bias_initializer, bias_axis_names, bias_partition_spec)

            self.bias = Parameter(
                bias_initializer(rngs(), out_features, dtype),
                axis_names=bias_axis_names,
                partition_spec=bias_partition_spec,
                metadata=bias_metadata
            )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the linear transformation to the input.

        Args:
            x (jax.Array): The input array.
            out_sharding (jax.sharding.Sharding | None, optional): The sharding for the output array. Defaults to None.

        Returns:
            jax.Array: The transformed output array.
        """
        in_dims = len(self.in_features)
        x_contracting_dims = tuple(range(x.ndim - in_dims, x.ndim))
        weight_contracting_dims = tuple(range(in_dims))
        dimension_numbers = (
            (x_contracting_dims, weight_contracting_dims),
            ((), ()),
        )
        weight = self.kernel.value

        if isinstance(weight, qwix.QArray):
            out = qwix.dot_general(
                x, 
                weight, 
                dimension_numbers,
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
                out_sharding=out_sharding,
            )
        elif self.dot_general is not None:
            out = self.dot_general(
                x, 
                weight, 
                dimension_numbers,
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
                out_sharding=out_sharding,
            )
        else:
            out = jax.lax.dot_general(
                x,
                weight,
                dimension_numbers,
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
                out_sharding=out_sharding,
            )

        if self.bias is not None:
            out += self.bias.value

        return _constrain(out, out_sharding)

    def extra_repr(self) -> str:
        in_str = 'x'.join(map(str, self.in_features))
        out_str = 'x'.join(map(str, self.out_features))
        quantized = isinstance(self.kernel.value, qwix.QArray)
        quant_str = ' (Qwix PTQ)' if quantized else ''
        custom_dot = ' (custom dot_general)' if self.dot_general is not None else ''
        return f'{in_str} -> {out_str}{quant_str}{custom_dot}'


class Bilinear(Module):
    """_summary_

    Args:
        in1_features (int | tuple[int, ...]): _description_
        in2_features (int | tuple[int, ...]): _description_
        out_features (int | tuple[int, ...]): _description_
        rngs (Rngs): _description_
        bias (bool, optional): _description_. Defaults to True.
        dtype (DType | None, optional): _description_. Defaults to None.
        kernel_initializer (Initializer, optional): _description_. Defaults to default_kernel_initializer.
        bias_initializer (Initializer, optional): _description_. Defaults to default_bias_initializer.
        quant (QuantConfig, optional): _description_. Defaults to None.
        dot_general (DotGeneral | None, optional): _description_. Defaults to None.
        axis_names (AxisNames | None, optional): _description_. Defaults to None.
        partition_spec (PartitionSpec | None, optional): _description_. Defaults to None.
        kernel_metadata (dict[str, Any] | Sequence[tuple[str, Any]] | None, optional): _description_. Defaults to None.
        bias_metadata (dict[str, Any] | Sequence[tuple[str, Any]] | None, optional): _description_. Defaults to None.
        precision (PrecisionLike, optional): _description_. Defaults to None.
        preferred_element_type (DTypeLike | None, optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        in1_features: int | tuple[int, ...],
        in2_features: int | tuple[int, ...],
        out_features: int | tuple[int, ...],
        *,
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_kernel_initializer,
        bias_initializer: Initializer = default_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: dict[str, Any] | Sequence[tuple[str, Any]] | None = None,
        bias_metadata: dict[str, Any] | Sequence[tuple[str, Any]] | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.in1_features = _normalize_shape(in1_features, 'in1_features')
        self.in2_features = _normalize_shape(in2_features, 'in2_features')
        self.out_features = _normalize_shape(out_features, 'out_features')
        self.has_bias = bias
        self.dot_general = dot_general

        if rngs is None:
            raise ValueError('A rngs must be provided to initialize Bilinear layer')

        weight_shape = (
            self.in1_features
            + self.in2_features
            + self.out_features
        )
        if axis_names is not None or partition_spec is not None:
            kernel_initializer = with_logical_partitioning(kernel_initializer, axis_names, partition_spec)
        
        weight_array = kernel_initializer(rngs(), weight_shape, dtype)
        
        if quant is not None:
            from taktiny.utils.quantization import resolve_quantization_rule, quantize_linear_weight
            rule = resolve_quantization_rule(quant, '', op_name='dot_general')
            if rule is not None:
                weight_array = quantize_linear_weight(
                    weight_array,
                    rule,
                    input_axis_count=len(self.in1_features) + len(self.in2_features),
                    batch_axis_count=0,
                )

        self.weight = Parameter(
            weight_array,
            axis_names=axis_names,
            partition_spec=partition_spec,
            metadata=kernel_metadata
        )

        self.bias = None
        if bias:
            bias_axis_names = None
            bias_partition_spec = None
            if axis_names is not None:
                bias_axis_names = axis_names[-len(self.out_features):]
            if partition_spec is not None:
                bias_partition_spec = partition_spec[-len(self.out_features):]
                
            if bias_axis_names is not None or bias_partition_spec is not None:
                bias_initializer = with_logical_partitioning(bias_initializer, bias_axis_names, bias_partition_spec)

            self.bias = Parameter(
                bias_initializer(rngs(), self.out_features, dtype),
                axis_names=bias_axis_names,
                partition_spec=bias_partition_spec,
                metadata=bias_metadata
            )

    def __call__(
        self,
        x1: jax.Array,
        x2: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the bilinear transformation to the inputs.

        Args:
            x1 (jax.Array): The first input array.
            x2 (jax.Array): The second input array.
            out_sharding (jax.sharding.Sharding | None, optional): The sharding for the output array. Defaults to None.

        Returns:
            jax.Array: The transformed output array.
        """
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
        output = jax.lax.dot_general(
            intermediate,
            x1,
            second_dimension_numbers,
            out_sharding=out_sharding,
        )

        if self.has_bias:
            output += self.bias.value
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        in1 = 'x'.join(map(str, self.in1_features))
        in2 = 'x'.join(map(str, self.in2_features))
        output = 'x'.join(map(str, self.out_features))
        quantized = isinstance(self.weight.value, qwix.QArray)
        quant = ' (Qwix PTQ)' if quantized else ''
        custom_dot = ' (custom dot_general)' if self.dot_general is not None else ''
        return f'{in1}, {in2} -> {output}{quant}{custom_dot}'


__all__ = ['Bilinear', 'Linear']
