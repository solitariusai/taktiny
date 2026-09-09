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

import jax
import qwix
from jax.lax import PrecisionLike
from jax.nn.initializers import lecun_uniform
from jax.sharding import PartitionSpec
from jax.typing import DTypeLike

from taktiny.nn.base import Module, Parameter
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import _constrain, _normalize_shape
from taktiny.utils.quantization import (
    quantize_linear_weight,
    resolve_quantization_rule,
)
from taktiny.utils.spmd import with_logical_partitioning
from taktiny.utils.typing import (
    AxisNames,
    DotGeneral,
    DType,
    GenericShape,
    Initializer,
    MetaData,
    QuantConfig,
)

default_kernel_initializer = lecun_uniform()
default_bias_initializer = jax.nn.initializers.zeros


class Linear(Module):
    """Applies a linear transformation to the input.

    .. math::

        Y = XW + b

    Here, ``W`` is the learnable kernel and ``b`` is the optional learnable
    bias. For one-dimensional features, an input of shape
    ``(..., in_features)`` produces an output of shape
    ``(..., out_features)``.

    When feature shapes are tuples, the same operation contracts all trailing
    input-feature axes: a kernel of shape
    ``(*in_features, *out_features)`` maps ``(..., *in_features)`` to
    ``(..., *out_features)``. The bias term is omitted when ``bias=False``.

    Args:
        in_features: Size of the trailing input feature axes. An integer is
            treated as a one-dimensional feature shape.
        out_features: Size of the output feature axes. An integer is treated
            as a one-dimensional feature shape.
        bias: Whether to create and add a learnable bias. Defaults to ``True``.
        dtype: Data type passed to the parameter initializers. Defaults to the
            initializer's default data type.
        rngs: Random number generator used to initialize the kernel and bias.
        kernel_initializer: Function used to initialize the kernel. Defaults
            to LeCun uniform initialization.
        bias_initializer: Function used to initialize the bias. Defaults to
            zeros.
        quant: Optional Qwix quantization configuration for the kernel.
        dot_general: Optional implementation of ``dot_general``. It is used
            for non-quantized kernels instead of ``jax.lax.dot_general``.
        axis_names: Optional logical axis names for the kernel. The names of
            the output axes are also assigned to the bias.
        partition_spec: Optional partition specification for the kernel. The
            specification for the output axes is also assigned to the bias.
        kernel_metadata: Optional metadata attached to the kernel parameter.
        bias_metadata: Optional metadata attached to the bias parameter.
        precision: Dot-product precision forwarded to ``dot_general``.
        preferred_element_type: Preferred result and accumulation data type
            forwarded to ``dot_general``.

    Attributes:
        kernel: The learnable kernel parameter.
        bias: The learnable bias parameter, or ``None`` when bias is disabled.

    Examples:
        Apply a conventional projection to the last axis:

        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> linear = nn.Linear(3, 2, rngs=nn.Rngs(0))
        >>> x = jnp.ones((4, 3))
        >>> linear(x).shape
        (4, 2)

        Contract multiple input axes and produce multiple output axes:

        >>> linear = nn.Linear((2, 3), (4, 5), rngs=nn.Rngs(1))
        >>> x = jnp.ones((8, 2, 3))
        >>> linear(x).shape
        (8, 4, 5)
    """

    def __init__(
        self,
        in_features: GenericShape,
        out_features: GenericShape,
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
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.in_features = _normalize_shape(in_features, 'in_features')
        self.out_features = _normalize_shape(out_features, 'out_features')
        self.dot_general = dot_general
        self.precision = precision
        self.preferred_element_type = preferred_element_type

        kernel_shape = self.in_features + self.out_features
        if axis_names is not None or partition_spec is not None:
            kernel_initializer = with_logical_partitioning(
                kernel_initializer,
                axis_names,
                partition_spec
            )

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
                    input_axis_count=len(self.in_features),
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
                bias_axis_names = axis_names[-len(self.out_features):]

            if partition_spec is not None:
                bias_partition_spec = PartitionSpec(
                    *partition_spec[-len(self.out_features):]
                )

            if bias_axis_names is not None or bias_partition_spec is not None:
                bias_initializer = with_logical_partitioning(
                    bias_initializer,
                    bias_axis_names,
                    bias_partition_spec
                )

            self.bias = Parameter(
                bias_initializer(rngs(), self.out_features, dtype),
                axis_names=bias_axis_names,
                partition_spec=bias_partition_spec,
                metadata=bias_metadata
            )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the linear transformation to an input array.

        Args:
            x: Input whose trailing dimensions match ``in_features``.
            out_sharding: Optional sharding for the result. This is forwarded
                to ``dot_general`` and applied to the final, bias-adjusted
                output as a sharding constraint.

        Returns:
            An array with the input's leading dimensions followed by
            ``out_features``.
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
            out += self.bias

        return _constrain(out, out_sharding)

    def extra_repr(self) -> str:
        in_str = '×'.join(map(str, self.in_features))
        out_str = '×'.join(map(str, self.out_features))
        quantized = isinstance(self.kernel.value, qwix.QArray)
        quant_str = ' (Qwix PTQ)' if quantized else ''
        custom_dot = ' (custom dot_general)' if self.dot_general is not None else ''
        return f'{in_str} ➤ {out_str}{quant_str}{custom_dot}'

class Bilinear(Module):
    r"""Applies a bilinear transformation to two inputs.

    .. math::

        y_k = x_1^{\mathsf{T}} W_k x_2 + b_k

    Here, ``W_k`` is the learnable kernel for output feature ``k`` and ``b_k``
    is the optional bias. For one-dimensional features, inputs of shape
    ``(..., in1_features)`` and ``(..., in2_features)`` produce an output of
    shape ``(..., out_features)``.

    When feature shapes are tuples, the same operation contracts all trailing
    feature axes. The kernel has shape
    ``(*in1_features, *in2_features, *out_features)``.

    Args:
        in1_features: Size of the trailing feature axes of the first input. An
            integer is treated as a one-dimensional feature shape.
        in2_features: Size of the trailing feature axes of the second input.
            An integer is treated as a one-dimensional feature shape.
        out_features: Size of the output feature axes. An integer is treated
            as a one-dimensional feature shape.
        bias: Whether to create and add a learnable bias. Defaults to ``True``.
        dtype: Data type passed to the parameter initializers. Defaults to the
            initializer's default data type.
        rngs: Random number generator used to initialize the kernel and bias.
        kernel_initializer: Function used to initialize the kernel. Defaults
            to LeCun uniform initialization.
        bias_initializer: Function used to initialize the bias. Defaults to
            zeros.
        quant: Optional Qwix quantization configuration for the kernel.
        dot_general: Optional implementation of ``dot_general``. It is used
            for non-quantized kernels instead of ``jax.lax.dot_general``.
        axis_names: Optional logical axis names for the kernel. The names of
            the output axes are also assigned to the bias.
        partition_spec: Optional partition specification for the kernel. The
            specification for the output axes is also assigned to the bias.
        kernel_metadata: Optional metadata attached to the kernel parameter.
        bias_metadata: Optional metadata attached to the bias parameter.
        precision: Dot-product precision forwarded to ``dot_general``.
        preferred_element_type: Preferred result and accumulation data type
            forwarded to ``dot_general``.

    Attributes:
        kernel: The learnable bilinear kernel parameter.
        bias: The learnable bias parameter, or ``None`` when bias is disabled.

    Examples:
        Apply a bilinear transformation to two batches of vectors:

        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> bilinear = nn.Bilinear(3, 4, 2, rngs=nn.Rngs(0))
        >>> x1 = jnp.ones((5, 3))
        >>> x2 = jnp.ones((5, 4))
        >>> bilinear(x1, x2).shape
        (5, 2)

        Contract multi-axis feature shapes:

        >>> bilinear = nn.Bilinear(
        ...     (2, 3), (4, 5), (6, 7), rngs=nn.Rngs(1)
        ... )
        >>> x1 = jnp.ones((8, 2, 3))
        >>> x2 = jnp.ones((8, 4, 5))
        >>> bilinear(x1, x2).shape
        (8, 6, 7)
    """

    def __init__(
        self,
        in1_features: GenericShape,
        in2_features: GenericShape,
        out_features: GenericShape,
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
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.in1_features = _normalize_shape(in1_features, 'in1_features')
        self.in2_features = _normalize_shape(in2_features, 'in2_features')
        self.out_features = _normalize_shape(out_features, 'out_features')
        self.has_bias = bias
        self.dot_general = dot_general
        self.precision = precision
        self.preferred_element_type = preferred_element_type

        kernel_shape = (
            self.in1_features
            + self.in2_features
            + self.out_features
        )
        if axis_names is not None or partition_spec is not None:
            kernel_initializer = with_logical_partitioning(
                kernel_initializer,
                axis_names,
                partition_spec
            )

        kernel_array = kernel_initializer(rngs(), kernel_shape, dtype)
        if quant is not None:
            rule = resolve_quantization_rule(quant, '', op_name='dot_general')
            if rule is not None:
                kernel_array = quantize_linear_weight(
                    kernel_array,
                    rule,
                    input_axis_count=len(self.in1_features) + len(self.in2_features),
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
                bias_axis_names = axis_names[-len(self.out_features):]

            if partition_spec is not None:
                bias_partition_spec = PartitionSpec(
                    *partition_spec[-len(self.out_features):]
                )

            if bias_axis_names is not None or bias_partition_spec is not None:
                bias_initializer = with_logical_partitioning(
                    bias_initializer,
                    bias_axis_names,
                    bias_partition_spec
                )

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
        """Applies the bilinear transformation to two input arrays.

        Args:
            x1: First input, with trailing dimensions matching
                ``in1_features``.
            x2: Second input, with trailing dimensions matching
                ``in2_features`` and leading dimensions matching ``x1``.
            out_sharding: Optional sharding for the result. This is forwarded
                to the final ``dot_general`` operation and applied to the
                bias-adjusted output as a sharding constraint.

        Returns:
            An array with the shared leading dimensions of the inputs followed
            by ``out_features``.

        Raises:
            ValueError: If either trailing feature shape is invalid or the
                inputs have different leading shapes.
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

        kernel_array = self.kernel.value
        first_dimension_numbers = (
            (
                tuple(range(x2.ndim - in2_dims, x2.ndim)),
                tuple(range(in1_dims, in1_dims + in2_dims)),
            ),
            ((), ()),
        )
        if isinstance(kernel_array, qwix.QArray):
            intermediate = qwix.dot_general(
                x2,
                kernel_array,
                first_dimension_numbers,
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
            )
        elif self.dot_general is not None:
            intermediate = self.dot_general(
                x2,
                kernel_array,
                first_dimension_numbers,
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
            )
        else:
            intermediate = jax.lax.dot_general(
                x2,
                kernel_array,
                first_dimension_numbers,
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
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
        if isinstance(intermediate, qwix.QArray):
            output = qwix.dot_general(
                intermediate,
                x1,
                second_dimension_numbers,
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
                out_sharding=out_sharding,
            )
        elif self.dot_general is not None:
            output = self.dot_general(
                intermediate,
                x1,
                second_dimension_numbers,
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
                out_sharding=out_sharding,
            )
        else:
            output = jax.lax.dot_general(
                intermediate,
                x1,
                second_dimension_numbers,
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
                out_sharding=out_sharding,
            )

        if self.bias is not None:
            output += self.bias

        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        in1 = '×'.join(map(str, self.in1_features))
        in2 = '×'.join(map(str, self.in2_features))
        output = '×'.join(map(str, self.out_features))
        quantized = isinstance(self.kernel.value, qwix.QArray)
        quant = ' (Qwix PTQ)' if quantized else ''
        custom_dot = ' (custom dot_general)' if self.dot_general is not None else ''
        return f'{in1}, {in2} ➤ {output}{quant}{custom_dot}'


__all__ = ['Bilinear', 'Linear']
