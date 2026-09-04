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
"""Convolution modules"""
from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

import jax
import jax.numpy as jnp
import qwix
from jax.lax import PrecisionLike
from jax.nn.initializers import lecun_uniform, zeros
from jax.sharding import PartitionSpec
from jax.typing import DTypeLike

from taktiny.nn.base import Module, Parameter
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import (
    _adaptive_pool,
    _as_batched,
    _canonical_padding,
    _constrain,
    _conv_dimension_numbers,
    _max_identity,
    _normalize_adaptive_size,
    _normalize_nonnegative,
    _normalize_shape,
    _pool_padding,
    _reduce_window_config,
    _restore_batch,
    _scatter_indices,
    _window_output_shape,
)
from taktiny.utils.quantization import (
    quantize_conv_weight,
    resolve_quantization_rule,
)
from taktiny.utils.spmd import with_logical_partitioning
from taktiny.utils.typing import (
    AxisNames,
    ConvGeneralDilated,
    DType,
    GenericShape,
    Initializer,
    MetaData,
    QuantConfig,
)

default_kernel_initializer = lecun_uniform()
default_bias_initializer = jax.nn.initializers.zeros
# Kept for modules that have not migrated to the new initializer name yet.
default_conv_initializer = default_kernel_initializer


class Conv(Module):
    """Applies an N-dimensional convolution to channels-last inputs.

    The spatial rank is inferred from ``kernel_size``. Scalar channel counts
    behave like a conventional convolution. Tuple-shaped channels are stored
    as structured trailing axes and flattened only for the underlying JAX
    convolution. Thus an input of shape
    ``[batch, *spatial, *in_channels]`` produces
    ``[batch, *output_spatial, *out_channels]``.

    When ``groups`` is greater than one, groups partition the first input and
    output channel axes. Both first channel-axis sizes must therefore be
    divisible by ``groups``.

    ``padding`` and ``pad_mode`` control different aspects of boundary
    handling. ``padding`` determines how many elements are added before and
    after each spatial axis, while ``pad_mode`` determines the values used for
    those elements. With the default ``pad_mode='zeros'``, padding is handled
    directly by the convolution and the added values are zero. The other
    modes explicitly extend the input before applying a ``VALID`` convolution:
    ``'reflect'`` mirrors the input without repeating its edge, ``'replicate'``
    repeats the edge value, and ``'circular'`` wraps values from the opposite
    edge.

    ``padding`` accepts ``'VALID'`` for no automatic padding and ``'SAME'`` or
    ``'SAME_LOWER'`` for the padding required to produce
    ``ceil(input_size / stride)`` positions along each spatial axis. When the
    total padding is odd, ``'SAME'`` places the extra element after the input,
    while ``'SAME_LOWER'`` places it before the input. String padding is
    supported only with ``pad_mode='zeros'``. Numeric padding can be expressed
    in several ways:

    - An integer applies that amount symmetrically to every spatial axis.
    - A sequence of integers supplies one symmetric amount per spatial axis.
    - A sequence of ``(before, after)`` pairs supplies asymmetric padding for
      every spatial axis.
    - For a one-dimensional convolution, a two-integer sequence is interpreted
      directly as ``(before, after)``.

    For example, ``padding=2`` pads every axis by two elements on each side;
    for a two-dimensional convolution, ``padding=(1, 2)`` pads the first axis
    by one and the second by two on each side, while
    ``padding=((1, 0), (2, 3))`` specifies every side independently.

    Args:
        in_channels: Shape of the trailing input-channel axes.
        out_channels: Shape of the trailing output-channel axes.
        kernel_size: Size of the spatial convolution window.
        stride: Step of the convolution window.
        padding: Spatial padding geometry. Use ``'VALID'`` for no automatic
            padding, ``'SAME'`` or ``'SAME_LOWER'`` to preserve the
            stride-scaled spatial size, a non-negative integer for symmetric
            padding on every axis, one integer per axis for per-axis symmetric
            padding, or a sequence of ``n`` ``(before, after)`` pairs—one for
            each spatial axis—for asymmetric padding. Defaults to ``0``.
        dilation: Spacing between kernel elements.
        groups: Number of feature groups.
        pad_mode: How values outside the input boundary are produced. One of
            ``'zeros'``, ``'reflect'``, ``'replicate'``, or ``'circular'``.
            Nonzero modes require explicit numeric ``padding``; ``'SAME'``,
            ``'SAME_LOWER'``, and ``'VALID'`` can only be used with ``'zeros'``.
            Defaults to ``'zeros'``.
        bias: Whether to add a learnable output bias.
        dtype: Data type passed to the parameter initializers.
        rngs: Random number generator used to initialize parameters.
        kernel_initializer: Function used to initialize the kernel.
        bias_initializer: Function used to initialize the bias.
        quant: Optional Qwix quantization configuration for the kernel.
        dot_general: Optional drop-in convolution callable. The name is kept
            for compatibility with other parameterized modules.
        axis_names: Optional logical names for every kernel axis.
        partition_spec: Optional partition specification for the kernel.
        kernel_metadata: Optional metadata attached to the kernel parameter.
        bias_metadata: Optional metadata attached to the bias parameter.
        precision: Convolution precision forwarded to the convolution callable.
        preferred_element_type: Preferred accumulation and result data type.

    Examples:
        Apply a one-dimensional convolution to a channels-last batch while
        preserving its spatial length:

        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> conv = nn.Conv(
        ...     3, 8, kernel_size=3, padding='SAME', rngs=nn.Rngs(0)
        ... )
        >>> x = jnp.ones((4, 16, 3))
        >>> conv(x).shape
        (4, 16, 8)

        Structured channel shapes remain visible in both the input and output.
        This example applies an unbatched two-dimensional convolution:

        >>> conv = nn.Conv(
        ...     (2, 3),
        ...     (4, 5),
        ...     kernel_size=(3, 3),
        ...     padding='SAME',
        ...     rngs=nn.Rngs(1),
        ... )
        >>> x = jnp.ones((8, 8, 2, 3))
        >>> conv(x).shape
        (8, 8, 4, 5)

        Nonzero boundary modes require explicit numeric padding. Here the
        spatial input is reflected by one element on each side:

        >>> conv = nn.Conv(
        ...     1,
        ...     4,
        ...     kernel_size=3,
        ...     padding=1,
        ...     pad_mode='reflect',
        ...     rngs=nn.Rngs(2),
        ... )
        >>> x = jnp.ones((6, 1))
        >>> conv(x).shape
        (6, 4)
    """

    def __init__(
        self,
        in_channels: GenericShape,
        out_channels: GenericShape,
        kernel_size: GenericShape,
        *,
        stride: GenericShape = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        dilation: GenericShape = 1,
        groups: int = 1,
        pad_mode: str = 'zeros',
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_kernel_initializer,
        bias_initializer: Initializer = default_bias_initializer,
        quant: QuantConfig = None,
        dot_general: ConvGeneralDilated | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        in_channels = _normalize_shape(in_channels, 'in_channels')
        out_channels = _normalize_shape(out_channels, 'out_channels')

        if not isinstance(groups, int) or groups <= 0:
            raise ValueError('groups must be a positive integer')

        if in_channels[0] % groups != 0:
            raise ValueError(
                f'in_channels[0] ({in_channels[0]}) must be divisible by groups '
                f'({groups})'
            )

        if out_channels[0] % groups != 0:
            raise ValueError(
                f'out_channels ({out_channels}) must be divisible by groups '
                f'({groups})'
            )

        kernel_size = self._normalize_spatial(kernel_size, name='kernel_size')
        spatial_rank = len(kernel_size)
        stride = self._normalize_spatial(
            stride,
            rank=spatial_rank,
            name='stride',
        )
        dilation = self._normalize_spatial(
            dilation,
            rank=spatial_rank,
            name='dilation',
        )
        padding = self._normalize_padding(padding, spatial_rank)

        pad_mode = pad_mode.lower()
        pad_modes = {
            'zeros': 'constant',
            'reflect': 'reflect',
            'replicate': 'edge',
            'circular': 'wrap',
        }
        if pad_mode not in pad_modes:
            choices = ', '.join(pad_modes)
            raise ValueError(
                f'pad_mode must be one of {choices}, got {pad_mode!r}'
            )
        if pad_mode != 'zeros' and isinstance(padding, str):
            raise ValueError(
                'nonzero padding modes require explicit numeric padding'
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self.pad_mode = pad_mode
        self.spatial_rank = spatial_rank
        self.dot_general = dot_general
        self.precision = precision
        self.preferred_element_type = preferred_element_type
        self._in_channel_count = math.prod(in_channels)
        self._out_channel_count = math.prod(out_channels)

        grouped_in_channels = (
            in_channels[0] // groups,
            *in_channels[1:],
        )
        weight_shape = kernel_size + grouped_in_channels + out_channels
        if axis_names is not None:
            axis_names = tuple(axis_names)

        if axis_names is not None or partition_spec is not None:
            kernel_initializer = with_logical_partitioning(
                kernel_initializer,
                axis_names,
                partition_spec,
            )

        kernel_array = kernel_initializer(rngs(), weight_shape, dtype)
        if quant is not None:
            rule = resolve_quantization_rule(
                quant,
                '',
                op_name='conv_general_dilated',
            )
            if rule is not None:
                kernel_array = quantize_conv_weight(
                    kernel_array,
                    rule,
                    output_axis_count=len(out_channels),
                )

        self.kernel = Parameter(
            kernel_array,
            axis_names=axis_names,
            partition_spec=partition_spec,
            metadata=kernel_metadata,
        )

        self.bias = None
        if bias:
            bias_axis_names = None
            bias_partition_spec = None
            if axis_names is not None:
                bias_axis_names = axis_names[-len(out_channels):]
            if partition_spec is not None:
                bias_partition_spec = PartitionSpec(
                    *partition_spec[-len(out_channels):]
                )
            if bias_axis_names is not None or bias_partition_spec is not None:
                bias_initializer = with_logical_partitioning(
                    bias_initializer,
                    bias_axis_names,
                    bias_partition_spec,
                )

            self.bias = Parameter(
                bias_initializer(rngs(), out_channels, dtype),
                axis_names=bias_axis_names,
                partition_spec=bias_partition_spec,
                metadata=bias_metadata,
            )

    @staticmethod
    def _normalize_spatial(
        value: int | Sequence[int],
        *,
        rank: int | None = None,
        name: str,
    ) -> tuple[int, ...]:
        """Normalizes a spatial argument into a tuple of integers.

        Args:
            value (int | Sequence[int]): The value to normalize.
            name (str): The name of the argument (used for error messages).
            rank (int | None, optional): The expected spatial rank. Defaults to None.

        Returns:
            tuple[int, ...]: The normalized spatial argument.
        """
        if isinstance(value, int):
            values = (value,) if rank is None else (value,) * rank
        else:
            values = tuple(value)

        if not values:
            raise ValueError(f'{name} must contain at least one dimension')

        if rank is not None and len(values) != rank:
            raise ValueError(
                f'{name} must contain {rank} values, got {len(values)}'
            )

        if any(not isinstance(item, int) or item <= 0 for item in values):
            raise ValueError(f'{name} values must be positive integers')

        return values

    @staticmethod
    def _normalize_padding(
        padding: str | int | Sequence[int | tuple[int, int]],
        rank: int,
    ) -> str | tuple[tuple[int, int], ...]:
        """Normalizes the padding argument into a canonical form.

        Args:
            padding (str | int | Sequence[int | tuple[int, int]]): The padding to normalize.
            rank (int): The spatial rank.

        Returns:
            str | tuple[tuple[int, int], ...]: The normalized padding.
        """
        if isinstance(padding, str):
            padding = padding.upper()
            if padding not in {'SAME', 'SAME_LOWER', 'VALID'}:
                raise ValueError(
                    "padding must be 'SAME', 'SAME_LOWER', 'VALID', or "
                    'explicit integers'
                )
            return padding

        if isinstance(padding, int):
            pairs = ((padding, padding),) * rank
        else:
            values = tuple(padding)
            if rank == 1 and len(values) == 2:
                low, high = values
                if isinstance(low, int) and isinstance(high, int):
                    pairs = ((low, high),)
                else:
                    raise ValueError(
                        'padding must describe 1 spatial dimension'
                    )

            elif len(values) != rank:
                raise ValueError(
                    f'padding must describe {rank} spatial dimensions'
                )

            elif isinstance(values[0], int):
                symmetric_pairs: list[tuple[int, int]] = []
                for value in values:
                    if not isinstance(value, int):
                        raise TypeError(
                            'padding values must either all be integers or '
                            'all be (before, after) pairs'
                        )
                    symmetric_pairs.append((value, value))
                pairs = tuple(symmetric_pairs)

            else:
                explicit_pairs: list[tuple[int, int]] = []
                for value in values:
                    if not isinstance(value, Sequence):
                        raise TypeError(
                            'padding values must either all be integers or '
                            'all be (before, after) pairs'
                        )
                    sides = tuple(value)
                    if len(sides) != 2:
                        raise ValueError(
                            'each padding pair must contain two integers'
                        )
                    low, high = sides
                    if not isinstance(low, int) or not isinstance(high, int):
                        raise TypeError(
                            'each padding pair must contain two integers'
                        )
                    explicit_pairs.append((low, high))
                pairs = tuple(explicit_pairs)

        if any(side < 0 for pair in pairs for side in pair):
            raise ValueError('padding values must be non-negative')

        return pairs

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the convolution to a batched or unbatched input.

        Args:
            x: Channels-last input whose trailing axes match ``in_channels``.
            out_sharding: Optional sharding constraint for the final output.

        Returns:
            The convolved array with trailing axes matching ``out_channels``.

        Raises:
            ValueError: If the input rank or channels are invalid, or if the
                kernel and padding would produce an empty spatial output.
        """
        expected_batched_rank = (
            self.spatial_rank
            + len(self.in_channels)
            + 1
        )
        if x.ndim not in {expected_batched_rank - 1, expected_batched_rank}:
            raise ValueError(
                f'expected an unbatched rank-{expected_batched_rank - 1} or '
                f'batched rank-{expected_batched_rank} input, got rank {x.ndim}'
            )
        if x.shape[-len(self.in_channels):] != self.in_channels:
            raise ValueError(
                f'expected trailing input channels {self.in_channels}, got '
                f'{x.shape[-len(self.in_channels):]}'
            )

        unbatched = x.ndim == expected_batched_rank - 1
        if unbatched:
            x = x[None, ...]

        padding = self.padding
        input_spatial_shape = x.shape[1:self.spatial_rank + 1]
        explicit_padding = _canonical_padding(
            padding,
            input_spatial_shape,
            self.kernel_size,
            self.stride,
            self.dilation,
        )
        try:
            _window_output_shape(
                input_spatial_shape,
                self.kernel_size,
                self.stride,
                self.dilation,
                explicit_padding,
            )
        except ValueError as error:
            effective_kernel = tuple(
                spacing * (size - 1) + 1
                for size, spacing in zip(self.kernel_size, self.dilation)
            )
            raise ValueError(
                f'input spatial shape {input_spatial_shape} is too small for '
                f'effective kernel shape {effective_kernel} with padding '
                f'{explicit_padding}'
            ) from error

        if self.pad_mode != 'zeros':
            if isinstance(padding, str):
                raise ValueError(
                    'nonzero padding modes require explicit numeric padding'
                )
            pad_width = (
                ((0, 0),)
                + padding
                + ((0, 0),) * len(self.in_channels)
            )
            mode = {
                'reflect': 'reflect',
                'replicate': 'edge',
                'circular': 'wrap',
            }[self.pad_mode]
            x = jnp.pad(x, pad_width, mode=mode)
            padding = 'VALID'

        x = x.reshape(
            *x.shape[:self.spatial_rank + 1],
            self._in_channel_count,
        )

        lhs_spec = (
            0,
            self.spatial_rank + 1,
            *range(1, self.spatial_rank + 1),
        )
        rhs_spec = (
            self.spatial_rank + 1,
            self.spatial_rank,
            *range(self.spatial_rank),
        )
        dimension_numbers = jax.lax.ConvDimensionNumbers(
            lhs_spec,
            rhs_spec,
            lhs_spec,
        )
        kernel = self.kernel.value.reshape(
            *self.kernel_size,
            self._in_channel_count // self.groups,
            self._out_channel_count,
        )
        conv_general_dilated = (
            qwix.conv_general_dilated
            if isinstance(kernel, qwix.QArray)
            else self.dot_general or jax.lax.conv_general_dilated
        )
        output = conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=self.stride,
            padding=padding,
            rhs_dilation=self.dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.groups,
            precision=self.precision,
            preferred_element_type=self.preferred_element_type,
        )
        output = output.reshape(*output.shape[:-1], *self.out_channels)
        if self.bias is not None:
            output = output + self.bias
        if unbatched:
            output = output[0]

        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        inputs = '×'.join(map(str, self.in_channels))
        outputs = '×'.join(map(str, self.out_channels))
        kernel = '×'.join(map(str, self.kernel_size))
        stride = '×'.join(map(str, self.stride))
        quantized = isinstance(self.kernel.value, qwix.QArray)
        quant = ' (Qwix PTQ)' if quantized else ''
        return (
            f'{inputs} ➤ {outputs}, k={kernel}, s={stride}{quant}'
        )

class ConvTranspose(Module):
    """Applies an N-dimensional transposed convolution.

    Inputs use the same channels-last layout as :class:`Conv`. An unbatched
    input has shape ``[*spatial, *in_channels]`` and a batched input has shape
    ``[batch, *spatial, *in_channels]``. Structured channel axes are flattened
    only for the underlying convolution and restored in the output.

    Numeric ``padding`` describes the padding of the corresponding forward
    convolution. Increasing it crops more values from the transposed output.
    For each spatial axis, the output size is

    ``(input - 1) * stride - before - after + effective_kernel + output_padding``,

    where ``effective_kernel = dilation * (kernel_size - 1) + 1``.
    ``'VALID'`` produces the full transposed-convolution output. ``'SAME'`` and
    ``'SAME_LOWER'`` produce ``input_size * stride`` positions and differ only
    in which boundary receives an odd extra amount. String padding cannot be
    combined with ``output_padding``.

    When ``groups`` is greater than one, groups partition the first input and
    output channel axes. Both first channel-axis sizes must be divisible by
    ``groups``.

    Args:
        in_channels: Shape of the trailing input-channel axes.
        out_channels: Shape of the trailing output-channel axes.
        kernel_size: Size of the spatial convolution window.
        stride: Factor by which each input position expands the spatial output.
        padding: Forward-convolution padding to remove from the transposed
            output. Accepts ``'VALID'``, ``'SAME'``, ``'SAME_LOWER'``, a
            non-negative integer, one symmetric integer per spatial axis, or
            one ``(before, after)`` pair per spatial axis. Defaults to ``0``.
        dilation: Spacing between kernel elements.
        groups: Number of independent channel groups.
        output_padding: Additional size added to the end of each output spatial
            axis. It resolves shape ambiguity when ``stride > 1`` and does not
            pad the output with values. Each amount must be smaller than either
            its stride or dilation. Defaults to ``0``.
        bias: Whether to add a learnable output bias.
        dtype: Data type passed to the parameter initializers.
        rngs: Random number generator used to initialize parameters.
        kernel_initializer: Function used to initialize the kernel.
        bias_initializer: Function used to initialize the bias.
        quant: Optional Qwix quantization configuration for the kernel.
        dot_general: Optional replacement for ``conv_general_dilated``.
        axis_names: Optional logical names for every kernel axis.
        partition_spec: Optional partition specification for the kernel.
        kernel_metadata: Optional metadata attached to the kernel parameter.
        bias_metadata: Optional metadata attached to the bias parameter.
        precision: Convolution precision forwarded to the convolution callable.
        preferred_element_type: Preferred accumulation and result data type.

    Attributes:
        kernel: Learnable kernel with shape
            ``(*kernel_size, *in_channels, *grouped_out_channels)``.
        bias: Learnable bias with shape ``out_channels``, or ``None``.

    Examples:
        Upsample a one-dimensional input by a factor of two:

        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> conv = nn.ConvTranspose(
        ...     3, 4, kernel_size=3, stride=2, rngs=nn.Rngs(0)
        ... )
        >>> conv(jnp.ones((5, 3))).shape
        (11, 4)

        Structured channel shapes are preserved:

        >>> conv = nn.ConvTranspose(
        ...     (2, 3),
        ...     (4, 5),
        ...     kernel_size=(2, 2),
        ...     stride=2,
        ...     rngs=nn.Rngs(1),
        ... )
        >>> conv(jnp.ones((3, 3, 2, 3))).shape
        (6, 6, 4, 5)
    """

    def __init__(
        self,
        in_channels: GenericShape,
        out_channels: GenericShape,
        kernel_size: GenericShape,
        *,
        stride: GenericShape = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        dilation: GenericShape = 1,
        groups: int = 1,
        output_padding: GenericShape = 0,
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_kernel_initializer,
        bias_initializer: Initializer = default_bias_initializer,
        quant: QuantConfig = None,
        dot_general: ConvGeneralDilated | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        in_channels = _normalize_shape(in_channels, 'in_channels')
        out_channels = _normalize_shape(out_channels, 'out_channels')

        if not isinstance(groups, int) or groups <= 0:
            raise ValueError('groups must be a positive integer')

        if in_channels[0] % groups != 0:
            raise ValueError(
                f'in_channels[0] ({in_channels[0]}) must be divisible by '
                f'groups ({groups})'
            )

        if out_channels[0] % groups != 0:
            raise ValueError(
                f'out_channels[0] ({out_channels[0]}) must be divisible by '
                f'groups ({groups})'
            )

        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        spatial_rank = len(kernel_size)
        stride = Conv._normalize_spatial(
            stride,
            rank=spatial_rank,
            name='stride',
        )
        dilation = Conv._normalize_spatial(
            dilation,
            rank=spatial_rank,
            name='dilation',
        )
        padding = Conv._normalize_padding(padding, spatial_rank)
        output_padding = _normalize_nonnegative(
            output_padding,
            spatial_rank,
            name='output_padding',
        )
        for index, (extra, step, spacing) in enumerate(
            zip(output_padding, stride, dilation)
        ):
            if extra >= step and extra >= spacing:
                raise ValueError(
                    f'output_padding[{index}] must be smaller than stride or '
                    'dilation'
                )

        if isinstance(padding, str) and any(output_padding):
            raise ValueError(
                'output_padding requires explicit numeric padding'
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.output_padding = output_padding
        self.has_bias = bias
        self.spatial_rank = spatial_rank
        self.dot_general = dot_general
        self.precision = precision
        self.preferred_element_type = preferred_element_type
        self._in_channel_count = math.prod(in_channels)
        self._out_channel_count = math.prod(out_channels)

        grouped_out_channels = (
            out_channels[0] // groups,
            *out_channels[1:],
        )
        kernel_shape = kernel_size + in_channels + grouped_out_channels
        if axis_names is not None:
            axis_names = tuple(axis_names)

        if axis_names is not None or partition_spec is not None:
            kernel_initializer = with_logical_partitioning(
                kernel_initializer,
                axis_names,
                partition_spec,
            )

        kernel_array = kernel_initializer(rngs(), kernel_shape, dtype)
        if quant is not None:
            rule = resolve_quantization_rule(
                quant,
                '',
                op_name='conv_general_dilated',
            )
            if rule is not None:
                kernel_array = quantize_conv_weight(
                    kernel_array,
                    rule,
                    output_axis_count=len(out_channels),
                )

        self.kernel = Parameter(
            kernel_array,
            axis_names=axis_names,
            partition_spec=partition_spec,
            metadata=kernel_metadata,
        )

        self.bias = None
        if bias:
            bias_axis_names = None
            bias_partition_spec = None
            if axis_names is not None:
                bias_axis_names = axis_names[-len(out_channels):]
            if partition_spec is not None:
                bias_partition_spec = PartitionSpec(
                    *partition_spec[-len(out_channels):]
                )
            if bias_axis_names is not None or bias_partition_spec is not None:
                bias_initializer = with_logical_partitioning(
                    bias_initializer,
                    bias_axis_names,
                    bias_partition_spec,
                )
            self.bias = Parameter(
                bias_initializer(rngs(), out_channels, dtype),
                axis_names=bias_axis_names,
                partition_spec=bias_partition_spec,
                metadata=bias_metadata,
            )

    @staticmethod
    def _transpose_padding(
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        padding: str | tuple[tuple[int, int], ...],
        output_padding: tuple[int, ...],
    ) -> tuple[tuple[int, int], ...]:
        """Converts forward padding into direct-convolution padding."""
        pairs: list[tuple[int, int]] = []
        for axis, (kernel, step, spacing, extra) in enumerate(
            zip(kernel_size, stride, dilation, output_padding)
        ):
            effective_kernel = spacing * (kernel - 1) + 1
            if isinstance(padding, str):
                if padding in {'SAME', 'SAME_LOWER'}:
                    total = effective_kernel + step - 2
                    if step > effective_kernel - 1:
                        low = effective_kernel - 1
                    else:
                        low = math.ceil(total / 2)
                    high = total - low
                    if padding == 'SAME_LOWER':
                        low, high = high, low
                else:
                    total = (
                        effective_kernel
                        + step
                        - 2
                        + max(effective_kernel - step, 0)
                    )
                    low = effective_kernel - 1
                    high = total - low
            else:
                forward_low, forward_high = padding[axis]
                low = effective_kernel - 1 - forward_low
                high = effective_kernel - 1 - forward_high + extra
            pairs.append((low, high))

        return tuple(pairs)

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the transposed convolution.

        Args:
            x: Channels-last input whose trailing axes match ``in_channels``.
            out_sharding: Optional sharding constraint for the final output.

        Returns:
            The transposed convolution with trailing axes matching
            ``out_channels``.

        Raises:
            ValueError: If the input rank or channels are invalid, or if the
                configuration would produce an empty spatial output.
        """
        expected_batched_rank = (
            self.spatial_rank
            + len(self.in_channels)
            + 1
        )
        if x.ndim not in {expected_batched_rank - 1, expected_batched_rank}:
            raise ValueError(
                f'expected an unbatched rank-{expected_batched_rank - 1} or '
                f'batched rank-{expected_batched_rank} input, got rank {x.ndim}'
            )
        if x.shape[-len(self.in_channels):] != self.in_channels:
            raise ValueError(
                f'expected trailing input channels {self.in_channels}, got '
                f'{x.shape[-len(self.in_channels):]}'
            )

        unbatched = x.ndim == expected_batched_rank - 1
        if unbatched:
            x = x[None, ...]

        transpose_padding = self._transpose_padding(
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding,
            self.output_padding,
        )
        input_spatial_shape = x.shape[1:self.spatial_rank + 1]
        effective_kernel = tuple(
            spacing * (size - 1) + 1
            for size, spacing in zip(self.kernel_size, self.dilation)
        )
        output_spatial_shape = tuple(
            (size - 1) * step + low + high - kernel + 2
            for size, step, kernel, (low, high) in zip(
                input_spatial_shape,
                self.stride,
                effective_kernel,
                transpose_padding,
            )
        )
        if any(size <= 0 for size in output_spatial_shape):
            raise ValueError(
                f'input spatial shape {input_spatial_shape} and transpose '
                f'padding {transpose_padding} produce empty output shape '
                f'{output_spatial_shape}'
            )

        x = x.reshape(
            *x.shape[:self.spatial_rank + 1],
            self._in_channel_count,
        )
        dimension_numbers = _conv_dimension_numbers(self.spatial_rank)
        kernel = self.kernel.value.reshape(
            *self.kernel_size,
            self._in_channel_count,
            self._out_channel_count // self.groups,
        )
        inputs_per_group = self._in_channel_count // self.groups
        reverse_slices = (
            (slice(None, None, -1),) * self.spatial_rank
            + (slice(None), slice(None))
        )
        outputs: list[jax.Array] = []
        for group in range(self.groups):
            start = group * inputs_per_group
            stop = start + inputs_per_group
            group_kernel = kernel[
                (slice(None),) * self.spatial_rank
                + (slice(start, stop), slice(None))
            ]
            group_kernel = group_kernel[reverse_slices]
            conv_general_dilated = (
                qwix.conv_general_dilated
                if isinstance(group_kernel, qwix.QArray)
                else self.dot_general or jax.lax.conv_general_dilated
            )
            outputs.append(
                conv_general_dilated(
                    lhs=x[..., start:stop],
                    rhs=group_kernel,
                    window_strides=(1,) * self.spatial_rank,
                    padding=transpose_padding,
                    lhs_dilation=self.stride,
                    rhs_dilation=self.dilation,
                    dimension_numbers=dimension_numbers,
                    feature_group_count=1,
                    precision=self.precision,
                    preferred_element_type=self.preferred_element_type,
                )
            )
        output = jnp.concatenate(outputs, axis=-1)
        output = output.reshape(*output.shape[:-1], *self.out_channels)
        if self.bias is not None:
            output = output + self.bias
        if unbatched:
            output = output[0]
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        inputs = '×'.join(map(str, self.in_channels))
        outputs = '×'.join(map(str, self.out_channels))
        kernel = '×'.join(map(str, self.kernel_size))
        stride = '×'.join(map(str, self.stride))
        quantized = isinstance(self.kernel.value, qwix.QArray)
        quant = ' (Qwix PTQ)' if quantized else ''
        custom_conv = (
            ' (custom conv_general_dilated)'
            if self.dot_general is not None
            else ''
        )
        return (
            f'{inputs} ➤ {outputs}, k={kernel}, s={stride}'
            f'{quant}{custom_conv}'
        )

class Unfold(Module):
    """
    Extracts sliding local blocks from a batched input tensor.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        stride: int | Sequence[int] = 1,
    ) -> None:
        """Initializes the Unfold module.

        Args:
            kernel_size (int | Sequence[int]): The size of the sliding blocks.
            dilation (int | Sequence[int], optional): A parameter that controls the stride of elements within the neighborhood. Defaults to 1.
            padding (str | int | Sequence[int | tuple[int, int]], optional): Implicit zero padding to be added on both sides of input. Defaults to 0.
            stride (int | Sequence[int], optional): The stride of the sliding blocks in the input spatial dimensions. Defaults to 1.
        """
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.kernel_size = kernel_size
        self.dilation = Conv._normalize_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.padding = Conv._normalize_padding(padding, rank)
        self.stride = Conv._normalize_spatial(
            stride,
            rank=rank,
            name='stride',
        )
        self.spatial_rank = rank

    def __call__(self, x: jax.Array) -> jax.Array:
        """Extracts patches from the input tensor.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: A tensor containing the extracted patches.
        """
        x, unbatched = _as_batched(x, self.spatial_rank)
        patches = jax.lax.conv_general_dilated_patches(
            x,
            filter_shape=self.kernel_size,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            dimension_numbers=_conv_dimension_numbers(self.spatial_rank),
        )
        patches = patches.reshape(
            patches.shape[0],
            math.prod(patches.shape[1:-1]),
            patches.shape[-1],
        )
        return _restore_batch(patches, unbatched)

class Fold(Module):
    """
    Combines an array of sliding local blocks into a large containing tensor.
    """

    def __init__(
        self,
        output_size: int | Sequence[int],
        kernel_size: int | Sequence[int],
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        stride: int | Sequence[int] = 1,
    ) -> None:
        """Initializes the Fold module.

        Args:
            output_size (int | Sequence[int]): The shape of the spatial dimensions of the output.
            kernel_size (int | Sequence[int]): The size of the sliding blocks.
            dilation (int | Sequence[int], optional): A parameter that controls the stride of elements within the neighborhood. Defaults to 1.
            padding (str | int | Sequence[int | tuple[int, int]], optional): Implicit zero padding to be added on both sides of input. Defaults to 0.
            stride (int | Sequence[int], optional): The stride of the sliding blocks in the input spatial dimensions. Defaults to 1.
        """
        output_size = Conv._normalize_spatial(output_size, name='output_size')
        rank = len(output_size)
        self.output_size = output_size
        self.kernel_size = Conv._normalize_spatial(
            kernel_size,
            rank=rank,
            name='kernel_size',
        )
        self.dilation = Conv._normalize_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.padding = Conv._normalize_padding(padding, rank)
        self.stride = Conv._normalize_spatial(
            stride,
            rank=rank,
            name='stride',
        )
        self.spatial_rank = rank

    def __call__(self, patches: jax.Array) -> jax.Array:
        """Folds the extracted patches back into an output tensor.

        Args:
            patches (jax.Array): The input patches array.

        Returns:
            jax.Array: The folded output tensor.
        """
        if patches.ndim not in {2, 3}:
            raise ValueError(
                'Fold expects [windows, patch] or [batch, windows, patch]'
            )
        unbatched = patches.ndim == 2
        if unbatched:
            patches = patches[None, ...]

        padding = _canonical_padding(
            self.padding,
            self.output_size,
            self.kernel_size,
            self.stride,
            self.dilation,
        )
        grid_shape = _window_output_shape(
            self.output_size,
            self.kernel_size,
            self.stride,
            self.dilation,
            padding,
        )
        windows = math.prod(grid_shape)
        if patches.shape[1] != windows:
            raise ValueError(
                f'expected {windows} windows for output_size={self.output_size}, '
                f'got {patches.shape[1]}'
            )
        kernel_volume = math.prod(self.kernel_size)
        if patches.shape[-1] % kernel_volume:
            raise ValueError(
                'patch width must be divisible by the kernel volume '
                f'({kernel_volume})'
            )
        channels = patches.shape[-1] // kernel_volume
        patches = patches.reshape(
            patches.shape[0],
            *grid_shape,
            channels,
            *self.kernel_size,
        )
        output = jnp.zeros(
            (patches.shape[0], *self.output_size, channels),
            dtype=patches.dtype,
        )
        indices = _scatter_indices(
            patches.shape[0],
            channels,
            grid_shape,
            self.stride,
            padding,
        )
        grid_slices = (slice(None),) * self.spatial_rank
        for kernel_index in product(
            *(range(size) for size in self.kernel_size)
        ):
            spatial_indices = tuple(
                index + offset * spacing
                for index, offset, spacing in zip(
                    indices[1:-1],
                    kernel_index,
                    self.dilation,
                )
            )
            values = patches[
                (slice(None), *grid_slices, slice(None), *kernel_index)
            ]
            output = output.at[
                (indices[0], *spatial_indices, indices[-1])
            ].add(values, mode='drop')
        return _restore_batch(output, unbatched)

class MaxPool(Module):
    """
    Applies a max pooling over an input signal.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes the MaxPool module.

        Args:
            kernel_size (int | Sequence[int]): The size of the window to take a max over.
            stride (int | Sequence[int] | None, optional): The stride of the window. Default value is kernel_size. Defaults to None.
            padding (str | int | Sequence[int | tuple[int, int]], optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int | Sequence[int], optional): A parameter that controls the stride of elements in the window. Defaults to 1.
            return_indices (bool, optional): If True, will return the max indices along with the outputs. Defaults to False.
            ceil_mode (bool, optional): When True, will use ceil instead of floor to compute the output shape. Defaults to False.
        """
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.kernel_size = kernel_size
        self.stride = Conv._normalize_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        self.padding = Conv._normalize_padding(padding, rank)
        self.dilation = Conv._normalize_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.spatial_rank = rank

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Applies the max pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array | tuple[jax.Array, jax.Array]: The pooled result, and optionally the indices of the maximum values.
        """
        x, unbatched = _as_batched(x, self.spatial_rank)
        spatial_shape = x.shape[1:-1]
        padding = _pool_padding(
            self.padding,
            spatial_shape,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.ceil_mode,
        )
        window, strides, reduce_padding, window_dilation = (
            _reduce_window_config(
                self.spatial_rank,
                self.kernel_size,
                self.stride,
                self.dilation,
                padding,
            )
        )
        initial = _max_identity(x.dtype)
        if not self.return_indices:
            output = jax.lax.reduce_window(
                x,
                initial,
                jax.lax.max,
                window,
                strides,
                reduce_padding,
                window_dilation=window_dilation,
            )
            return _restore_batch(output, unbatched)

        flat_indices = jnp.arange(
            math.prod(spatial_shape),
            dtype=jnp.int32,
        ).reshape((1, *spatial_shape, 1))
        flat_indices = jnp.broadcast_to(flat_indices, x.shape)
        no_index = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)

        def select_max(
            left: tuple[jax.Array, jax.Array],
            right: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            left_value, left_index = left
            right_value, right_index = right
            choose_right = (right_value > left_value) | (
                (right_value == left_value) & (right_index < left_index)
            )
            return (
                jnp.where(choose_right, right_value, left_value),
                jnp.where(choose_right, right_index, left_index),
            )

        output, indices = jax.lax.reduce_window(
            (x, flat_indices),
            (initial, no_index),
            select_max,
            window,
            strides,
            reduce_padding,
            window_dilation=window_dilation,
        )
        return (
            _restore_batch(output, unbatched),
            _restore_batch(indices, unbatched),
        )

class MaxUnpool(Module):
    """
    Computes a partial inverse of MaxPool.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int | tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> None:
        """Initializes the MaxUnpool module.

        Args:
            kernel_size (int | Sequence[int]): Size of the max pooling window.
            stride (int | Sequence[int] | None, optional): Stride of the max pooling window. Defaults to None.
            padding (int | Sequence[int | tuple[int, int]], optional): Padding that was added to the input. Defaults to 0.
            dilation (int | Sequence[int], optional): Spacing between window elements. Defaults to 1.
        """
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.kernel_size = kernel_size
        self.stride = Conv._normalize_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        normalized_padding = Conv._normalize_padding(padding, rank)
        if isinstance(normalized_padding, str):
            raise TypeError('MaxUnpool requires explicit numeric padding')
        self.padding = normalized_padding
        self.dilation = Conv._normalize_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.spatial_rank = rank

    def __call__(
        self,
        x: jax.Array,
        indices: jax.Array,
        output_size: int | Sequence[int] | None = None,
    ) -> jax.Array:
        """Applies the unpooling operation.

        Args:
            x (jax.Array): The input array to unpool.
            indices (jax.Array): The indices returned by MaxPool.
            output_size (int | Sequence[int] | None, optional): The targeted output size. Defaults to None.

        Returns:
            jax.Array: The unpooled result.
        """
        x, unbatched = _as_batched(x, self.spatial_rank)
        indices, indices_unbatched = _as_batched(indices, self.spatial_rank)
        if indices_unbatched != unbatched or indices.shape != x.shape:
            raise ValueError('indices must have the same shape as the pooled input')

        if output_size is None:
            output_size = tuple(
                (size - 1) * step
                - low
                - high
                + spacing * (kernel - 1)
                + 1
                for size, step, (low, high), spacing, kernel in zip(
                    x.shape[1:-1],
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.kernel_size,
                )
            )
        else:
            output_size = Conv._normalize_spatial(
                output_size,
                rank=self.spatial_rank,
                name='output_size',
            )

        batch_size, channels = x.shape[0], x.shape[-1]
        values = x.reshape(batch_size, -1, channels)
        flat_indices = indices.reshape(batch_size, -1, channels)
        output = jnp.zeros(
            (batch_size, math.prod(output_size), channels),
            dtype=x.dtype,
        )
        batch = jnp.arange(batch_size).reshape(batch_size, 1, 1)
        channel = jnp.arange(channels).reshape(1, 1, channels)
        output = output.at[batch, flat_indices, channel].set(
            values,
            mode='drop',
        )
        output = output.reshape(batch_size, *output_size, channels)
        return _restore_batch(output, unbatched)

class AvgPool(Module):
    """
    Applies an average pooling over an input signal.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        """Initializes the AvgPool module.

        Args:
            kernel_size (int | Sequence[int]): The size of the window.
            stride (int | Sequence[int] | None, optional): The stride of the window. Defaults to None.
            padding (str | int | Sequence[int | tuple[int, int]], optional): Implicit zero padding to be added on both sides. Defaults to 0.
            ceil_mode (bool, optional): When True, will use ceil instead of floor to compute the output shape. Defaults to False.
            count_include_pad (bool, optional): When True, will include the zero-padding in the averaging calculation. Defaults to True.
            divisor_override (int | None, optional): If specified, it will be used as divisor, otherwise size of the pooling region will be used. Defaults to None.
        """
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        if divisor_override is not None and divisor_override <= 0:
            raise ValueError('divisor_override must be positive')
        self.kernel_size = kernel_size
        self.stride = Conv._normalize_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        self.padding = Conv._normalize_padding(padding, rank)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.spatial_rank = rank

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the average pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The pooled result.
        """
        x, unbatched = _as_batched(x, self.spatial_rank)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            x = x.astype(jnp.float32)
        configured_padding = _canonical_padding(
            self.padding,
            x.shape[1:-1],
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
        )
        padding = _pool_padding(
            self.padding,
            x.shape[1:-1],
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
            self.ceil_mode,
        )
        window, strides, reduce_padding, _ = _reduce_window_config(
            self.spatial_rank,
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
            padding,
        )
        total = jax.lax.reduce_window(
            x,
            jnp.asarray(0, dtype=x.dtype),
            jax.lax.add,
            window,
            strides,
            reduce_padding,
        )
        if self.divisor_override is not None:
            divisor = self.divisor_override
        elif self.count_include_pad and not self.ceil_mode:
            divisor = math.prod(self.kernel_size)
        else:
            count_input = jnp.ones_like(x[..., :1])
            count_padding = reduce_padding
            if self.count_include_pad:
                count_input = jnp.pad(
                    count_input,
                    ((0, 0), *configured_padding, (0, 0)),
                    mode='constant',
                    constant_values=1,
                )
                count_padding = (
                    (0, 0),
                    *(
                        (
                            total_low - configured_low,
                            total_high - configured_high,
                        )
                        for (total_low, total_high), (
                            configured_low,
                            configured_high,
                        ) in zip(padding, configured_padding)
                    ),
                    (0, 0),
                )
            valid = jax.lax.reduce_window(
                count_input,
                jnp.asarray(0, dtype=x.dtype),
                jax.lax.add,
                window,
                strides,
                count_padding,
            )
            divisor = valid
        return _restore_batch(total / divisor, unbatched)

class FractionalMaxPool(Module):
    """
    Applies a fractional max pooling over an input signal.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        output_size: int | Sequence[int] | None = None,
        output_ratio: float | Sequence[float] | None = None,
        return_indices: bool = False,
        random_samples: jax.Array | Sequence[float] | None = None,
        *,
        rngs: Rngs | None = None,
    ) -> None:
        """Initializes the FractionalMaxPool module.

        Args:
            kernel_size (int | Sequence[int]): The size of the window to take a max over.
            output_size (int | Sequence[int] | None, optional): The target output size. Defaults to None.
            output_ratio (float | Sequence[float] | None, optional): The ratio of output size to input size. Defaults to None.
            return_indices (bool, optional): If True, will return the max indices along with the outputs. Defaults to False.
            random_samples (jax.Array | Sequence[float] | None, optional): Optional random samples for pooling grid generation. Defaults to None.
            rngs (Rngs | None, optional): PRNG key generator. Defaults to None.
        """
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        if (output_size is None) == (output_ratio is None):
            raise ValueError(
                'exactly one of output_size or output_ratio must be provided'
            )
        if output_size is not None:
            output_size = Conv._normalize_spatial(
                output_size,
                rank=rank,
                name='output_size',
            )
        if output_ratio is not None:
            if isinstance(output_ratio, (int, float)):
                output_ratio = (float(output_ratio),) * rank
            else:
                output_ratio = tuple(float(value) for value in output_ratio)
            if len(output_ratio) != rank:
                raise ValueError(
                    f'output_ratio must contain {rank} values, '
                    f'got {len(output_ratio)}'
                )
            if any(value <= 0 or value > 1 for value in output_ratio):
                raise ValueError('output_ratio values must be in (0, 1]')

        if random_samples is None:
            if rngs is None:
                samples = jnp.full((rank,), 0.5, dtype=jnp.float32)
            else:
                samples = jax.random.uniform(rngs(), (rank,))
        else:
            if isinstance(random_samples, Sequence) and any(
                float(value) < 0 or float(value) >= 1
                for value in random_samples
            ):
                raise ValueError('random_samples values must be in [0, 1)')
            samples = jnp.asarray(random_samples, dtype=jnp.float32)
            if samples.shape != (rank,):
                raise ValueError(
                    f'random_samples must have shape ({rank},), '
                    f'got {samples.shape}'
                )

        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices
        self.random_samples = samples
        self.spatial_rank = rank

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Applies the fractional max pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array | tuple[jax.Array, jax.Array]: The pooled result, and optionally the indices of the maximum values.
        """
        x, unbatched = _as_batched(x, self.spatial_rank)
        spatial_shape = x.shape[1:-1]
        if self.output_size is None:
            output_size = tuple(
                max(1, math.floor(size * ratio))
                for size, ratio in zip(spatial_shape, self.output_ratio)
            )
        else:
            output_size = self.output_size
        if any(
            kernel > size or output > size - kernel + 1
            for kernel, size, output in zip(
                self.kernel_size,
                spatial_shape,
                output_size,
            )
        ):
            raise ValueError(
                'kernel_size and output size must fit within the input'
            )

        starts = []
        for size, kernel, output, sample in zip(
            spatial_shape,
            self.kernel_size,
            output_size,
            self.random_samples,
        ):
            maximum = size - kernel
            if output == 1:
                positions = jnp.asarray(
                    [jnp.floor(sample * (maximum + 1))],
                    dtype=jnp.int32,
                )
            else:
                alpha = maximum / (output - 1)
                positions = jnp.floor(
                    (jnp.arange(output) + sample) * alpha
                ) - jnp.floor(sample * alpha)
                positions = positions.astype(jnp.int32).at[-1].set(maximum)
            starts.append(positions)

        values = []
        indices = []
        batch_size, channels = x.shape[0], x.shape[-1]
        kernel_volume = math.prod(self.kernel_size)
        for output_index in product(*(range(size) for size in output_size)):
            start = tuple(
                starts[axis][position]
                for axis, position in enumerate(output_index)
            )
            patch = jax.lax.dynamic_slice(
                x,
                (0, *start, 0),
                (batch_size, *self.kernel_size, channels),
            )
            patch = patch.reshape(batch_size, kernel_volume, channels)
            local_index = jnp.argmax(patch, axis=1).astype(jnp.int32)
            values.append(jnp.take_along_axis(
                patch,
                local_index[:, None, :],
                axis=1,
            )[:, 0, :])
            if self.return_indices:
                remainder = local_index
                coordinates = []
                for kernel in reversed(self.kernel_size):
                    coordinates.append(remainder % kernel)
                    remainder = remainder // kernel
                coordinates.reverse()
                global_index = jnp.zeros_like(local_index)
                for size, offset, coordinate in zip(
                    spatial_shape,
                    start,
                    coordinates,
                ):
                    global_index = global_index * size + offset + coordinate
                indices.append(global_index)

        output = jnp.stack(values, axis=1).reshape(
            batch_size,
            *output_size,
            channels,
        )
        output = _restore_batch(output, unbatched)
        if not self.return_indices:
            return output
        index_output = jnp.stack(indices, axis=1).reshape(
            batch_size,
            *output_size,
            channels,
        )
        return output, _restore_batch(index_output, unbatched)

class LPPool(Module):
    """
    Applies a power-average pooling over an input signal.
    """

    def __init__(
        self,
        norm_type: float,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes the LPPool module.

        Args:
            norm_type (float): The power to use.
            kernel_size (int | Sequence[int]): The size of the window.
            stride (int | Sequence[int] | None, optional): The stride of the window. Defaults to None.
            ceil_mode (bool, optional): When True, will use ceil instead of floor to compute the output shape. Defaults to False.
        """
        if norm_type <= 0:
            raise ValueError('norm_type must be positive')
        kernel_size = Conv._normalize_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = Conv._normalize_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        self.ceil_mode = ceil_mode
        self.spatial_rank = rank

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the LPPool operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The pooled result.
        """
        x, unbatched = _as_batched(x, self.spatial_rank)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            x = x.astype(jnp.float32)
        padding = _pool_padding(
            ((0, 0),) * self.spatial_rank,
            x.shape[1:-1],
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
            self.ceil_mode,
        )
        window, strides, reduce_padding, _ = _reduce_window_config(
            self.spatial_rank,
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
            padding,
        )
        powered = jnp.abs(x) ** self.norm_type
        total = jax.lax.reduce_window(
            powered,
            jnp.asarray(0, dtype=x.dtype),
            jax.lax.add,
            window,
            strides,
            reduce_padding,
        )
        output = total ** (1.0 / self.norm_type)
        return _restore_batch(output, unbatched)

class AdaptiveMaxPool(Module):
    """
    Applies an adaptive max pooling over an input signal.
    """

    def __init__(
        self,
        output_size: int | Sequence[int | None],
        return_indices: bool = False,
    ) -> None:
        self.output_size = _normalize_adaptive_size(output_size)
        self.return_indices = return_indices

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Applies the adaptive max pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array | tuple[jax.Array, jax.Array]: The pooled result, and optionally the indices of the maximum values.
        """
        rank = len(self.output_size)
        x, unbatched = _as_batched(x, rank)
        spatial_shape = x.shape[1:-1]
        output_size = tuple(
            size if requested is None else requested
            for size, requested in zip(spatial_shape, self.output_size)
        )
        values, indices = _adaptive_pool(
            x,
            output_size,
            reduction='max',
            return_indices=self.return_indices,
        )
        values = _restore_batch(values, unbatched)
        if not self.return_indices:
            return values
        return values, _restore_batch(indices, unbatched)

class AdaptiveAvgPool(Module):
    """
    Applies an adaptive average pooling over an input signal.
    """

    def __init__(
        self,
        output_size: int | Sequence[int | None],
    ) -> None:
        """Initializes the AdaptiveAvgPool module.

        Args:
            output_size (int | Sequence[int | None]): The target output size.
        """
        self.output_size = _normalize_adaptive_size(output_size)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the adaptive average pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The pooled result.
        """
        rank = len(self.output_size)
        x, unbatched = _as_batched(x, rank)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            x = x.astype(jnp.float32)
        output_size = tuple(
            size if requested is None else requested
            for size, requested in zip(x.shape[1:-1], self.output_size)
        )
        values, _ = _adaptive_pool(
            x,
            output_size,
            reduction='mean',
            return_indices=False,
        )
        return _restore_batch(values, unbatched)

class Padding(Module):
    """
    Pads an input array.
    """

    def __init__(
        self,
        padding: int | Sequence[int] | Sequence[tuple[int, int]],
        mode: str = 'constant',
        value: float = 0.0,
    ) -> None:
        """Initializes the Padding module.

        Args:
            padding (int | Sequence[int] | Sequence[tuple[int, int]]): The size of the padding.
            mode (str, optional): The padding mode. Defaults to 'constant'.
            value (float, optional): The fill value for 'constant' padding. Defaults to 0.0.
        """
        aliases = {
            'zeros': 'constant',
            'replicate': 'edge',
            'circular': 'wrap',
        }
        mode = aliases.get(mode.lower(), mode.lower())
        supported = {
            'constant',
            'edge',
            'reflect',
            'symmetric',
            'wrap',
        }
        if mode not in supported:
            choices = ', '.join(sorted(supported | set(aliases)))
            raise ValueError(f'padding mode must be one of {choices}')
        self.padding = padding
        self.mode = mode
        self.value = value

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the padding operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The padded result.
        """
        if self.mode == 'constant':
            return jnp.pad(
                x,
                self.padding,
                mode=self.mode,
                constant_values=self.value,
            )
        return jnp.pad(x, self.padding, mode=self.mode)


__all__ = [
    'AdaptiveAvgPool',
    'AdaptiveMaxPool',
    'AvgPool',
    'Conv',
    'ConvTranspose',
    'Fold',
    'FractionalMaxPool',
    'LPPool',
    'MaxPool',
    'MaxUnpool',
    'Padding',
    'Unfold',
]
