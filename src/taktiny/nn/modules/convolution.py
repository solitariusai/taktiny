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
from functools import partial
from itertools import product
from typing import cast

import jax
import jax.numpy as jnp
import qwix
from jax.lax import PrecisionLike
from jax.nn.initializers import lecun_uniform
from jax.sharding import NamedSharding, PartitionSpec
from jax.typing import DTypeLike

from taktiny.nn.base import Module, Parameter
from taktiny.nn.rng import Rngs, get_context_rng
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
    _validate_integer,
    _validate_positive_float,
    _window_output_shape,
)
from taktiny.utils.ops import _rule, _rule_conv, _validate_conv_training_rule
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


def _channel_groups(
    groups: GenericShape,
    in_channels: tuple[int, ...],
    out_channels: tuple[int, ...],
    *,
    transpose: bool = False,
) -> tuple[int | tuple[int, ...], int, tuple[int, ...] | None]:
    """Validate legacy scalar grouping or structured per-axis grouping."""
    if isinstance(groups, int):
        _validate_integer(groups, 'groups')
        if in_channels[0] % groups:
            raise ValueError('in_channels[0] must be divisible by groups')
        output_size = out_channels[0] if transpose else math.prod(out_channels)
        if output_size % groups:
            raise ValueError('out_channels must be divisible by groups')
        return groups, groups, None
    shape = _normalize_shape(groups, 'groups')
    if len(shape) != len(in_channels) or len(shape) != len(out_channels):
        raise ValueError('groups must have one entry per input and output channel axis')
    for name, channels in (('in_channels', in_channels), ('out_channels', out_channels)):
        for axis, (size, count) in enumerate(zip(channels, shape)):
            if size % count:
                raise ValueError(f'{name}[{axis}] ({size}) must be divisible by groups[{axis}] ({count})')
    return shape, math.prod(shape), shape


def _group_channel_axes[T: (jax.Array, qwix.QArray)](
    array: T,
    start: int,
    channels: tuple[int, ...],
    groups: tuple[int, ...],
) -> T:
    """Reorder channel blocks from (g0,c0,g1,c1,...) to (g0,g1,...,c0,c1,...)."""
    rank = len(channels)
    split = tuple(v for size, count in zip(channels, groups) for v in (count, size // count))
    shape = array.shape
    reshaped = array.reshape(*shape[:start], *split, *shape[start + rank:])
    order = (
        tuple(range(start))
        + tuple(range(start, start + 2 * rank, 2))
        + tuple(range(start + 1, start + 2 * rank, 2))
        + tuple(range(start + 2 * rank, reshaped.ndim))
    )
    return cast(T, reshaped.transpose(order).reshape(*shape))


def _restore_channel_axes(
    array: jax.Array,
    channels: tuple[int, ...],
    groups: tuple[int, ...],
) -> jax.Array:
    """Restore structured channels from group-major convolution output."""
    prefix = array.shape[:-1]
    start, rank = len(prefix), len(channels)
    array = array.reshape(*prefix, *groups, *(size // count for size, count in zip(channels, groups)))
    order = tuple(range(start)) + tuple(
        axis for i in range(rank) for axis in (start + i, start + rank + i)
    )
    return array.transpose(order).reshape(*prefix, *channels)
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

    A sequence ``groups=(g0, g1, ...)`` partitions each channel axis into
    contiguous blocks. It must have the same rank as both channel shapes,
    and each entry must divide the corresponding input and output dimension.
    Each of the ``prod(groups)`` independent groups mixes only its own
    ``in_channels[i] // groups[i]`` channels along each axis. Channel blocks
    are rearranged internally; callers keep the structured channel layout.
    The kernel shape is ``(*kernel_size, *channels_per_group, *out_channels)``.

    For channels ``(8, 32)``, ``groups=(8, 32)`` is depthwise,
    ``groups=(8, 8)`` mixes four features within each head, and
    ``groups=(2, 2)`` mixes blocks of four heads by sixteen features.
    For example, ``Conv((8, 32), (8, 32), 3, groups=(8, 32),
    padding='SAME', rngs=Rngs(0))`` preserves the shape of an input
    ``(batch, length, 8, 32)`` without manual flattening.

    An integer retains legacy grouping: it divides the first input channel
    axis and partitions flattened output channels into contiguous groups.
    The first input dimension and total output count must be divisible by it.

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
        groups: Positive integer group count, or a sequence of positive
            per-axis group counts. Defaults to ``1``. Sequence entries must
            divide both corresponding channel dimensions; scalar values are
            not broadcast across axes.
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
        quant: Optional Qwix quantization configuration. A ``QtRule`` or
            ``QtProvider`` keeps floating-point trainable parameters and
            quantizes convolution operands; ``bwd_qtype`` controls gradient
            quantization. Supported training formats and grouping depend on
            Qwix QT kernels and the execution backend.
            Tiled training quantization and ``additional_qt_config`` are
            unsupported. Training rules cannot be combined with ``dot_general``.
            Other rules retain weight-only quantization behavior.
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
        groups: GenericShape = 1,
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

        groups, group_count, group_shape = _channel_groups(groups, in_channels, out_channels)

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
        self._group_count = group_count
        self._group_shape = group_shape
        self.has_bias = bias
        self.pad_mode = pad_mode
        self.spatial_rank = spatial_rank
        self.dot_general = dot_general
        self.precision = precision
        self.preferred_element_type = preferred_element_type
        self._in_channel_count = math.prod(in_channels)
        self._out_channel_count = math.prod(out_channels)

        grouped_in_channels = (
            tuple(size // count for size, count in zip(in_channels, group_shape))
            if group_shape is not None else
            (in_channels[0] // group_count, *in_channels[1:])
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
        selected_rule = _rule(quant, '', 'conv_general_dilated')
        self._training_rule = selected_rule if isinstance(selected_rule, qwix.QtRule) else None
        if self._training_rule is not None:
            _validate_conv_training_rule(self._training_rule)
            if dot_general is not None:
                raise ValueError('QtRule and a custom dot_general cannot be supplied together')
        if quant is not None and self._training_rule is None:
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

        if self._group_shape is not None:
            x = _group_channel_axes(x, self.spatial_rank + 1, self.in_channels, self._group_shape)
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
        kernel = self.kernel.value
        if self._group_shape is not None:
            kernel = _group_channel_axes(
                kernel, self.spatial_rank + len(self.in_channels), self.out_channels, self._group_shape,
            )
        kernel = kernel.reshape(
            *self.kernel_size,
            self._in_channel_count // self._group_count,
            self._out_channel_count,
        )
        conv_general_dilated = (
            partial(_rule_conv, rule=self._training_rule)
            if self._training_rule is not None else
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
            feature_group_count=self._group_count,
            precision=self.precision,
            preferred_element_type=self.preferred_element_type,
        )
        output = (
            _restore_channel_axes(output, self.out_channels, self._group_shape)
            if self._group_shape is not None else
            output.reshape(*output.shape[:-1], *self.out_channels)
        )
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

    A sequence ``groups=(g0, g1, ...)`` partitions each channel axis into
    contiguous blocks. Its rank must match both channel shapes, and each
    entry must divide the corresponding input and output dimension. There
    are ``prod(groups)`` independent groups; channels mix within each block,
    never between blocks. Rearrangement is internal, so structured trailing
    channel axes are preserved. The kernel shape is
    ``(*kernel_size, *in_channels, *output_channels_per_group)``.

    For channels ``(8, 32)``, ``groups=(8, 32)`` is depthwise,
    ``groups=(8, 8)`` mixes four features per head, and ``groups=(2, 2)``
    mixes blocks of four heads by sixteen features. For example,
    ``ConvTranspose((8, 32), (8, 32), 3, stride=2, padding='SAME',
    groups=(8, 32), rngs=Rngs(0))`` maps ``(batch, length, 8, 32)`` to
    ``(batch, 2 * length, 8, 32)`` without manual flattening.

    An integer retains legacy grouping along the first input and output
    channel axes, both of which must be divisible by that integer. Integers
    are not broadcast across channel axes.

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
        groups: Positive integer group count, or a sequence of positive
            per-axis group counts dividing the corresponding input and output
            dimensions. Defaults to ``1``.
        output_padding: Additional size added to the end of each output spatial
            axis. It resolves shape ambiguity when ``stride > 1`` and does not
            pad the output with values. Each amount must be smaller than either
            its stride or dilation. Defaults to ``0``.
        bias: Whether to add a learnable output bias.
        dtype: Data type passed to the parameter initializers.
        rngs: Random number generator used to initialize parameters.
        kernel_initializer: Function used to initialize the kernel.
        bias_initializer: Function used to initialize the bias.
        quant: Optional Qwix quantization configuration. A ``QtRule`` or
            ``QtProvider`` keeps floating-point trainable parameters and
            quantizes convolution operands; ``bwd_qtype`` controls gradient
            quantization. Supported training formats depend on Qwix QT
            kernels and the execution backend.
            Tiled training quantization and ``additional_qt_config`` are
            unsupported. Training rules cannot be combined with ``dot_general``.
            Other rules retain weight-only quantization behavior.
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
        groups: GenericShape = 1,
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

        groups, group_count, group_shape = _channel_groups(
            groups, in_channels, out_channels, transpose=True,
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
        self._group_count = group_count
        self._group_shape = group_shape
        self.output_padding = output_padding
        self.has_bias = bias
        self.spatial_rank = spatial_rank
        self.dot_general = dot_general
        self.precision = precision
        self.preferred_element_type = preferred_element_type
        self._in_channel_count = math.prod(in_channels)
        self._out_channel_count = math.prod(out_channels)

        grouped_out_channels = (
            tuple(size // count for size, count in zip(out_channels, group_shape))
            if group_shape is not None else
            (out_channels[0] // group_count, *out_channels[1:])
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
        selected_rule = _rule(quant, '', 'conv_general_dilated')
        self._training_rule = selected_rule if isinstance(selected_rule, qwix.QtRule) else None
        if self._training_rule is not None:
            _validate_conv_training_rule(self._training_rule)
            if dot_general is not None:
                raise ValueError('QtRule and a custom dot_general cannot be supplied together')
        if quant is not None and self._training_rule is None:
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

        if self._group_shape is not None:
            x = _group_channel_axes(x, self.spatial_rank + 1, self.in_channels, self._group_shape)
        x = x.reshape(
            *x.shape[:self.spatial_rank + 1],
            self._in_channel_count,
        )
        dimension_numbers = _conv_dimension_numbers(self.spatial_rank)
        kernel = self.kernel.value
        if self._group_shape is not None:
            kernel = _group_channel_axes(kernel, self.spatial_rank, self.in_channels, self._group_shape)
        kernel = kernel.reshape(
            *self.kernel_size,
            self._in_channel_count,
            self._out_channel_count // self._group_count,
        )
        inputs_per_group = self._in_channel_count // self._group_count
        reverse_slices = (
            (slice(None, None, -1),) * self.spatial_rank
            + (slice(None), slice(None))
        )
        outputs: list[jax.Array] = []
        for group in range(self._group_count):
            start = group * inputs_per_group
            stop = start + inputs_per_group
            group_kernel = kernel[
                (slice(None),) * self.spatial_rank
                + (slice(start, stop), slice(None))
            ]
            group_kernel = group_kernel[reverse_slices]
            conv_general_dilated = (
                partial(_rule_conv, rule=self._training_rule)
                if self._training_rule is not None else
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
        output = (
            _restore_channel_axes(output, self.out_channels, self._group_shape)
            if self._group_shape is not None else
            output.reshape(*output.shape[:-1], *self.out_channels)
        )
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

def _positive_spatial(
    value: GenericShape, *, name: str, rank: int | None = None,
) -> tuple[int, ...]:
    values = _normalize_shape(value, name)
    if rank is not None and isinstance(value, int):
        values = values * rank
    if rank is not None and len(values) != rank:
        raise ValueError(f'{name} must contain {rank} values, got {len(values)}')
    return values

def _spatial_padding(
    padding: str | int | Sequence[int | tuple[int, int]], rank: int,
) -> str | tuple[tuple[int, int], ...]:
    result = Conv._normalize_padding(padding, rank)
    if not isinstance(result, str):
        for low, high in result:
            if isinstance(low, bool) or isinstance(high, bool):
                raise TypeError('padding values must be integers, not booleans')
    return result

def _boolean(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f'{name} must be a boolean')
    return value

def _adaptive_size(value: int | Sequence[int | None]) -> tuple[int | None, ...]:
    sizes = _normalize_adaptive_size(value)
    for size in sizes:
        if size is not None:
            _validate_integer(size, 'output_size')
    return sizes

class _SpatialOp(Module):
    """Common channels-last input validation, output placement and repr."""

    spatial_rank: int

    def _input(self, x: jax.Array) -> tuple[jax.Array, bool]:
        x, unbatched = _as_batched(jnp.asarray(x), self.spatial_rank)
        if any(size <= 0 for size in x.shape[1:]):
            raise ValueError('spatial and channel dimensions must be nonempty')
        return x, unbatched

    def _finish(
        self, output: jax.Array, unbatched: bool,
        out_sharding: jax.sharding.Sharding | None,
    ) -> jax.Array:
        output = _restore_batch(output, unbatched)
        if (isinstance(out_sharding, NamedSharding)
                and out_sharding.mesh.are_all_axes_explicit):
            return jax.sharding.reshard(output, out_sharding)
        return _constrain(output, out_sharding)

    def extra_repr(self) -> str:
        fields = []
        for name in ('output_size', 'kernel_size', 'stride', 'padding', 'dilation',
                     'norm_type', 'output_ratio', 'return_indices', 'ceil_mode',
                     'count_include_pad', 'divisor_override', 'mode', 'value'):
            value = getattr(self, name, None)
            if value is None:
                continue
            label = {'kernel_size': 'k', 'stride': 's', 'dilation': 'd'}.get(name, name)
            if name in {'kernel_size', 'stride', 'dilation', 'output_size'}:
                value = '×'.join('*' if v is None else str(v) for v in value)
            fields.append(f'{label}={value}')
        return ', '.join(fields)

class Unfold(_SpatialOp):
    """Extract channels-last sliding windows into flattened patches.

    Inputs are ``(*spatial, channels)`` or ``(batch, *spatial, channels)``, with one
    trailing channel axis. Scalars specify 1-D spatial shapes; sequences specify
    the spatial rank. Spatial and channel dimensions must be nonempty.
    __call__ accepts out_sharding for the final output.

    Args:
        kernel_size: Positive spatial window sizes (GenericShape).
        dilation: Positive spacing within each window, scalar or per-axis.
        padding: VALID, SAME, SAME_LOWER, a nonnegative symmetric integer,
            per-axis integers, or per-axis (before, after) pairs. For 1-D,
            a flat pair denotes asymmetric padding. Padding contributes zeros.
        stride: Positive window strides; defaults to 1.

    Returns (windows, patch_width) or (batch, windows, patch_width). Windows
    are flattened in spatial row-major order. Within each patch, channel comes
    before the kernel axes: patch_width = channels * prod(kernel_size).
    Raises ValueError when effective kernel and padding produce no windows.
    Fold sums overlapping patches; it is not automatically an inverse.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.Unfold(2)(jnp.ones((4, 1))).shape
        (3, 2)
    """

    def __init__(
        self,
        kernel_size: GenericShape,
        dilation: GenericShape = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        stride: GenericShape = 1,
    ) -> None:
        kernel_size = _positive_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.kernel_size = kernel_size
        self.dilation = _positive_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.padding = _spatial_padding(padding, rank)
        self.stride = _positive_spatial(
            stride,
            rank=rank,
            name='stride',
        )
        self.spatial_rank = rank

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Extracts patches from the input tensor.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: A tensor containing the extracted patches.
        """
        x, unbatched = self._input(x)
        padding = _canonical_padding(self.padding, x.shape[1:-1], self.kernel_size,
                                     self.stride, self.dilation)
        _window_output_shape(x.shape[1:-1], self.kernel_size, self.stride,
                             self.dilation, padding)
        patches = jax.lax.conv_general_dilated_patches(
            x,
            filter_shape=self.kernel_size,
            window_strides=self.stride,
            padding=padding,
            rhs_dilation=self.dilation,
            dimension_numbers=_conv_dimension_numbers(self.spatial_rank),
        )
        patches = patches.reshape(
            patches.shape[0],
            math.prod(patches.shape[1:-1]),
            patches.shape[-1],
        )
        return self._finish(patches, unbatched, out_sharding)

class Fold(_SpatialOp):
    """Overlap-add flattened sliding patches into a channels-last array.

    Args:
        output_size: Positive target spatial dimensions; determines spatial rank.
        kernel_size: Positive window sizes, scalar or per-axis.
        dilation: Positive spacing within windows; defaults to 1.
        padding: The same VALID/SAME/SAME_LOWER or numeric padding used by Unfold.
        stride: Positive window strides; defaults to 1.

    Input is (windows, patch_width) or (batch, windows, patch_width). Patch
    layout must match Unfold: channel followed by flattened kernel dimensions.
    Output is ``(*output_size, channels)``, optionally with a batch axis.
    Overlaps are summed and padded positions are discarded, including negative
    coordinates; they never wrap around the output. To invert Unfold, divide
    by the overlap counts where those counts are nonzero. __call__ accepts
    out_sharding. Patch width must be a positive multiple of kernel volume.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.Fold(4, 2)(jnp.ones((3, 2)))[:, 0].tolist()
        [1.0, 2.0, 2.0, 1.0]
    """

    def __init__(
        self,
        output_size: GenericShape,
        kernel_size: GenericShape,
        dilation: GenericShape = 1,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        stride: GenericShape = 1,
    ) -> None:
        output_size = _positive_spatial(output_size, name='output_size')
        rank = len(output_size)
        self.output_size = output_size
        self.kernel_size = _positive_spatial(
            kernel_size,
            rank=rank,
            name='kernel_size',
        )
        self.dilation = _positive_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.padding = _spatial_padding(padding, rank)
        self.stride = _positive_spatial(
            stride,
            rank=rank,
            name='stride',
        )
        self.spatial_rank = rank

    def __call__(
        self, patches: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Folds the extracted patches back into an output tensor.

        Args:
            patches (jax.Array): The input patches array.

        Returns:
            jax.Array: The folded output tensor.
        """
        patches = jnp.asarray(patches)
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
        if patches.shape[-1] == 0 or patches.shape[-1] % kernel_volume:
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
            ].add(values, mode='drop', wrap_negative_indices=False)
        return self._finish(output, unbatched, out_sharding)

class MaxPool(_SpatialOp):
    """Take maxima over channels-last spatial windows.

    Inputs are ``(*spatial, channels)`` or ``(batch, *spatial, channels)``, with one
    trailing channel axis. Scalars specify 1-D spatial shapes; sequences specify
    the spatial rank. Spatial and channel dimensions must be nonempty.
    __call__ accepts out_sharding for the final output.

    Args:
        kernel_size: Positive spatial window sizes.
        stride: Positive strides; None uses kernel_size.
        padding: VALID, SAME, SAME_LOWER or nonnegative symmetric/asymmetric
            numeric padding, as in Conv. Padding uses the dtype's minimum
            identity (negative infinity for floating-point inputs), not zero.
        dilation: Positive spacing within each pooling window; defaults to 1.
        return_indices: Return (values, indices) when True.
        ceil_mode: Include a final partial window by extending right padding.

    Indices are int32 row-major spatial offsets, excluding batch and channel.
    Ties choose the smallest spatial offset. NaNs propagate; tied NaNs choose
    the first offset. Fully padded windows use the maximum int32 index sentinel.
    Complex inputs are unsupported. With indices enabled, out_sharding is
    applied to both result arrays.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> values, indices = nn.MaxPool(2, return_indices=True)(jnp.arange(4.)[:, None])
        >>> values[:, 0].tolist(), indices[:, 0].tolist()
        ([1.0, 3.0], [1, 3])
    """

    def __init__(
        self,
        kernel_size: GenericShape,
        stride: GenericShape | None = None,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        dilation: GenericShape = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        kernel_size = _positive_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.kernel_size = kernel_size
        self.stride = _positive_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        self.padding = _spatial_padding(padding, rank)
        self.dilation = _positive_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.return_indices = _boolean(return_indices, 'return_indices')
        self.ceil_mode = _boolean(ceil_mode, 'ceil_mode')
        self.spatial_rank = rank

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Applies the max pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array | tuple[jax.Array, jax.Array]: The pooled result, and optionally the indices of the maximum values.
        """
        x, unbatched = self._input(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError('MaxPool does not support complex inputs')
        spatial_shape = x.shape[1:-1]
        padding = _pool_padding(
            self.padding,
            spatial_shape,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.ceil_mode,
        )
        _window_output_shape(spatial_shape, self.kernel_size, self.stride,
                             self.dilation, padding)
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
            return self._finish(output, unbatched, out_sharding)

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
            if jnp.issubdtype(x.dtype, jnp.floating):
                right_nan, left_nan = jnp.isnan(right_value), jnp.isnan(left_value)
                choose_right = choose_right | (right_nan & ~left_nan) | (
                    right_nan & left_nan & (right_index < left_index)
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
            self._finish(output, unbatched, out_sharding),
            self._finish(indices, unbatched, out_sharding),
        )

class MaxUnpool(_SpatialOp):
    """Scatter pooled values back to their flattened spatial indices.

    Inputs are ``(*spatial, channels)`` or ``(batch, *spatial, channels)``, with one
    trailing channel axis. Scalars specify 1-D spatial shapes; sequences specify
    the spatial rank. Spatial and channel dimensions must be nonempty.
    __call__ accepts out_sharding for the final output.

    Args:
        kernel_size: Original positive pooling window sizes.
        stride: Original strides; None uses kernel_size.
        padding: Original explicit numeric padding; strings are unsupported.
        dilation: Original positive window dilation; defaults to 1.

    __call__(x, indices, output_size=None, out_sharding=None) requires integer
    indices with the same shape as x. output_size contains spatial dimensions
    only and overrides the shape inferred from kernel, stride, dilation and
    padding. Supply it when ceil-mode or strided pooling made the original
    size ambiguous. Out-of-range indices are discarded, not wrapped. Unfilled
    positions are zero. Repeated indices have unspecified write order, so this
    is only a partial inverse of MaxPool, not a reconstruction of lost values.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(4.)[:, None]
        >>> values, indices = nn.MaxPool(2, return_indices=True)(x)
        >>> nn.MaxUnpool(2)(values, indices)[:, 0].tolist()
        [0.0, 1.0, 0.0, 3.0]
    """

    def __init__(
        self,
        kernel_size: GenericShape,
        stride: GenericShape | None = None,
        padding: int | Sequence[int | tuple[int, int]] = 0,
        dilation: GenericShape = 1,
    ) -> None:
        kernel_size = _positive_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.kernel_size = kernel_size
        self.stride = _positive_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        normalized_padding = _spatial_padding(padding, rank)
        if isinstance(normalized_padding, str):
            raise TypeError('MaxUnpool requires explicit numeric padding')
        self.padding = normalized_padding
        self.dilation = _positive_spatial(
            dilation,
            rank=rank,
            name='dilation',
        )
        self.spatial_rank = rank

    def __call__(
        self,
        x: jax.Array,
        indices: jax.Array,
        output_size: GenericShape | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the unpooling operation.

        Args:
            x (jax.Array): The input array to unpool.
            indices (jax.Array): The indices returned by MaxPool.
            output_size (GenericShape | None, optional): The targeted output size. Defaults to None.

        Returns:
            jax.Array: The unpooled result.
        """
        x, unbatched = self._input(x)
        indices, indices_unbatched = _as_batched(jnp.asarray(indices), self.spatial_rank)
        if indices_unbatched != unbatched or indices.shape != x.shape:
            raise ValueError('indices must have the same shape as the pooled input')
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError('indices must have an integer dtype')

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
            output_size = _positive_spatial(
                output_size,
                rank=self.spatial_rank,
                name='output_size',
            )

        if any(size <= 0 for size in output_size):
            raise ValueError('output_size must contain positive spatial dimensions')
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
            wrap_negative_indices=False,
        )
        output = output.reshape(batch_size, *output_size, channels)
        return self._finish(output, unbatched, out_sharding)

class AvgPool(_SpatialOp):
    """Average channels-last spatial windows.

    Inputs are ``(*spatial, channels)`` or ``(batch, *spatial, channels)``, with one
    trailing channel axis. Scalars specify 1-D spatial shapes; sequences specify
    the spatial rank. Spatial and channel dimensions must be nonempty.
    __call__ accepts out_sharding for the final output.

    Args:
        kernel_size: Positive window sizes.
        stride: Positive window strides; None uses kernel_size.
        padding: VALID, SAME, SAME_LOWER or explicit nonnegative padding.
            Outside-input values contribute zero to each window sum.
        ceil_mode: Permit a final partial window with extra right padding.
        count_include_pad: Include configured zero padding in the divisor.
            Extra padding introduced solely by ceil_mode is never counted.
        divisor_override: Optional positive integer divisor for every window.

    Integer and boolean inputs are promoted to float32. No RNGs or parameters
    are used. With count_include_pad=False, the divisor counts actual input
    elements; configurations containing wholly padded windows have zero count.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.AvgPool(2)(jnp.arange(4.)[:, None])[:, 0].tolist()
        [0.5, 2.5]
    """

    def __init__(
        self,
        kernel_size: GenericShape,
        stride: GenericShape | None = None,
        padding: str | int | Sequence[int | tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        kernel_size = _positive_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        if divisor_override is not None:
            _validate_integer(divisor_override, 'divisor_override')
        self.kernel_size = kernel_size
        self.stride = _positive_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        self.padding = _spatial_padding(padding, rank)
        self.ceil_mode = _boolean(ceil_mode, 'ceil_mode')
        self.count_include_pad = _boolean(count_include_pad, 'count_include_pad')
        self.divisor_override = divisor_override
        self.spatial_rank = rank

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the average pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The pooled result.
        """
        x, unbatched = self._input(x)
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
        _window_output_shape(x.shape[1:-1], self.kernel_size, self.stride,
                             (1,) * self.spatial_rank, padding)
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
        return self._finish(total / divisor, unbatched, out_sharding)

class FractionalMaxPool(_SpatialOp):
    """Max-pool windows placed on a fractional, optionally random spatial grid.

    Inputs are ``(*spatial, channels)`` or ``(batch, *spatial, channels)``, with one
    trailing channel axis. Scalars specify 1-D spatial shapes; sequences specify
    the spatial rank. Spatial and channel dimensions must be nonempty.
    __call__ accepts out_sharding for the final output.

    Args:
        kernel_size: Positive window sizes.
        output_size: Positive target spatial sizes; mutually exclusive with
            output_ratio. Exactly one must be provided.
        output_ratio: Finite ratios in (0, 1], scalar or per-axis. Target sizes
            are max(1, floor(input_size * ratio)). Windows must fit the input.
        return_indices: Also return int32 flattened spatial maximum indices.
        random_samples: Optional fixed array/sequence of shape (spatial_rank,)
            with values in [0, 1). Eager values are validated; traced values
            must satisfy this domain. One grid is shared by batches/channels.
        rngs: Explicit runtime stream. If fixed samples are absent, consumes
            a fresh key on each call; None uses get_context_rng at call time.
            A missing stream raises an error. Fixed samples consume no RNG.

    Sampling also occurs in eval mode: this is grid sampling, not dropout.
    To retain the former deterministic default grid, pass random_samples=(0.5,)
    for 1-D, or one 0.5 per spatial axis. With owned RNGs under jit, pass the
    module in and return its updated state. Values and optional indices both
    receive out_sharding. Complex inputs are unsupported.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> pool = nn.FractionalMaxPool(2, output_size=3, random_samples=(0.5,))
        >>> pool(jnp.arange(6.)[:, None])[:, 0].tolist()
        [1.0, 3.0, 5.0]
    """

    def __init__(
        self,
        kernel_size: GenericShape,
        output_size: GenericShape | None = None,
        output_ratio: float | Sequence[float] | None = None,
        return_indices: bool = False,
        random_samples: jax.Array | Sequence[float] | None = None,
        *,
        rngs: Rngs | None = None,
    ) -> None:
        kernel_size = _positive_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        if (output_size is None) == (output_ratio is None):
            raise ValueError(
                'exactly one of output_size or output_ratio must be provided'
            )
        if output_size is not None:
            output_size = _positive_spatial(
                output_size,
                rank=rank,
                name='output_size',
            )
        if output_ratio is not None:
            if isinstance(output_ratio, (int, float)):
                ratios = (_validate_positive_float(output_ratio, 'output_ratio'),) * rank
            else:
                ratios = tuple(_validate_positive_float(value, 'output_ratio')
                               for value in output_ratio)
            if len(ratios) != rank:
                raise ValueError(
                    f'output_ratio must contain {rank} values, '
                    f'got {len(ratios)}'
                )
            if any(value > 1 for value in ratios):
                raise ValueError('output_ratio values must be in (0, 1]')
            output_ratio = ratios

        if rngs is not None and not isinstance(rngs, Rngs):
            raise TypeError('rngs must be an Rngs or None')
        samples = None
        if random_samples is not None:
            samples = jnp.asarray(random_samples, dtype=jnp.float32)
            if samples.shape != (rank,):
                raise ValueError(
                    f'random_samples must have shape ({rank},), '
                    f'got {samples.shape}'
                )
            if not isinstance(samples, jax.core.Tracer) and not bool(jnp.all(
                jnp.isfinite(samples) & (samples >= 0) & (samples < 1)
            )):
                raise ValueError('random_samples values must be finite and in [0, 1)')

        self.kernel_size = kernel_size
        self.output_size: tuple[int, ...] | None = output_size
        self.output_ratio: tuple[float, ...] | None = output_ratio
        self.return_indices = _boolean(return_indices, 'return_indices')
        self.random_samples = samples
        self.rngs = rngs
        self.spatial_rank = rank

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Applies the fractional max pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array | tuple[jax.Array, jax.Array]: The pooled result, and optionally the indices of the maximum values.
        """
        x, unbatched = self._input(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError('FractionalMaxPool does not support complex inputs')
        spatial_shape = x.shape[1:-1]
        if self.output_size is None:
            assert self.output_ratio is not None
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

        samples = self.random_samples
        if samples is None:
            rngs = self.rngs if self.rngs is not None else get_context_rng()
            samples = jax.random.uniform(rngs(), (self.spatial_rank,))
        starts = []
        for size, kernel, output, sample in zip(
            spatial_shape,
            self.kernel_size,
            output_size,
            samples,
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

        batch_size, channels = x.shape[0], x.shape[-1]
        kernel_volume = math.prod(self.kernel_size)
        positions = tuple(grid.reshape(-1) for grid in jnp.meshgrid(*starts, indexing='ij'))

        def pool_window(start: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array | None]:
            patch = jax.lax.dynamic_slice(
                x,
                (0, *start, 0),
                (batch_size, *self.kernel_size, channels),
            )
            patch = patch.reshape(batch_size, kernel_volume, channels)
            local_index = jnp.argmax(patch, axis=1).astype(jnp.int32)
            value = jnp.take_along_axis(
                patch,
                local_index[:, None, :],
                axis=1,
            )[:, 0, :]
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
                return value, global_index
            return value, None

        values, indices = jax.vmap(pool_window)(positions)
        output = jnp.moveaxis(values, 0, 1).reshape(
            batch_size,
            *output_size,
            channels,
        )
        output = self._finish(output, unbatched, out_sharding)
        if not self.return_indices:
            return output
        assert indices is not None
        index_output = jnp.moveaxis(indices, 0, 1).reshape(
            batch_size,
            *output_size,
            channels,
        )
        return output, self._finish(index_output, unbatched, out_sharding)

class LPPool(_SpatialOp):
    """Compute (sum(abs(x)**p))**(1/p) over spatial windows.

    Inputs are ``(*spatial, channels)`` or ``(batch, *spatial, channels)``, with one
    trailing channel axis. Scalars specify 1-D spatial shapes; sequences specify
    the spatial rank. Spatial and channel dimensions must be nonempty.
    __call__ accepts out_sharding for the final output.

    Args:
        norm_type: Finite positive exponent p. This is a true norm when p>=1;
            values below one are allowed as a power aggregation.
        kernel_size: Positive spatial window sizes.
        stride: Positive strides; None uses kernel_size.
        ceil_mode: Permit a final partial window, treating missing values as zero.

    This is a sum-based Lp aggregation, not a power mean: there is no division
    by window volume. Integer inputs promote to float32; complex inputs use
    real magnitudes. The zero-total output uses a zero derivative convention.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.LPPool(2, 2)(jnp.array([[3.], [4.]]))[:, 0].tolist()
        [5.0]
    """

    def __init__(
        self,
        norm_type: float,
        kernel_size: GenericShape,
        stride: GenericShape | None = None,
        ceil_mode: bool = False,
    ) -> None:
        norm_type = _validate_positive_float(norm_type, 'norm_type')
        kernel_size = _positive_spatial(kernel_size, name='kernel_size')
        rank = len(kernel_size)
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = _positive_spatial(
            kernel_size if stride is None else stride,
            rank=rank,
            name='stride',
        )
        self.ceil_mode = _boolean(ceil_mode, 'ceil_mode')
        self.spatial_rank = rank

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the LPPool operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The pooled result.
        """
        x, unbatched = self._input(x)
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
        _window_output_shape(x.shape[1:-1], self.kernel_size, self.stride,
                             (1,) * self.spatial_rank, padding)
        window, strides, reduce_padding, _ = _reduce_window_config(
            self.spatial_rank,
            self.kernel_size,
            self.stride,
            (1,) * self.spatial_rank,
            padding,
        )
        magnitude = jnp.abs(x)
        powered = jnp.where(
            magnitude > 0,
            jnp.where(magnitude > 0, magnitude, 1) ** self.norm_type,
            0,
        )
        total = jax.lax.reduce_window(
            powered,
            jnp.asarray(0, dtype=powered.dtype),
            jax.lax.add,
            window,
            strides,
            reduce_padding,
        )
        output = jnp.where(total > 0, jnp.where(total > 0, total, 1) **
                           (1.0 / self.norm_type), 0)
        return self._finish(output, unbatched, out_sharding)

class AdaptiveMaxPool(_SpatialOp):
    """Pool variable-size spatial bins to a requested output shape.

    Inputs are ``(*spatial, channels)`` or ``(batch, *spatial, channels)``, with one
    trailing channel axis. Scalars specify 1-D spatial shapes; sequences specify
    the spatial rank. Spatial and channel dimensions must be nonempty.
    __call__ accepts out_sharding for the final output.

    Args:
        output_size: Positive target sizes. A None entry preserves that input
            spatial dimension. A scalar specifies one spatial dimension.
        return_indices: Return values and int32 row-major spatial offsets.

    Bin i spans floor(i * input / output) through ceil((i+1) * input / output),
    excluding the end. Bins may overlap; output sizes may exceed input sizes.
    Each channel is pooled separately. Ties choose the first element in the
    bin. Complex inputs are unsupported. out_sharding applies to both arrays
    when return_indices=True.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.AdaptiveMaxPool(3)(jnp.arange(6.)[:, None])[:, 0].tolist()
        [1.0, 3.0, 5.0]
    """

    def __init__(
        self,
        output_size: int | Sequence[int | None],
        return_indices: bool = False,
    ) -> None:
        self.output_size = _adaptive_size(output_size)
        self.spatial_rank = len(self.output_size)
        self.return_indices = _boolean(return_indices, 'return_indices')

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Applies the adaptive max pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array | tuple[jax.Array, jax.Array]: The pooled result, and optionally the indices of the maximum values.
        """
        rank = len(self.output_size)
        x, unbatched = self._input(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError('AdaptiveMaxPool does not support complex inputs')
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
        values = self._finish(values, unbatched, out_sharding)
        if not self.return_indices:
            return values
        assert indices is not None
        return values, self._finish(indices, unbatched, out_sharding)

class AdaptiveAvgPool(_SpatialOp):
    """Average variable-size spatial bins to a requested output shape.

    Inputs are ``(*spatial, channels)`` or ``(batch, *spatial, channels)``, with one
    trailing channel axis. Scalars specify 1-D spatial shapes; sequences specify
    the spatial rank. Spatial and channel dimensions must be nonempty.
    __call__ accepts out_sharding for the final output.

    Args:
        output_size: Positive target sizes; None entries preserve the matching
            input dimensions. Scalars specify one spatial dimension.

    Bin i spans floor(i * input / output) through ceil((i+1) * input / output),
    excluding the end. Bins may overlap and each is divided by its own number
    of elements. Output sizes may exceed input sizes. Integer and boolean
    inputs promote to float32. This module has no parameters or randomness.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.AdaptiveAvgPool((2, None))(jnp.ones((4, 3, 1))).shape
        (2, 3, 1)
    """

    def __init__(
        self,
        output_size: int | Sequence[int | None],
    ) -> None:
        self.output_size = _adaptive_size(output_size)
        self.spatial_rank = len(self.output_size)

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the adaptive average pooling operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The pooled result.
        """
        rank = len(self.output_size)
        x, unbatched = self._input(x)
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
        return self._finish(values, unbatched, out_sharding)

class Padding(_SpatialOp):
    """Pad arbitrary array axes using jax.numpy.pad conventions.

    Unlike the pooling modules, every axis is eligible, including batch and
    channels. No channels-last interpretation or batch insertion is performed.

    Args:
        padding: Nonnegative integer applied to both ends of every axis; a
            flat (before, after) pair broadcast to every axis; or one such
            pair per array axis. A one-element sequence broadcasts symmetrically.
        mode: constant, edge, reflect, symmetric or wrap. Aliases zeros,
            replicate and circular map to constant, edge and wrap.
        value: Fill value for constant mode only; defaults to 0.

    reflect excludes the edge element when mirroring; symmetric repeats it.
    Numeric padding is normalized to immutable tuples at construction.
    __call__ accepts out_sharding for the final padded array.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> nn.Padding((1, 2), value=9)(jnp.array([1, 2])).tolist()
        [9, 1, 2, 9, 9]
    """

    def __init__(
        self,
        padding: GenericShape | Sequence[tuple[int, int]],
        mode: str = 'constant',
        value: float = 0.0,
    ) -> None:
        if not isinstance(mode, str):
            raise TypeError('mode must be a string')
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
        widths: tuple[int, ...]
        normalized: int | tuple[int, ...] | tuple[tuple[int, int], ...]
        if isinstance(padding, int):
            widths = (padding,)
            normalized = padding
        elif isinstance(padding, Sequence) and not isinstance(padding, (str, bytes)):
            entries = tuple(padding)
            if not entries:
                raise ValueError('padding must not be empty')
            if all(isinstance(entry, int) for entry in entries):
                if len(entries) not in {1, 2}:
                    raise ValueError('flat padding must have one or two values')
                widths = tuple(entry for entry in entries if isinstance(entry, int))
                normalized = widths
            else:
                pairs: list[tuple[int, int]] = []
                for entry in entries:
                    if not isinstance(entry, Sequence) or len(entry) != 2:
                        raise ValueError('padding entries must be (before, after) pairs')
                    pairs.append((entry[0], entry[1]))
                widths = tuple(value for pair in pairs for value in pair)
                normalized = tuple(pairs)
        else:
            raise TypeError('padding must be an integer or sequence')
        if any(isinstance(width, bool) or not isinstance(width, int) or width < 0
               for width in widths):
            raise ValueError('padding widths must be nonnegative integers')
        self.padding = normalized
        self.mode = mode
        self.value = value

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Applies the padding operation.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The padded result.
        """
        if self.mode == 'constant':
            output = jnp.pad(
                x,
                self.padding,
                mode=self.mode,
                constant_values=self.value,
            )
        else:
            output = jnp.pad(x, self.padding, mode=self.mode)
        return self._finish(output, False, out_sharding)


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
