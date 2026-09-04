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
"""Recurrent neural-network cells and sequence layers."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax.lax import PrecisionLike
from jax.nn import initializers
from jax.sharding import PartitionSpec
from jax.typing import DTypeLike

from taktiny.nn.base import Module
from taktiny.nn.block import List
from taktiny.nn.modules.linear import Linear
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import _constrain, _validate_integer
from taktiny.utils.typing import (
    AxisNames,
    DotGeneral,
    DType,
    Initializer,
    MetaData,
    QuantConfig,
)

type Nonlinearity = (
    Literal['tanh', 'relu'] | Callable[[jax.Array], jax.Array]
)
type CellState = tuple[jax.Array, ...]

default_recurrent_initializer = initializers.lecun_uniform()
default_recurrent_bias_initializer = initializers.zeros


def _validate_projection_axes(
    axis_names: AxisNames | None,
    partition_spec: PartitionSpec | None,
) -> tuple[AxisNames | None, PartitionSpec | None]:
    """Validate the logical and physical axes of a recurrent projection.

    Args:
        axis_names: Optional logical names for the input and hidden axes.
        partition_spec: Optional partition specifications for the input and
            hidden axes.

    Returns:
        The normalized ``(axis_names, partition_spec)`` pair.

    Raises:
        TypeError: If ``axis_names`` is a :class:`PartitionSpec`.
        ValueError: If either argument does not describe exactly two axes.
    """
    if axis_names is not None:
        if isinstance(axis_names, PartitionSpec):
            raise TypeError(
                "Passing a PartitionSpec to 'axis_names' is forbidden; "
                "use 'partition_spec' instead"
            )
        axis_names = tuple(axis_names)
        if len(axis_names) != 2:
            raise ValueError(
                'axis_names must contain input and hidden logical axes'
            )
    if partition_spec is not None:
        partition_spec = PartitionSpec(*partition_spec)
        if len(partition_spec) != 2:
            raise ValueError(
                'partition_spec must contain input and hidden specifications'
            )
    return axis_names, partition_spec


def _projection_axis_names(
    axis_names: AxisNames | None,
    gates: int,
    *,
    recurrent: bool,
) -> AxisNames | None:
    """Expand semantic axis names for a packed-gate projection.

    Args:
        axis_names: Logical names for the input and hidden axes.
        gates: Number of packed gates in the projection output.
        recurrent: Whether this is a hidden-to-hidden projection. Recurrent
            contracting axes are left unnamed to avoid duplicate mesh axes.

    Returns:
        Logical names matching the projection kernel, or ``None``.
    """
    if axis_names is None:
        return None
    input_axis = None if recurrent else axis_names[0]
    hidden_axis = axis_names[1]
    if gates == 1:
        return (input_axis, hidden_axis)
    return (input_axis, None, hidden_axis)


def _projection_partition_spec(
    partition_spec: PartitionSpec | None,
    gates: int,
    *,
    recurrent: bool,
) -> PartitionSpec | None:
    """Expand a partition specification for a packed-gate projection.

    Args:
        partition_spec: Partition specifications for the input and hidden
            axes.
        gates: Number of packed gates in the projection output.
        recurrent: Whether this is a hidden-to-hidden projection. Recurrent
            contracting axes are left unsharded.

    Returns:
        A partition specification matching the projection kernel, or ``None``.
    """
    if partition_spec is None:
        return None
    input_spec = None if recurrent else partition_spec[0]
    hidden_spec = partition_spec[1]
    if gates == 1:
        return PartitionSpec(input_spec, hidden_spec)
    return PartitionSpec(input_spec, None, hidden_spec)


def _resolve_nonlinearity(
    nonlinearity: Nonlinearity,
) -> tuple[Callable[[jax.Array], jax.Array], str]:
    """Resolve a named or callable RNN activation.

    Args:
        nonlinearity: ``'tanh'``, ``'relu'``, or a callable activation.

    Returns:
        The activation function and its display name.

    Raises:
        TypeError: If ``nonlinearity`` is neither a string nor callable.
        ValueError: If a string other than ``'tanh'`` or ``'relu'`` is given.
    """
    if isinstance(nonlinearity, str):
        name = nonlinearity.lower()
        functions = {'tanh': jnp.tanh, 'relu': jax.nn.relu}
        if name not in functions:
            raise ValueError(
                "nonlinearity must be 'tanh', 'relu', or callable"
            )
        return functions[name], name
    if callable(nonlinearity):
        return nonlinearity, getattr(
            nonlinearity,
            '__name__',
            type(nonlinearity).__name__,
        )
    raise TypeError('nonlinearity must be a string or callable')


def _validate_cell_inputs(
    x: jax.Array,
    state: CellState,
    *,
    input_size: int,
    state_sizes: tuple[int, ...],
) -> tuple[jax.Array, CellState]:
    """Validate one time-step input and its recurrent state.

    Args:
        x: Input with shape ``(..., input_size)``.
        state: Tuple of recurrent state arrays.
        input_size: Expected size of the input's trailing dimension.
        state_sizes: Expected trailing size of each state array.

    Returns:
        The validated input and state.

    Raises:
        TypeError: If the state is not a tuple or an array is not floating
            point.
        ValueError: If state counts, feature sizes, or leading shapes differ.
    """
    if not isinstance(state, tuple):
        raise TypeError('state must be a tuple of arrays')

    if x.ndim == 0 or x.shape[-1] != input_size:
        raise ValueError(
            f'expected input trailing dimension {input_size}, got {x.shape}'
        )

    if len(state) != len(state_sizes):
        raise ValueError(f'expected {len(state_sizes)} state arrays')

    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError('recurrent inputs must have a floating-point dtype')

    for index, (component, size) in enumerate(zip(state, state_sizes)):
        if component.ndim == 0 or component.shape[-1] != size:
            raise ValueError(
                f'state {index} must have trailing dimension {size}, '
                f'got {component.shape}'
            )

        if component.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f'state {index} leading shape {component.shape[:-1]} must '
                f'match input leading shape {x.shape[:-1]}'
            )

        if not jnp.issubdtype(component.dtype, jnp.floating):
            raise TypeError('recurrent states must have a floating-point dtype')

    return x, state


def _initial_state(
    sizes: tuple[int, ...],
    batch_shape: Sequence[int],
    dtype: DType,
) -> CellState:
    batch_shape = tuple(batch_shape)
    for index, size in enumerate(batch_shape):
        _validate_integer(size, f'batch_shape[{index}]', minimum=0)

    return tuple(
        jnp.zeros((*batch_shape, size), dtype=dtype)
        for size in sizes
    )


class RNNCell(Module):
    r"""Apply one Elman recurrent-network step.

    .. math::

        h_t = \phi\left(
            x_t W_{ih} + b_{ih}
            + h_{t-1} W_{hh} + b_{hh}
        \right)

    Here, :math:`\phi` is ``tanh``, ``relu``, or a custom activation. The
    output is :math:`h_t`. State is stored as the one-element tuple
    ``(hidden,)`` so the cell follows the ``(state, input)`` convention used
    by :func:`jax.lax.scan`. Both bias terms are omitted when ``bias=False``.

    ``axis_names`` and ``partition_spec`` describe the semantic input and
    hidden dimensions. The recurrent kernel leaves its contracting dimension
    unassigned, preventing the same logical mesh axis from appearing twice.

    Args:
        input_size: Number of features in :math:`x_t`.
        hidden_size: Number of features in :math:`h_t`.
        nonlinearity: Hidden-state activation: ``'tanh'``, ``'relu'``, or a
            callable. Defaults to ``'tanh'``.
        bias: Whether both projections include a learnable bias. Defaults to
            ``True``.
        dtype: Data type passed to parameter initializers. Defaults to each
            initializer's default data type.
        rngs: Random number generator used to initialize parameters.
        kernel_initializer: Input-kernel initializer. Defaults to LeCun
            uniform initialization.
        recurrent_initializer: Recurrent-kernel initializer. Defaults to
            ``kernel_initializer``.
        bias_initializer: Bias initializer. Defaults to zeros.
        quant: Optional Qwix quantization configuration for both kernels.
        dot_general: Optional ``dot_general`` implementation used by the
            projections.
        axis_names: Optional logical names for the input and hidden axes.
        partition_spec: Optional partition specifications for the input and
            hidden axes.
        kernel_metadata: Optional metadata attached to both kernel parameters.
        bias_metadata: Optional metadata attached to both bias parameters.
        precision: Dot-product precision forwarded to the projections.
        preferred_element_type: Preferred result and accumulation data type
            forwarded to the projections.

    Examples:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> cell = nn.RNNCell(3, 4, rngs=nn.Rngs(0))
        >>> state = cell.initial_state((2,))
        >>> next_state, output = cell(state, jnp.ones((2, 3)))
        >>> (next_state[0].shape, output.shape)
        ((2, 4), (2, 4))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        nonlinearity: Nonlinearity = 'tanh',
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_recurrent_initializer,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer = default_recurrent_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:

        self.input_size = _validate_integer(input_size, 'input_size')
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.nonlinearity, self.nonlinearity_name = _resolve_nonlinearity(
            nonlinearity
        )
        axis_names, partition_spec = _validate_projection_axes(
            axis_names,
            partition_spec,
        )
        recurrent_initializer = (
            kernel_initializer
            if recurrent_initializer is None
            else recurrent_initializer
        )

        self.input_proj = Linear(
            input_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_projection_axis_names(
                axis_names,
                1,
                recurrent=False,
            ),
            partition_spec=_projection_partition_spec(
                partition_spec,
                1,
                recurrent=False,
            ),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.hidden_proj = Linear(
            hidden_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_projection_axis_names(
                axis_names,
                1,
                recurrent=True,
            ),
            partition_spec=_projection_partition_spec(
                partition_spec,
                1,
                recurrent=True,
            ),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    def initial_state(
        self,
        batch_shape: Sequence[int] = (),
        dtype: DType = jnp.float32,
    ) -> tuple[jax.Array]:
        """Create a zero-valued hidden state.

        Args:
            batch_shape: Leading dimensions of the state. Defaults to no
                leading dimensions.
            dtype: State data type. Defaults to ``jnp.float32``.

        Returns:
            A one-element tuple containing an array of shape
            ``(*batch_shape, hidden_size)``.
        """
        state = _initial_state((self.hidden_size,), batch_shape, dtype)
        return (state[0],)

    def __call__(
        self,
        state: tuple[jax.Array],
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[tuple[jax.Array], jax.Array]:
        """Advance the cell by one time step.

        Args:
            state: One-element tuple containing the previous hidden state with
                shape ``(..., hidden_size)``.
            x: Current input with shape ``(..., input_size)`` and the same
                leading dimensions as the state.
            out_sharding: Optional sharding forwarded to both projections.

        Returns:
            ``((hidden,), hidden)``, where each hidden array has shape
            ``(..., hidden_size)``.

        Raises:
            TypeError: If inputs or state are not floating-point arrays, or
                if ``state`` is not a tuple.
            ValueError: If state counts, feature sizes, or leading shapes are
                invalid.
        """
        x, validated_state = _validate_cell_inputs(
            x,
            state,
            input_size=self.input_size,
            state_sizes=(self.hidden_size,),
        )
        hidden = self.nonlinearity(
            self.input_proj(x, out_sharding) + self.hidden_proj(validated_state[0], out_sharding)
        )
        return (hidden,), hidden

    def extra_repr(self) -> str:
        return (
            f'{self.input_size} ➤ {self.hidden_size}, '
            f'nonlinearity={self.nonlinearity_name}'
        )


class LSTMCell(Module):
    r"""Apply one long short-term memory step.

    .. math::

        \begin{aligned}
        i_t &= \sigma(x_t W_{ii} + b_{ii}
                       + h_{t-1} W_{hi} + b_{hi}) \\
        f_t &= \sigma(x_t W_{if} + b_{if}
                       + h_{t-1} W_{hf} + b_{hf}) \\
        g_t &= \tanh(x_t W_{ig} + b_{ig}
                      + h_{t-1} W_{hg} + b_{hg}) \\
        o_t &= \sigma(x_t W_{io} + b_{io}
                       + h_{t-1} W_{ho} + b_{ho}) \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
        \widetilde{h}_t &= o_t \odot \tanh(c_t)
        \end{aligned}

    Without projection, :math:`h_t = \widetilde{h}_t`. When ``proj_size`` is
    nonzero, the exposed hidden state is instead

    .. math::

        h_t = \widetilde{h}_t W_{hr}.

    State is ``(hidden, cell)``. A projected hidden state has width
    ``proj_size``, while the cell state always retains ``hidden_size``. Both
    bias terms in each gate are omitted when ``bias=False``.

    Args:
        input_size: Number of features in :math:`x_t`.
        hidden_size: Number of features in the cell state :math:`c_t`.
        proj_size: Width of the exposed hidden state. ``0`` disables the
            projection. A nonzero value must be smaller than ``hidden_size``.
        bias: Whether the input and recurrent gate projections include
            learnable biases. Defaults to ``True``.
        dtype: Data type passed to parameter initializers. Defaults to each
            initializer's default data type.
        rngs: Random number generator used to initialize parameters.
        kernel_initializer: Input-kernel initializer. Defaults to LeCun
            uniform initialization.
        recurrent_initializer: Recurrent and output-projection initializer.
            Defaults to ``kernel_initializer``.
        bias_initializer: Gate-bias initializer. Defaults to zeros.
        quant: Optional Qwix quantization configuration for all kernels.
        dot_general: Optional ``dot_general`` implementation used by the
            projections.
        axis_names: Optional logical names for the input and hidden axes.
        partition_spec: Optional partition specifications for the input and
            hidden axes.
        kernel_metadata: Optional metadata attached to kernel parameters.
        bias_metadata: Optional metadata attached to gate-bias parameters.
        precision: Dot-product precision forwarded to the projections.
        preferred_element_type: Preferred result and accumulation data type
            forwarded to the projections.

    Examples:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> cell = nn.LSTMCell(3, 5, proj_size=2, rngs=nn.Rngs(0))
        >>> state = cell.initial_state((4,))
        >>> next_state, output = cell(state, jnp.ones((4, 3)))
        >>> (output.shape, next_state[1].shape)
        ((4, 2), (4, 5))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        proj_size: int = 0,
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_recurrent_initializer,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer = default_recurrent_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.input_size = _validate_integer(input_size, 'input_size')
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.proj_size = _validate_integer(
            proj_size,
            'proj_size',
            minimum=0,
        )
        if self.proj_size >= hidden_size and self.proj_size != 0:
            raise ValueError('proj_size must be smaller than hidden_size')

        self.output_size = self.proj_size or hidden_size
        axis_names, partition_spec = _validate_projection_axes(
            axis_names,
            partition_spec,
        )
        recurrent_initializer = (
            kernel_initializer
            if recurrent_initializer is None
            else recurrent_initializer
        )

        self.input_proj = Linear(
            input_size,
            (4, hidden_size),
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_projection_axis_names(
                axis_names,
                4,
                recurrent=False,
            ),
            partition_spec=_projection_partition_spec(
                partition_spec,
                4,
                recurrent=False,
            ),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.hidden_proj = Linear(
            self.output_size,
            (4, hidden_size),
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_projection_axis_names(
                axis_names,
                4,
                recurrent=True,
            ),
            partition_spec=_projection_partition_spec(
                partition_spec,
                4,
                recurrent=True,
            ),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.projection = None
        if self.proj_size:
            projection_axis_names = (
                None
                if axis_names is None
                else (None, axis_names[1])
            )
            projection_spec = (
                None
                if partition_spec is None
                else PartitionSpec(None, partition_spec[1])
            )
            self.projection = Linear(
                hidden_size,
                self.output_size,
                bias=False,
                dtype=dtype,
                rngs=rngs,
                kernel_initializer=recurrent_initializer,
                quant=quant,
                dot_general=dot_general,
                axis_names=projection_axis_names,
                partition_spec=projection_spec,
                kernel_metadata=kernel_metadata,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )

    def initial_state(
        self,
        batch_shape: Sequence[int] = (),
        dtype: DType = jnp.float32,
    ) -> tuple[jax.Array, jax.Array]:
        """Create zero-valued hidden and cell states.

        Args:
            batch_shape: Leading dimensions of both states. Defaults to no
                leading dimensions.
            dtype: State data type. Defaults to ``jnp.float32``.

        Returns:
            ``(hidden, cell)`` with shapes
            ``(*batch_shape, output_size)`` and
            ``(*batch_shape, hidden_size)``, respectively. ``output_size`` is
            ``proj_size`` when projection is enabled and ``hidden_size``
            otherwise.
        """
        state = _initial_state(
            (self.output_size, self.hidden_size),
            batch_shape,
            dtype,
        )
        return state[0], state[1]

    def __call__(
        self,
        state: tuple[jax.Array, jax.Array],
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        """Advance the cell by one time step.

        Args:
            state: ``(hidden, cell)`` from the previous step. Their trailing
                dimensions must be ``output_size`` and ``hidden_size``.
            x: Current input with shape ``(..., input_size)`` and the same
                leading dimensions as both state arrays.
            out_sharding: Optional sharding constraint for the new hidden
                state and output.

        Returns:
            ``((hidden, cell), hidden)``. The hidden arrays have shape
            ``(..., output_size)`` and the cell has shape
            ``(..., hidden_size)``.

        Raises:
            TypeError: If inputs or state are not floating-point arrays, or
                if ``state`` is not a tuple.
            ValueError: If state counts, feature sizes, or leading shapes are
                invalid.
        """
        x, validated_state = _validate_cell_inputs(
            x,
            state,
            input_size=self.input_size,
            state_sizes=(self.output_size, self.hidden_size),
        )
        hidden = validated_state[0]
        cell = validated_state[1]
        gates = self.input_proj(x) + self.hidden_proj(hidden)
        input_gate, forget_gate, candidate, output_gate = jnp.unstack(
            gates,
            axis=-2,
        )
        input_gate = jax.nn.sigmoid(input_gate)
        forget_gate = jax.nn.sigmoid(forget_gate)
        candidate = jnp.tanh(candidate)
        output_gate = jax.nn.sigmoid(output_gate)

        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * jnp.tanh(cell)
        if self.projection is not None:
            hidden = self.projection(hidden)

        hidden = _constrain(hidden, out_sharding)
        return (hidden, cell), hidden

    def extra_repr(self) -> str:
        representation = f'{self.input_size} ➤ {self.hidden_size}'
        if self.proj_size:
            representation += f' ➤ {self.proj_size}'
        return representation


class GRUCell(Module):
    r"""Apply one gated recurrent unit step.

    .. math::

        \begin{aligned}
        r_t &= \sigma(x_t W_{ir} + b_{ir}
                       + h_{t-1} W_{hr} + b_{hr}) \\
        z_t &= \sigma(x_t W_{iz} + b_{iz}
                       + h_{t-1} W_{hz} + b_{hz}) \\
        n_t &= \tanh\left(x_t W_{in} + b_{in}
                    + r_t \odot (h_{t-1} W_{hn} + b_{hn})\right) \\
        h_t &= (1-z_t) \odot n_t + z_t \odot h_{t-1}
        \end{aligned}

    Here, :math:`r_t`, :math:`z_t`, and :math:`n_t` are the reset, update,
    and candidate activations. This implementation uses the reset-after form:
    the reset gate is applied after the candidate's recurrent projection.
    State is stored as ``(hidden,)``, and the output is the new hidden state.

    Args:
        input_size: Number of features in :math:`x_t`.
        hidden_size: Number of features in :math:`h_t`.
        bias: Whether the input and recurrent gate projections include
            learnable biases. Defaults to ``True``.
        dtype: Data type passed to parameter initializers. Defaults to each
            initializer's default data type.
        rngs: Random number generator used to initialize parameters.
        kernel_initializer: Input-kernel initializer. Defaults to LeCun
            uniform initialization.
        recurrent_initializer: Recurrent-kernel initializer. Defaults to
            ``kernel_initializer``.
        bias_initializer: Gate-bias initializer. Defaults to zeros.
        quant: Optional Qwix quantization configuration for both kernels.
        dot_general: Optional ``dot_general`` implementation used by the
            projections.
        axis_names: Optional logical names for the input and hidden axes.
        partition_spec: Optional partition specifications for the input and
            hidden axes.
        kernel_metadata: Optional metadata attached to both kernel parameters.
        bias_metadata: Optional metadata attached to both bias parameters.
        precision: Dot-product precision forwarded to the projections.
        preferred_element_type: Preferred result and accumulation data type
            forwarded to the projections.

    Examples:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> cell = nn.GRUCell(3, 4, rngs=nn.Rngs(0))
        >>> state = cell.initial_state((2,))
        >>> next_state, output = cell(state, jnp.ones((2, 3)))
        >>> (next_state[0].shape, output.shape)
        ((2, 4), (2, 4))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_recurrent_initializer,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer = default_recurrent_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.input_size = _validate_integer(input_size, 'input_size')
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        axis_names, partition_spec = _validate_projection_axes(
            axis_names,
            partition_spec,
        )
        recurrent_initializer = (
            kernel_initializer
            if recurrent_initializer is None
            else recurrent_initializer
        )

        self.input_proj = Linear(
            input_size,
            (3, hidden_size),
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_projection_axis_names(
                axis_names,
                3,
                recurrent=False,
            ),
            partition_spec=_projection_partition_spec(
                partition_spec,
                3,
                recurrent=False,
            ),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        self.hidden_proj = Linear(
            hidden_size,
            (3, hidden_size),
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=_projection_axis_names(
                axis_names,
                3,
                recurrent=True,
            ),
            partition_spec=_projection_partition_spec(
                partition_spec,
                3,
                recurrent=True,
            ),
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    def initial_state(
        self,
        batch_shape: Sequence[int] = (),
        dtype: DType = jnp.float32,
    ) -> tuple[jax.Array]:
        """Create a zero-valued hidden state.

        Args:
            batch_shape: Leading dimensions of the state. Defaults to no
                leading dimensions.
            dtype: State data type. Defaults to ``jnp.float32``.

        Returns:
            A one-element tuple containing an array of shape
            ``(*batch_shape, hidden_size)``.
        """
        state = _initial_state((self.hidden_size,), batch_shape, dtype)
        return (state[0],)

    def __call__(
        self,
        state: tuple[jax.Array],
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[tuple[jax.Array], jax.Array]:
        """Advance the cell by one time step.

        Args:
            state: One-element tuple containing the previous hidden state with
                shape ``(..., hidden_size)``.
            x: Current input with shape ``(..., input_size)`` and the same
                leading dimensions as the state.
            out_sharding: Optional sharding constraint for the new hidden
                state and output.

        Returns:
            ``((hidden,), hidden)``, where each hidden array has shape
            ``(..., hidden_size)``.

        Raises:
            TypeError: If inputs or state are not floating-point arrays, or
                if ``state`` is not a tuple.
            ValueError: If state counts, feature sizes, or leading shapes are
                invalid.
        """
        x, validated_state = _validate_cell_inputs(
            x,
            state,
            input_size=self.input_size,
            state_sizes=(self.hidden_size,),
        )
        hidden = validated_state[0]
        input_reset, input_update, input_candidate = jnp.unstack(
            self.input_proj(x),
            axis=-2,
        )
        hidden_reset, hidden_update, hidden_candidate = jnp.unstack(
            self.hidden_proj(hidden),
            axis=-2,
        )

        reset = jax.nn.sigmoid(input_reset + hidden_reset)
        update = jax.nn.sigmoid(input_update + hidden_update)
        candidate = jnp.tanh(input_candidate + reset * hidden_candidate)
        hidden = (1.0 - update) * candidate + update * hidden
        hidden = _constrain(hidden, out_sharding)
        return (hidden,), hidden

    def extra_repr(self) -> str:
        return f'{self.input_size} ➤ {self.hidden_size}'


class RecurrentLayer(Module):
    """Hold the forward and optional reverse cells for one recurrent layer.

    Args:
        forward_cell: Cell applied in sequence order.
        reverse_cell: Optional cell applied in reverse sequence order. If
            present, the layer is bidirectional.

    Attributes:
        forward_cell: The forward recurrent cell.
        reverse_cell: The reverse recurrent cell, or ``None``.
    """

    def __init__(
        self,
        forward_cell: Module,
        reverse_cell: Module | None = None,
    ) -> None:
        if not isinstance(forward_cell, Module):
            raise TypeError('forward_cell must be a Module')

        if reverse_cell is not None and not isinstance(reverse_cell, Module):
            raise TypeError('reverse_cell must be a Module or None')

        self.forward_cell = forward_cell
        self.reverse_cell = reverse_cell


class RecurrentBase(Module):
    """Provide stacked sequence traversal for recurrent network subclasses.

    Each layer is evaluated with :func:`jax.lax.scan`. A bidirectional layer
    scans independently in both directions and concatenates their outputs on
    the feature axis. Dropout is applied between stacked layers only, never
    after the final layer.

    This is the shared implementation behind :class:`RNN`, :class:`LSTM`, and
    :class:`GRU`; users normally instantiate one of those subclasses.

    Args:
        input_size: Number of features in each input time step.
        hidden_size: Internal hidden width reported by the subclass.
        num_layers: Number of stacked recurrent layers.
        bias: Whether recurrent cells use biases.
        batch_first: Whether batched inputs use ``(batch, sequence, feature)``
            instead of ``(sequence, batch, feature)``.
        dropout: Dropout probability between recurrent layers.
        bidirectional: Whether every layer scans in both directions.
        dtype: Parameter data type requested by the subclass.
        axis_names: Optional logical names for input and hidden axes.
        unroll: Loop-unrolling option forwarded to :func:`jax.lax.scan`.
        output_size: Per-direction output width of one cell.
        state_sizes: Trailing width of each state component.
        cell_factory: Factory receiving a layer input width and its logical
            axis name, and returning a recurrent cell.

    Attributes:
        layers: Recurrent layers in stack order.
        num_directions: ``2`` for a bidirectional network, otherwise ``1``.
        output_size: Per-direction output width.
        state_sizes: Trailing width of each recurrent state component.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        bias: bool,
        batch_first: bool,
        dropout: float,
        bidirectional: bool,
        dtype: DType | None,
        axis_names: AxisNames | None,
        unroll: int | bool,
        output_size: int,
        state_sizes: tuple[int, ...],
        cell_factory: Callable[[int, str | None], Module],
    ) -> None:

        self.input_size = _validate_integer(input_size, 'input_size')
        self.hidden_size = _validate_integer(hidden_size, 'hidden_size')
        self.num_layers = _validate_integer(num_layers, 'num_layers')
        if not isinstance(bias, bool):
            raise TypeError('bias must be a boolean')

        if not isinstance(batch_first, bool):
            raise TypeError('batch_first must be a boolean')

        if not isinstance(bidirectional, bool):
            raise TypeError('bidirectional must be a boolean')

        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError('dropout must be a number')

        if not 0.0 <= dropout <= 1.0:
            raise ValueError('dropout must be between 0 and 1')

        if not isinstance(unroll, (int, bool)) or (
            isinstance(unroll, int)
            and not isinstance(unroll, bool)
            and unroll <= 0
        ):
            raise ValueError('unroll must be a positive integer or boolean')

        axis_names, _ = _validate_projection_axes(axis_names, None)

        self.has_bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dtype = dtype
        self.unroll = unroll
        self.output_size = output_size
        self.state_sizes = state_sizes

        input_axis = None if axis_names is None else axis_names[0]
        layers = []
        layer_input_size = input_size
        for layer_index in range(num_layers):
            layer_input_axis = input_axis if layer_index == 0 else None
            forward_cell = cell_factory(layer_input_size, layer_input_axis)
            reverse_cell = (
                cell_factory(layer_input_size, layer_input_axis)
                if bidirectional
                else None
            )
            layers.append(RecurrentLayer(forward_cell, reverse_cell))
            layer_input_size = self.num_directions * output_size
        self.layers = List(layers)

    def _prepare_input(
        self,
        x: jax.Array,
    ) -> tuple[jax.Array, bool, int]:
        x = jnp.asarray(x)
        if x.ndim not in {2, 3}:
            raise ValueError(
                'input must have shape [sequence, input], '
                '[sequence, batch, input], or [batch, sequence, input]'
            )

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f'expected input_size={self.input_size}, got {x.shape[-1]}'
            )

        if not jnp.issubdtype(x.dtype, jnp.floating):
            raise TypeError('recurrent inputs must have a floating-point dtype')

        unbatched = x.ndim == 2
        if unbatched:
            x = x[:, None, :]
        elif self.batch_first:
            x = jnp.swapaxes(x, 0, 1)
        return x, unbatched, x.shape[1]

    def _prepare_states(
        self,
        hx: jax.Array | Sequence[jax.Array] | None,
        *,
        batch_size: int,
        unbatched: bool,
        dtype: DType,
    ) -> CellState:
        state_count = self.num_layers * self.num_directions
        if hx is None:
            return tuple(
                jnp.zeros((state_count, batch_size, width), dtype=dtype)
                for width in self.state_sizes
            )

        if len(self.state_sizes) == 1:
            values = (jnp.asarray(hx),)
        else:
            if not isinstance(hx, Sequence) or len(hx) != len(self.state_sizes):
                raise ValueError(
                    f'initial state must contain {len(self.state_sizes)} arrays'
                )
            values = tuple(jnp.asarray(value) for value in hx)

        normalized = []
        for index, (value, width) in enumerate(zip(values, self.state_sizes)):
            expected = (
                (state_count, width)
                if unbatched
                else (state_count, batch_size, width)
            )
            if value.shape != expected:
                raise ValueError(
                    f'initial state {index} must have shape {expected}, '
                    f'got {value.shape}'
                )
            if not jnp.issubdtype(value.dtype, jnp.floating):
                raise TypeError('recurrent states must have a floating-point dtype')
            normalized.append(value[:, None, :] if unbatched else value)
        return tuple(normalized)

    def _apply_dropout(self, x: jax.Array, key: jax.Array) -> jax.Array:
        if self.dropout == 1.0:
            return jnp.zeros_like(x)
        keep_probability = 1.0 - self.dropout
        mask = jax.random.bernoulli(key, keep_probability, x.shape)
        return jnp.where(mask, x / keep_probability, jnp.zeros_like(x))

    def __call__(
        self,
        x: jax.Array,
        hx: jax.Array | Sequence[jax.Array] | None = None,
        *,
        rngs: Rngs | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, Any]:
        """Apply the recurrent stack to a complete sequence.

        Args:
            x: Input sequence. Its shape is ``(sequence, input_size)`` when
                unbatched, ``(sequence, batch, input_size)`` when batched, or
                ``(batch, sequence, input_size)`` when ``batch_first=True``.
            hx: Optional initial state. Each state component has leading size
                ``num_layers * num_directions``, followed by the batch axis
                when present and its corresponding width from ``state_sizes``.
                A single-state network accepts an array; an LSTM accepts
                ``(hidden, cell)``. Defaults to zero-valued states.
            rngs: Random number generator for inter-layer dropout. It is
                required only in training mode when dropout is nonzero and
                ``num_layers > 1``.
            out_sharding: Optional sharding constraint for the output sequence.

        Returns:
            ``(output, final_state)``. The output preserves the input's
            sequence and batch layout and has feature width
            ``num_directions * output_size``. Final states are ordered by
            layer, then direction.

        Raises:
            TypeError: If an input or state is not floating point.
            ValueError: If an input or state shape is invalid, or if dropout
                is active but ``rngs`` is missing.
        """
        x, unbatched, batch_size = self._prepare_input(x)
        states = self._prepare_states(
            hx,
            batch_size=batch_size,
            unbatched=unbatched,
            dtype=x.dtype,
        )

        use_dropout = (
            self.is_training
            and self.dropout > 0.0
            and self.num_layers > 1
        )
        if use_dropout:
            if rngs is None:
                raise ValueError(
                    'rngs is required when recurrent dropout is active'
                )
            dropout_keys = jax.random.split(rngs(), self.num_layers - 1)
        else:
            dropout_keys = ()

        final_states: list[list[jax.Array]] = [
            [] for _ in self.state_sizes
        ]
        layer_input = x
        for layer_index, layer_module in enumerate(self.layers):
            layer = cast(RecurrentLayer, layer_module)
            direction_outputs = []
            cells = [layer.forward_cell]
            if layer.reverse_cell is not None:
                cells.append(layer.reverse_cell)

            for direction, cell in enumerate(cells):
                state_index = layer_index * self.num_directions + direction
                initial_state = tuple(
                    state[state_index] for state in states
                )

                def step(
                    carry: CellState,
                    value: jax.Array,
                    cell: Module = cell,
                ) -> tuple[CellState, jax.Array]:
                    return cell(carry, value)

                final_state, direction_output = jax.lax.scan(
                    step,
                    initial_state,
                    layer_input,
                    reverse=direction == 1,
                    unroll=self.unroll,
                )
                direction_outputs.append(direction_output)
                for component, value in zip(final_states, final_state):
                    component.append(value)

            layer_input = (
                direction_outputs[0]
                if self.num_directions == 1
                else jnp.concatenate(direction_outputs, axis=-1)
            )
            if use_dropout and layer_index < self.num_layers - 1:
                layer_input = self._apply_dropout(
                    layer_input,
                    dropout_keys[layer_index],
                )

        output = layer_input
        hidden = tuple(jnp.stack(component) for component in final_states)
        if unbatched:
            output = output[:, 0, :]
            hidden = tuple(component[:, 0, :] for component in hidden)
        elif self.batch_first:
            output = jnp.swapaxes(output, 0, 1)

        output = _constrain(output, out_sharding)
        return output, hidden[0] if len(hidden) == 1 else hidden

    def extra_repr(self) -> str:
        options = [f'{self.input_size} ➤ {self.hidden_size}']
        if self.num_layers != 1:
            options.append(f'layers={self.num_layers}')
        if self.bidirectional:
            options.append('bidirectional=True')
        if self.dropout:
            options.append(f'dropout={self.dropout:g}')
        return ', '.join(options)


class RNN(RecurrentBase):
    r"""Apply a stacked Elman RNN to a complete sequence.

    At each time step and layer, the recurrence is

    .. math::

        h_t = \phi\left(
            x_t W_{ih} + b_{ih}
            + h_{t-1} W_{hh} + b_{hh}
        \right).

    Inputs are ``[sequence, input]`` when unbatched. Batched inputs are
    ``[sequence, batch, input]`` unless ``batch_first=True``. The returned
    hidden state is ordered by layer, then direction. Bidirectional outputs
    concatenate the forward and reverse hidden states on the feature axis.

    Args:
        input_size: Number of features in each input time step.
        hidden_size: Number of features in each directional hidden state.
        num_layers: Number of stacked recurrent layers. Defaults to ``1``.
        nonlinearity: ``'tanh'``, ``'relu'``, or a callable activation.
            Defaults to ``'tanh'``.
        bias: Whether both cell projections include biases. Defaults to
            ``True``.
        batch_first: Whether batched inputs use batch-major layout. Defaults
            to ``False``.
        dropout: Dropout probability between layers. Dropout is active only
            during training and is not applied after the last layer. Defaults
            to ``0.0``.
        bidirectional: Whether each layer scans in both directions. Defaults
            to ``False``.
        dtype: Data type passed to parameter initializers.
        rngs: Random number generator used to initialize all cells.
        kernel_initializer: Input-kernel initializer. Defaults to LeCun
            uniform initialization.
        recurrent_initializer: Recurrent-kernel initializer. Defaults to
            ``kernel_initializer``.
        bias_initializer: Bias initializer. Defaults to zeros.
        quant: Optional Qwix quantization configuration for projection kernels.
        dot_general: Optional ``dot_general`` implementation used by cells.
        axis_names: Optional logical names for the input and hidden axes.
        partition_spec: Optional partition specifications for the input and
            hidden axes.
        kernel_metadata: Optional metadata attached to kernel parameters.
        bias_metadata: Optional metadata attached to bias parameters.
        precision: Dot-product precision forwarded to cell projections.
        preferred_element_type: Preferred result and accumulation data type
            forwarded to cell projections.
        unroll: Loop-unrolling option passed to :func:`jax.lax.scan`. Defaults
            to ``1``.

    References:
        - Elman, J. L. (1990). `Finding Structure in Time
          <https://doi.org/10.1207/S15516709COG1402_1>`_.

    Examples:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> rnn = nn.RNN(3, 4, rngs=nn.Rngs(0))
        >>> output, hidden = rnn(jnp.ones((5, 3)))
        >>> (output.shape, hidden.shape)
        ((5, 4), (1, 4))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        *,
        nonlinearity: Nonlinearity = 'tanh',
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_recurrent_initializer,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer = default_recurrent_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        unroll: int | bool = 1,
    ) -> None:
        axis_names, partition_spec = _validate_projection_axes(
            axis_names,
            partition_spec,
        )
        hidden_axis = None if axis_names is None else axis_names[1]

        def cell_factory(
            layer_input_size: int,
            layer_input_axis: str | None,
        ) -> Module:
            cell_axes = (
                None
                if axis_names is None
                else (layer_input_axis, hidden_axis)
            )
            return RNNCell(
                layer_input_size,
                hidden_size,
                nonlinearity=nonlinearity,
                bias=bias,
                dtype=dtype,
                rngs=rngs,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
                quant=quant,
                dot_general=dot_general,
                axis_names=cell_axes,
                partition_spec=partition_spec,
                kernel_metadata=kernel_metadata,
                bias_metadata=bias_metadata,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )

        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            dtype=dtype,
            axis_names=axis_names,
            unroll=unroll,
            output_size=hidden_size,
            state_sizes=(hidden_size,),
            cell_factory=cell_factory,
        )


class LSTM(RecurrentBase):
    r"""Apply a stacked long short-term memory network to a sequence.

    Each cell computes input, forget, candidate, and output gates from the
    current input and previous hidden state, then updates

    .. math::

        \begin{aligned}
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
        \widetilde{h}_t &= o_t \odot \tanh(c_t)
        \end{aligned}

    Without projection, :math:`h_t = \widetilde{h}_t`. With ``proj_size > 0``,
    :math:`h_t = \widetilde{h}_t W_{hr}`.

    The returned state is ``(hidden, cell)``. With ``proj_size > 0``, hidden
    outputs have width ``proj_size`` while cell states retain ``hidden_size``.
    Bidirectional outputs concatenate both directions on the feature axis.

    Args:
        input_size: Number of features in each input time step.
        hidden_size: Number of features in each cell state.
        num_layers: Number of stacked recurrent layers. Defaults to ``1``.
        bias: Whether the input and recurrent gate projections include
            biases. Defaults to ``True``.
        batch_first: Whether batched inputs use batch-major layout. Defaults
            to ``False``.
        dropout: Dropout probability between layers. Dropout is active only
            during training and is not applied after the last layer. Defaults
            to ``0.0``.
        bidirectional: Whether each layer scans in both directions. Defaults
            to ``False``.
        proj_size: Per-direction hidden output width. ``0`` disables the
            projection; otherwise it must be smaller than ``hidden_size``.
        dtype: Data type passed to parameter initializers.
        rngs: Random number generator used to initialize all cells.
        kernel_initializer: Input-kernel initializer. Defaults to LeCun
            uniform initialization.
        recurrent_initializer: Recurrent and output-projection initializer.
            Defaults to ``kernel_initializer``.
        bias_initializer: Gate-bias initializer. Defaults to zeros.
        quant: Optional Qwix quantization configuration for projection kernels.
        dot_general: Optional ``dot_general`` implementation used by cells.
        axis_names: Optional logical names for the input and hidden axes.
        partition_spec: Optional partition specifications for the input and
            hidden axes.
        kernel_metadata: Optional metadata attached to kernel parameters.
        bias_metadata: Optional metadata attached to gate-bias parameters.
        precision: Dot-product precision forwarded to cell projections.
        preferred_element_type: Preferred result and accumulation data type
            forwarded to cell projections.
        unroll: Loop-unrolling option passed to :func:`jax.lax.scan`. Defaults
            to ``1``.

    References:
        - Gers, F. A., Schmidhuber, J., and Cummins, F. (2000). `Learning to
          Forget: Continual Prediction with LSTM
          <https://doi.org/10.1162/089976600300015015>`_.
        - Sak, H., Senior, A., and Beaufays, F. (2014). `Long Short-Term
          Memory Based Recurrent Neural Network Architectures for Large
          Vocabulary Speech Recognition <https://arxiv.org/abs/1402.1128>`_.

    Examples:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> lstm = nn.LSTM(3, 5, proj_size=2, rngs=nn.Rngs(0))
        >>> output, (hidden, cell) = lstm(jnp.ones((4, 3)))
        >>> (output.shape, hidden.shape, cell.shape)
        ((4, 2), (1, 2), (1, 5))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        *,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_recurrent_initializer,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer = default_recurrent_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        unroll: int | bool = 1,
    ) -> None:
        proj_size = _validate_integer(proj_size, 'proj_size', minimum=0)
        if proj_size >= hidden_size and proj_size != 0:
            raise ValueError('proj_size must be smaller than hidden_size')
        output_size = proj_size or hidden_size
        axis_names, partition_spec = _validate_projection_axes(
            axis_names,
            partition_spec,
        )
        hidden_axis = None if axis_names is None else axis_names[1]

        def cell_factory(
            layer_input_size: int,
            layer_input_axis: str | None,
        ) -> Module:
            cell_axes = (
                None
                if axis_names is None
                else (layer_input_axis, hidden_axis)
            )
            return LSTMCell(
                layer_input_size,
                hidden_size,
                proj_size=proj_size,
                bias=bias,
                dtype=dtype,
                rngs=rngs,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
                quant=quant,
                dot_general=dot_general,
                axis_names=cell_axes,
                partition_spec=partition_spec,
                kernel_metadata=kernel_metadata,
                bias_metadata=bias_metadata,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )

        self.proj_size = proj_size
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            dtype=dtype,
            axis_names=axis_names,
            unroll=unroll,
            output_size=output_size,
            state_sizes=(output_size, hidden_size),
            cell_factory=cell_factory,
        )

    def extra_repr(self) -> str:
        representation = super().extra_repr()
        if self.proj_size:
            representation += f', proj_size={self.proj_size}'
        return representation


class GRU(RecurrentBase):
    r"""Apply a stacked gated recurrent unit network to a sequence.

    Each cell combines a reset gate :math:`r_t`, update gate :math:`z_t`, and
    candidate activation :math:`n_t`:

    .. math::

        \begin{aligned}
        r_t &= \sigma(x_t W_{ir} + b_{ir}
                       + h_{t-1} W_{hr} + b_{hr}) \\
        z_t &= \sigma(x_t W_{iz} + b_{iz}
                       + h_{t-1} W_{hz} + b_{hz}) \\
        n_t &= \tanh\left(x_t W_{in} + b_{in}
                    + r_t \odot (h_{t-1} W_{hn} + b_{hn})\right) \\
        h_t &= (1-z_t) \odot n_t + z_t \odot h_{t-1}
        \end{aligned}

    This implementation uses the reset-after GRU form. Bidirectional outputs
    concatenate the forward and reverse hidden states on the feature axis.

    Args:
        input_size: Number of features in each input time step.
        hidden_size: Number of features in each directional hidden state.
        num_layers: Number of stacked recurrent layers. Defaults to ``1``.
        bias: Whether the input and recurrent gate projections include
            biases. Defaults to ``True``.
        batch_first: Whether batched inputs use batch-major layout. Defaults
            to ``False``.
        dropout: Dropout probability between layers. Dropout is active only
            during training and is not applied after the last layer. Defaults
            to ``0.0``.
        bidirectional: Whether each layer scans in both directions. Defaults
            to ``False``.
        dtype: Data type passed to parameter initializers.
        rngs: Random number generator used to initialize all cells.
        kernel_initializer: Input-kernel initializer. Defaults to LeCun
            uniform initialization.
        recurrent_initializer: Recurrent-kernel initializer. Defaults to
            ``kernel_initializer``.
        bias_initializer: Gate-bias initializer. Defaults to zeros.
        quant: Optional Qwix quantization configuration for projection kernels.
        dot_general: Optional ``dot_general`` implementation used by cells.
        axis_names: Optional logical names for the input and hidden axes.
        partition_spec: Optional partition specifications for the input and
            hidden axes.
        kernel_metadata: Optional metadata attached to kernel parameters.
        bias_metadata: Optional metadata attached to gate-bias parameters.
        precision: Dot-product precision forwarded to cell projections.
        preferred_element_type: Preferred result and accumulation data type
            forwarded to cell projections.
        unroll: Loop-unrolling option passed to :func:`jax.lax.scan`. Defaults
            to ``1``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from taktiny import nn
        >>> gru = nn.GRU(3, 4, rngs=nn.Rngs(0))
        >>> output, hidden = gru(jnp.ones((5, 3)))
        >>> (output.shape, hidden.shape)
        ((5, 4), (1, 4))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        *,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_recurrent_initializer,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer = default_recurrent_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        unroll: int | bool = 1,
    ) -> None:
        
        axis_names, partition_spec = _validate_projection_axes(
            axis_names,
            partition_spec,
        )
        hidden_axis = None if axis_names is None else axis_names[1]

        def cell_factory(
            layer_input_size: int,
            layer_input_axis: str | None,
        ) -> Module:
            cell_axes = (
                None
                if axis_names is None
                else (layer_input_axis, hidden_axis)
            )
            return GRUCell(
                layer_input_size,
                hidden_size,
                bias=bias,
                dtype=dtype,
                rngs=rngs,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
                quant=quant,
                dot_general=dot_general,
                axis_names=cell_axes,
                partition_spec=partition_spec,
                kernel_metadata=kernel_metadata,
                bias_metadata=bias_metadata,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )

        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            dtype=dtype,
            axis_names=axis_names,
            unroll=unroll,
            output_size=hidden_size,
            state_sizes=(hidden_size,),
            cell_factory=cell_factory,
        )


__all__ = [
    'GRU',
    'LSTM',
    'RNN',
    'GRUCell',
    'LSTMCell',
    'RNNCell',
    'RecurrentBase',
    'RecurrentLayer',
]
