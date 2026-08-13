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
"""Recurrent neural-network layers."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import Any
import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_uniform

from taktiny import nn
from taktiny.nn._continuo import _constrain, _validate_integer
from taktiny.utils.typing import AxisName, AxisNames, DType, Initializer, ShardMode


default_recurrent_initializer = lecun_uniform()
def _projection_axis_names(
    input_axis: AxisName,
    hidden_axis: AxisName,
    gates: int,
) -> AxisNames:
    if gates == 1:
        return (input_axis, hidden_axis)
    return (input_axis, None, hidden_axis)

class _RNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        nonlinearity: str | Callable[[jax.Array], jax.Array],
        bias: bool,
        dtype: DType,
        rngs: nn.Rngs,
        initializer: Initializer,
        input_axis: AxisName,
        hidden_axis: AxisName,
        shard_mode: ShardMode,
        quant: Any,
        dot_general: Any,
    ) -> None:
        if isinstance(nonlinearity, str):
            name = nonlinearity.lower()
            functions = {'tanh': jnp.tanh, 'relu': jax.nn.relu}
            if name not in functions:
                raise ValueError("nonlinearity must be 'tanh', 'relu', or callable")
            self.nonlinearity = functions[name]
            self.nonlinearity_name = name
        elif callable(nonlinearity):
            self.nonlinearity = nonlinearity
            self.nonlinearity_name = getattr(
                nonlinearity,
                '__name__',
                type(nonlinearity).__name__,
            )
        else:
            raise TypeError('nonlinearity must be a string or callable')

        self.input_proj = nn.Linear(
            input_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            axis_names=_projection_axis_names(input_axis, hidden_axis, 1),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.hidden_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            axis_names=_projection_axis_names(hidden_axis, hidden_axis, 1),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )

    def __call__(
        self,
        state: tuple[jax.Array],
        x: jax.Array,
    ) -> tuple[tuple[jax.Array], jax.Array]:
        hidden = self.nonlinearity(
            self.input_proj(x) + self.hidden_proj(state[0])
        )
        return (hidden,), hidden

class _LSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        bias: bool,
        dtype: DType,
        rngs: nn.Rngs,
        initializer: Initializer,
        input_axis: AxisName,
        hidden_axis: AxisName,
        shard_mode: ShardMode,
        quant: Any,
        dot_general: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_proj = nn.Linear(
            input_size,
            (4, hidden_size),
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            axis_names=_projection_axis_names(input_axis, hidden_axis, 4),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.hidden_proj = nn.Linear(
            output_size,
            (4, hidden_size),
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            axis_names=_projection_axis_names(hidden_axis, hidden_axis, 4),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        if output_size != hidden_size:
            self.projection = nn.Linear(
                hidden_size,
                output_size,
                bias=False,
                dtype=dtype,
                rngs=rngs,
                initializer=initializer,
                axis_names=(hidden_axis, hidden_axis),
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
            )

    def __call__(
        self,
        state: tuple[jax.Array, jax.Array],
        x: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        hidden, cell = state
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
        if self.output_size != self.hidden_size:
            hidden = self.projection(hidden)
        return (hidden, cell), hidden

class _GRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool,
        dtype: DType,
        rngs: nn.Rngs,
        initializer: Initializer,
        input_axis: AxisName,
        hidden_axis: AxisName,
        shard_mode: ShardMode,
        quant: Any,
        dot_general: Any,
    ) -> None:
        self.input_proj = nn.Linear(
            input_size,
            (3, hidden_size),
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            axis_names=_projection_axis_names(input_axis, hidden_axis, 3),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.hidden_proj = nn.Linear(
            hidden_size,
            (3, hidden_size),
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            axis_names=_projection_axis_names(hidden_axis, hidden_axis, 3),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )

    def __call__(
        self,
        state: tuple[jax.Array],
        x: jax.Array,
    ) -> tuple[tuple[jax.Array], jax.Array]:
        hidden = state[0]
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
        return (hidden,), hidden

class _RecurrentLayer(nn.Module):
    def __init__(
        self,
        forward_cell: nn.Module,
        reverse_cell: nn.Module | None = None,
    ) -> None:
        self.forward_cell = forward_cell
        if reverse_cell is not None:
            self.reverse_cell = reverse_cell

class _RecurrentBase(nn.Module):
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
        shard_mode: ShardMode,
        unroll: int | bool,
        output_size: int,
        state_sizes: tuple[int, ...],
        cell_factory: Callable[[int, AxisName, AxisName], nn.Module],
    ) -> None:
        _validate_integer(input_size, 'input_size')
        _validate_integer(hidden_size, 'hidden_size')
        _validate_integer(num_layers, 'num_layers')
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError('dropout must be a number')
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError('dropout must be between 0 and 1')
        if not isinstance(unroll, (int, bool)) or (
            isinstance(unroll, int)
            and not isinstance(unroll, bool)
            and unroll <= 0
        ):
            raise ValueError('unroll must be a positive integer or boolean')
        if axis_names is not None:
            axis_names = tuple(axis_names)
            if len(axis_names) != 2:
                raise ValueError(
                    'axis_names must contain input and hidden logical axes'
                )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.has_bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dtype = jnp.float32 if dtype is None else dtype
        self.shard_mode = shard_mode
        self.unroll = unroll
        self.output_size = output_size
        self.state_sizes = state_sizes

        input_axis, hidden_axis = (
            (None, None) if axis_names is None else axis_names
        )
        layers = []
        layer_input_size = input_size
        for layer_index in range(num_layers):
            layer_input_axis = input_axis if layer_index == 0 else hidden_axis
            forward_cell = cell_factory(
                layer_input_size,
                layer_input_axis,
                hidden_axis,
            )
            reverse_cell = None
            if bidirectional:
                reverse_cell = cell_factory(
                    layer_input_size,
                    layer_input_axis,
                    hidden_axis,
                )
            layers.append(_RecurrentLayer(forward_cell, reverse_cell))
            layer_input_size = self.num_directions * output_size
        self.layers = nn.List(layers)

    def _prepare_input(
        self,
        x: jax.Array,
    ) -> tuple[jax.Array, bool, int]:
        if x.ndim not in {2, 3}:
            raise ValueError(
                'input must have shape [sequence, input] or contain one batch '
                'dimension'
            )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f'expected input_size={self.input_size}, got {x.shape[-1]}'
            )
        if not jnp.issubdtype(x.dtype, jnp.inexact):
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
    ) -> tuple[jax.Array, ...]:
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
            normalized.append(value[:, None, :] if unbatched else value)
        return tuple(normalized)

    def _apply_dropout(self, x: jax.Array, key: jax.Array) -> jax.Array:
        if self.dropout == 1.0:
            return jnp.zeros_like(x)
        keep_probability = 1.0 - self.dropout
        mask = jax.random.bernoulli(key, keep_probability, x.shape)
        return jnp.where(mask, x / keep_probability, 0)

    def __call__(
        self,
        x: jax.Array,
        hx: jax.Array | Sequence[jax.Array] | None = None,
        *,
        training: bool = False,
        key: jax.Array | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, Any]:
        x, unbatched, batch_size = self._prepare_input(x)
        states = self._prepare_states(
            hx,
            batch_size=batch_size,
            unbatched=unbatched,
            dtype=x.dtype,
        )

        use_dropout = training and self.dropout > 0.0 and self.num_layers > 1
        if use_dropout:
            if key is None:
                raise ValueError(
                    'key is required when recurrent dropout is enabled during training'
                )
            dropout_keys = jax.random.split(key, self.num_layers - 1)
        else:
            dropout_keys = ()

        final_states: list[list[jax.Array]] = [
            [] for _ in self.state_sizes
        ]
        layer_input = x
        for layer_index, layer in enumerate(self.layers):
            direction_outputs = []
            cells = [layer.forward_cell]
            if self.bidirectional:
                cells.append(layer.reverse_cell)

            for direction, cell in enumerate(cells):
                state_index = layer_index * self.num_directions + direction
                initial_state = tuple(
                    state[state_index] for state in states
                )

                def step(
                    carry: tuple[jax.Array, ...],
                    value: jax.Array,
                ) -> tuple[tuple[jax.Array, ...], jax.Array]:
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

        output = _constrain(output, out_sharding, self.shard_mode)
        return output, hidden[0] if len(hidden) == 1 else hidden

    def extra_repr(self) -> str:
        options = [f'{self.input_size} -> {self.hidden_size}']
        if self.num_layers != 1:
            options.append(f'layers={self.num_layers}')
        if self.bidirectional:
            options.append('bidirectional=True')
        if self.dropout:
            options.append(f'dropout={self.dropout}')
        return ', '.join(options)

class RNN(_RecurrentBase):
    """Apply an Elman RNN over a sequence using :func:`jax.lax.scan`."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        *,
        nonlinearity: str | Callable[[jax.Array], jax.Array] = 'tanh',
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        dtype: DType | None = None,
        rngs: nn.Rngs,
        initializer: Initializer = default_recurrent_initializer,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
        unroll: int | bool = 1,
    ) -> None:
        parameter_dtype = jnp.float32 if dtype is None else dtype

        def cell_factory(
            layer_input_size: int,
            input_axis: str | None,
            hidden_axis: str | None,
        ) -> nn.Module:
            return _RNNCell(
                layer_input_size,
                hidden_size,
                nonlinearity=nonlinearity,
                bias=bias,
                dtype=parameter_dtype,
                rngs=rngs,
                initializer=initializer,
                input_axis=input_axis,
                hidden_axis=hidden_axis,
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
            )

        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            dtype=parameter_dtype,
            axis_names=axis_names,
            shard_mode=shard_mode,
            unroll=unroll,
            output_size=hidden_size,
            state_sizes=(hidden_size,),
            cell_factory=cell_factory,
        )

class LSTM(_RecurrentBase):
    """Apply a long short-term memory network over a sequence."""

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
        rngs: nn.Rngs,
        initializer: Initializer = default_recurrent_initializer,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
        unroll: int | bool = 1,
    ) -> None:
        _validate_integer(proj_size, 'proj_size', minimum=0)
        if proj_size >= hidden_size and proj_size != 0:
            raise ValueError('proj_size must be smaller than hidden_size')
        output_size = proj_size or hidden_size
        parameter_dtype = jnp.float32 if dtype is None else dtype

        def cell_factory(
            layer_input_size: int,
            input_axis: str | None,
            hidden_axis: str | None,
        ) -> nn.Module:
            return _LSTMCell(
                layer_input_size,
                hidden_size,
                output_size,
                bias=bias,
                dtype=parameter_dtype,
                rngs=rngs,
                initializer=initializer,
                input_axis=input_axis,
                hidden_axis=hidden_axis,
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
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
            dtype=parameter_dtype,
            axis_names=axis_names,
            shard_mode=shard_mode,
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

class GRU(_RecurrentBase):
    """Apply a gated recurrent unit network over a sequence."""

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
        rngs: nn.Rngs,
        initializer: Initializer = default_recurrent_initializer,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
        unroll: int | bool = 1,
    ) -> None:
        parameter_dtype = jnp.float32 if dtype is None else dtype

        def cell_factory(
            layer_input_size: int,
            input_axis: str | None,
            hidden_axis: str | None,
        ) -> nn.Module:
            return _GRUCell(
                layer_input_size,
                hidden_size,
                bias=bias,
                dtype=parameter_dtype,
                rngs=rngs,
                initializer=initializer,
                input_axis=input_axis,
                hidden_axis=hidden_axis,
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
            )

        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            dtype=parameter_dtype,
            axis_names=axis_names,
            shard_mode=shard_mode,
            unroll=unroll,
            output_size=hidden_size,
            state_sizes=(hidden_size,),
            cell_factory=cell_factory,
        )


__all__ = ['RNN', 'LSTM', 'GRU']
