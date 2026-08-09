import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.layers import GRU, LSTM, RNN
from taktiny.utils.typing import ShardMode


def test_rnn_matches_explicit_recurrence():
    model = RNN(
        1,
        1,
        bias=False,
        nonlinearity='tanh',
        rngs=nn.Rngs(0),
    )
    cell = model.layers[0].forward_cell
    cell.input_proj.weight.value = jnp.asarray([[0.7]])
    cell.hidden_proj.weight.value = jnp.asarray([[-0.2]])
    x = jnp.asarray([[0.1], [0.4], [-0.3]])
    h0 = jnp.asarray([[0.25]])

    expected = []
    hidden = h0[0]
    for value in x:
        hidden = jnp.tanh(value * 0.7 + hidden * -0.2)
        expected.append(hidden)
    expected = jnp.stack(expected)

    output, hn = model(x, h0)

    assert jnp.allclose(output, expected)
    assert jnp.allclose(hn[0], expected[-1])


def test_lstm_matches_explicit_gate_equations():
    model = LSTM(1, 1, bias=False, rngs=nn.Rngs(0))
    recurrent_cell = model.layers[0].forward_cell
    recurrent_cell.input_proj.weight.value = jnp.asarray(
        [[[0.2], [0.3], [0.4], [0.5]]]
    )
    recurrent_cell.hidden_proj.weight.value = jnp.asarray(
        [[[0.1], [-0.2], [0.3], [-0.4]]]
    )
    x = jnp.asarray([[0.5], [-0.25], [0.75]])
    h0 = jnp.asarray([[0.15]])
    c0 = jnp.asarray([[-0.1]])

    outputs = []
    hidden = h0[0]
    cell = c0[0]
    input_weights = jnp.asarray([0.2, 0.3, 0.4, 0.5])
    hidden_weights = jnp.asarray([0.1, -0.2, 0.3, -0.4])
    for value in x[:, 0]:
        gates = value * input_weights + hidden[0] * hidden_weights
        input_gate, forget_gate, candidate, output_gate = gates
        input_gate = jax.nn.sigmoid(input_gate)
        forget_gate = jax.nn.sigmoid(forget_gate)
        candidate = jnp.tanh(candidate)
        output_gate = jax.nn.sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = jnp.asarray([output_gate * jnp.tanh(cell[0])])
        outputs.append(hidden)

    output, (hn, cn) = model(x, (h0, c0))

    assert jnp.allclose(output, jnp.stack(outputs))
    assert jnp.allclose(hn[0], hidden)
    assert jnp.allclose(cn[0], cell)


def test_gru_matches_explicit_gate_equations():
    model = GRU(1, 1, bias=False, rngs=nn.Rngs(0))
    cell = model.layers[0].forward_cell
    cell.input_proj.weight.value = jnp.asarray(
        [[[0.2], [0.3], [0.4]]]
    )
    cell.hidden_proj.weight.value = jnp.asarray(
        [[[-0.1], [0.25], [0.5]]]
    )
    x = jnp.asarray([[0.5], [-0.25], [0.75]])
    h0 = jnp.asarray([[0.15]])

    outputs = []
    hidden = h0[0, 0]
    for value in x[:, 0]:
        reset = jax.nn.sigmoid(value * 0.2 + hidden * -0.1)
        update = jax.nn.sigmoid(value * 0.3 + hidden * 0.25)
        candidate = jnp.tanh(value * 0.4 + reset * hidden * 0.5)
        hidden = (1.0 - update) * candidate + update * hidden
        outputs.append(jnp.asarray([hidden]))

    output, hn = model(x, h0)

    assert jnp.allclose(output, jnp.stack(outputs))
    assert jnp.allclose(hn[0, 0], hidden)


@pytest.mark.parametrize('module_type', [RNN, LSTM, GRU])
def test_recurrent_stacked_bidirectional_batch_first_shapes(module_type):
    model = module_type(
        3,
        4,
        2,
        batch_first=True,
        bidirectional=True,
        rngs=nn.Rngs(0),
    )
    x = jnp.ones((2, 5, 3))

    output, state = jax.jit(model)(x)

    assert output.shape == (2, 5, 8)
    if module_type is LSTM:
        assert state[0].shape == (4, 2, 4)
        assert state[1].shape == (4, 2, 4)
    else:
        assert state.shape == (4, 2, 4)


def test_bidirectional_outputs_align_with_original_sequence_order():
    model = RNN(
        1,
        1,
        nonlinearity='relu',
        bias=False,
        bidirectional=True,
        rngs=nn.Rngs(0),
    )
    forward = model.layers[0].forward_cell
    reverse = model.layers[0].reverse_cell
    for cell in (forward, reverse):
        cell.input_proj.weight.value = jnp.ones((1, 1))
        cell.hidden_proj.weight.value = jnp.ones((1, 1))
    x = jnp.asarray([[1.0], [2.0], [3.0]])

    output, hn = model(x)

    assert jnp.array_equal(output[:, 0], jnp.asarray([1.0, 3.0, 6.0]))
    assert jnp.array_equal(output[:, 1], jnp.asarray([6.0, 5.0, 3.0]))
    assert jnp.array_equal(hn[:, 0], jnp.asarray([6.0, 6.0]))


def test_lstm_projection_uses_distinct_hidden_and_cell_widths():
    model = LSTM(3, 5, proj_size=2, rngs=nn.Rngs(0))

    output, (hidden, cell) = model(jnp.ones((4, 3)))

    assert output.shape == (4, 2)
    assert hidden.shape == (1, 2)
    assert cell.shape == (1, 5)


def test_recurrent_dropout_requires_and_reuses_explicit_key():
    model = GRU(2, 3, 2, dropout=0.5, rngs=nn.Rngs(0))
    x = jnp.ones((6, 2))

    with pytest.raises(ValueError, match='key is required'):
        model(x, training=True)

    key = jax.random.key(9)
    first, _ = model(x, training=True, key=key)
    second, _ = model(x, training=True, key=key)

    assert jnp.array_equal(first, second)


def test_recurrent_parameter_axis_names_follow_layer_inputs():
    model = LSTM(
        3,
        4,
        2,
        axis_names=('features', 'hidden'),
        rngs=nn.Rngs(0),
    )
    first = model.layers[0].forward_cell
    second = model.layers[1].forward_cell

    assert first.input_proj.weight.axis_names == (
        'features',
        None,
        'hidden',
    )
    assert first.input_proj.bias.axis_names == (None, 'hidden')
    assert second.input_proj.weight.axis_names == (
        'hidden',
        None,
        'hidden',
    )


def test_recurrent_explicit_sharding_covers_final_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    model = RNN(
        2,
        3,
        shard_mode=ShardMode.EXPLICIT,
        rngs=nn.Rngs(0),
    )
    apply = lambda value: model(value, out_sharding=out_sharding)[0]
    x = jnp.ones((4, 2))

    jaxpr = jax.make_jaxpr(apply)(x).jaxpr
    output = jax.jit(apply)(x)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)


def test_recurrent_validates_input_and_state_shapes():
    model = RNN(2, 3, rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match='expected input_size=2'):
        model(jnp.ones((4, 5)))
    with pytest.raises(ValueError, match='initial state 0'):
        model(jnp.ones((4, 2)), jnp.ones((2, 3)))
    with pytest.raises(TypeError, match='floating-point'):
        model(jnp.ones((4, 2), dtype=jnp.int32))


def test_recurrent_empty_sequence_preserves_initial_state():
    model = GRU(2, 3, rngs=nn.Rngs(0))
    initial = jnp.arange(3, dtype=jnp.float32)[None, :]

    output, hidden = model(jnp.empty((0, 2)), initial)

    assert output.shape == (0, 3)
    assert jnp.array_equal(hidden, initial)
