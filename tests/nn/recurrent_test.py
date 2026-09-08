import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from taktiny import nn
from taktiny.utils.spmd import map_logical_axis_names


def test_rnn_matches_explicit_recurrence():
    model = nn.RNN(
        1,
        1,
        bias=False,
        nonlinearity='tanh',
        rngs=nn.Rngs(0),
    )
    cell = model.layers[0].forward_cell
    cell.input_proj.load_state_dict({'kernel': jnp.asarray([[0.7]])})
    cell.hidden_proj.load_state_dict({'kernel': jnp.asarray([[-0.2]])})
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
    model = nn.LSTM(1, 1, bias=False, rngs=nn.Rngs(0))
    cell_module = model.layers[0].forward_cell
    cell_module.input_proj.load_state_dict({
        'kernel': jnp.asarray([[[0.2], [0.3], [0.4], [0.5]]]),
    })
    cell_module.hidden_proj.load_state_dict({
        'kernel': jnp.asarray([[[0.1], [-0.2], [0.3], [-0.4]]]),
    })
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
    model = nn.GRU(1, 1, bias=False, rngs=nn.Rngs(0))
    cell = model.layers[0].forward_cell
    cell.input_proj.load_state_dict({
        'kernel': jnp.asarray([[[0.2], [0.3], [0.4]]]),
    })
    cell.hidden_proj.load_state_dict({
        'kernel': jnp.asarray([[[-0.1], [0.25], [0.5]]]),
    })
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


@pytest.mark.parametrize(
    ('cell_type', 'state_shapes'),
    [
        (nn.RNNCell, ((2, 4),)),
        (nn.GRUCell, ((2, 4),)),
        (nn.LSTMCell, ((2, 4), (2, 4))),
    ],
)
def test_cells_use_scan_compatible_state_input_order(cell_type, state_shapes):
    cell = cell_type(3, 4, rngs=nn.Rngs(0))
    state = cell.initial_state((2,))

    next_state, output = jax.jit(cell)(state, jnp.ones((2, 3)))

    assert tuple(component.shape for component in state) == state_shapes
    assert tuple(component.shape for component in next_state) == state_shapes
    assert output.shape == (2, 4)


def test_rnn_supports_callable_activation_and_separate_initializers():
    cell = nn.RNNCell(
        3,
        2,
        nonlinearity=lambda value: value,
        bias=False,
        kernel_initializer=jax.nn.initializers.ones,
        recurrent_initializer=jax.nn.initializers.zeros,
        rngs=nn.Rngs(0),
    )

    _, output = cell(cell.initial_state((1,)), jnp.ones((1, 3)))

    assert jnp.array_equal(output, jnp.full((1, 2), 3.0))
    assert cell.nonlinearity_name == '<lambda>'


@pytest.mark.parametrize('module_type', [nn.RNN, nn.LSTM, nn.GRU])
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
    if module_type is nn.LSTM:
        assert state[0].shape == (4, 2, 4)
        assert state[1].shape == (4, 2, 4)
    else:
        assert state.shape == (4, 2, 4)


def test_bidirectional_outputs_align_with_original_sequence_order():
    model = nn.RNN(
        1,
        1,
        nonlinearity='relu',
        bias=False,
        bidirectional=True,
        rngs=nn.Rngs(0),
    )
    for cell in (
        model.layers[0].forward_cell,
        model.layers[0].reverse_cell,
    ):
        cell.input_proj.load_state_dict({'kernel': jnp.ones((1, 1))})
        cell.hidden_proj.load_state_dict({'kernel': jnp.ones((1, 1))})
    x = jnp.asarray([[1.0], [2.0], [3.0]])

    output, hn = model(x)

    assert jnp.array_equal(output[:, 0], jnp.asarray([1.0, 3.0, 6.0]))
    assert jnp.array_equal(output[:, 1], jnp.asarray([6.0, 5.0, 3.0]))
    assert jnp.array_equal(hn[:, 0], jnp.asarray([6.0, 6.0]))


def test_lstm_projection_uses_distinct_hidden_and_cell_widths():
    model = nn.LSTM(3, 5, proj_size=2, rngs=nn.Rngs(0))

    output, (hidden, cell) = model(jnp.ones((4, 3)))

    assert output.shape == (4, 2)
    assert hidden.shape == (1, 2)
    assert cell.shape == (1, 5)


def test_recurrent_dropout_uses_module_mode_and_explicit_rngs():
    model = nn.GRU(2, 3, 2, dropout=0.5, rngs=nn.Rngs(0))
    x = jnp.ones((6, 2))

    with pytest.raises(ValueError, match='rngs is required'):
        model(x)

    first, _ = model(x, rngs=nn.Rngs(9))
    second, _ = model(x, rngs=nn.Rngs(9))
    assert jnp.array_equal(first, second)

    model.eval()
    evaluation, _ = model(x)
    assert evaluation.shape == (6, 3)


def test_recurrent_parameter_configuration_reaches_every_projection():
    model = nn.LSTM(
        3,
        4,
        2,
        axis_names=('features', 'hidden'),
        kernel_metadata={'kind': 'recurrent'},
        bias_metadata={'kind': 'offset'},
        precision=jax.lax.Precision.HIGHEST,
        rngs=nn.Rngs(0),
    )
    first = model.layers[0].forward_cell
    second = model.layers[1].forward_cell

    assert first.input_proj.kernel.axis_names == (
        'features',
        None,
        'hidden',
    )
    assert first.hidden_proj.kernel.axis_names == (None, None, 'hidden')
    assert first.input_proj.bias.axis_names == (None, 'hidden')
    assert second.input_proj.kernel.axis_names == (None, None, 'hidden')
    assert first.input_proj.kernel.metadata == {'kind': 'recurrent'}
    assert first.hidden_proj.kernel.metadata == {'kind': 'recurrent'}
    assert first.input_proj.bias.metadata == {'kind': 'offset'}
    assert first.input_proj.precision == jax.lax.Precision.HIGHEST


def test_recurrent_expands_semantic_partition_spec_for_gate_kernels():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('model',))
    semantic_spec = P(None, 'model')
    gate_spec = P(None, None, 'model')
    bias_spec = P(None, 'model')

    with jax.set_mesh(mesh):
        model = nn.GRU(
            3,
            4,
            partition_spec=semantic_spec,
            rngs=nn.Rngs(0),
        )

    cell = model.layers[0].forward_cell
    assert cell.input_proj.kernel.partition_spec == gate_spec
    assert cell.hidden_proj.kernel.partition_spec == gate_spec
    assert cell.input_proj.bias.partition_spec == bias_spec
    assert cell.input_proj.kernel.value.sharding.is_equivalent_to(
        NamedSharding(mesh, gate_spec),
        cell.input_proj.kernel.ndim,
    )


def test_recurrent_logical_axes_override_explicit_partitioning():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ('data', 'model'))
    explicit_spec = P('data', None)
    input_spec = P('data', None, 'model')
    recurrent_spec = P(None, None, 'model')

    with jax.set_mesh(mesh), map_logical_axis_names({
        'features': 'data',
        'hidden': 'model',
    }):
        model = nn.GRU(
            3,
            4,
            axis_names=('features', 'hidden'),
            partition_spec=explicit_spec,
            rngs=nn.Rngs(0),
        )

    cell = model.layers[0].forward_cell
    assert cell.input_proj.kernel.partition_spec == input_spec
    assert cell.hidden_proj.kernel.partition_spec == recurrent_spec
    assert cell.input_proj.kernel.value.sharding.spec == input_spec
    assert cell.hidden_proj.kernel.value.sharding.spec == recurrent_spec


def test_recurrent_supports_quantized_projection_kernels():
    model = nn.RNN(
        4,
        3,
        bias=False,
        quant='int8',
        rngs=nn.Rngs(0),
    )
    cell = model.layers[0].forward_cell

    output, hidden = jax.jit(model)(jnp.ones((2, 4)))

    assert isinstance(cell.input_proj.kernel.value, qwix.QArray)
    assert isinstance(cell.hidden_proj.kernel.value, qwix.QArray)
    assert output.shape == (2, 3)
    assert hidden.shape == (1, 3)
    assert jnp.all(jnp.isfinite(output))


def test_recurrent_custom_dot_general_is_used_by_both_projections():
    calls = []

    def custom_dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type,
        *,
        out_sharding=None,
    ):
        calls.append((precision, preferred_element_type, out_sharding))
        return jax.lax.dot_general(
            lhs,
            rhs,
            dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )

    cell = nn.RNNCell(
        2,
        3,
        dot_general=custom_dot_general,
        precision=jax.lax.Precision.HIGHEST,
        rngs=nn.Rngs(0),
    )

    cell(cell.initial_state((1,)), jnp.ones((1, 2)))

    assert calls == [
        (jax.lax.Precision.HIGHEST, None, None),
        (jax.lax.Precision.HIGHEST, None, None),
    ]


def test_recurrent_explicit_sharding_covers_final_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    model = nn.RNN(2, 3, rngs=nn.Rngs(0))
    apply = lambda value: model(value, out_sharding=out_sharding)[0]
    x = jnp.ones((4, 2))

    jaxpr = jax.make_jaxpr(apply)(x).jaxpr
    output = jax.jit(apply)(x)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)


def test_recurrent_validates_input_state_and_configuration():
    model = nn.RNN(2, 3, rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match='expected input_size=2'):
        model(jnp.ones((4, 5)))
    with pytest.raises(ValueError, match='initial state 0'):
        model(jnp.ones((4, 2)), jnp.ones((2, 3)))
    with pytest.raises(TypeError, match='floating-point'):
        model(jnp.ones((4, 2), dtype=jnp.int32))
    with pytest.raises(ValueError, match='axis_names'):
        nn.RNNCell(2, 3, axis_names=('only',), rngs=nn.Rngs(0))
    with pytest.raises(TypeError, match='partition_spec'):
        nn.RNNCell(2, 3, axis_names=P(None, None), rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='partition_spec'):
        nn.GRU(2, 3, partition_spec=P(None), rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='proj_size'):
        nn.LSTMCell(2, 3, proj_size=3, rngs=nn.Rngs(0))


def test_recurrent_empty_sequence_preserves_initial_state():
    model = nn.GRU(2, 3, rngs=nn.Rngs(0))
    initial = jnp.arange(3, dtype=jnp.float32)[None, :]

    output, hidden = model(jnp.empty((0, 2)), initial)

    assert output.shape == (0, 3)
    assert jnp.array_equal(hidden, initial)
