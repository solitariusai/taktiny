import jax
import jax.numpy as jnp
import pytest
import qwix
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.utils.spmd import map_logical_axis_names


class Add(nn.Module):
    def __init__(self, value):
        self.value = nn.Parameter(jnp.asarray(value))

    def __call__(self, x):
        return x + self.value


class AxisAdd(Add):
    def __call__(self, x):
        return jax.lax.psum(super().__call__(x), 'layers')


class ConfiguredAdd(Add):
    def __init__(self, value, mode):
        super().__init__(value)
        self.mode = mode


class Double(nn.Module):
    def __call__(self, x):
        return x * 2


def test_list_accepts_one_module_sequence():
    modules = nn.List([Add(1), Add(2)])

    assert len(modules) == 2
    assert [float(layer.value.value) for layer in modules] == [1.0, 2.0]
    assert modules[0] is modules.layers[0]


def test_list_is_a_jax_pytree():
    modules = nn.List((Add(1), Add(2)))

    leaves = jax.tree.leaves(modules)

    assert len(leaves) == 2
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_list_slice_preserves_container_and_parameter_paths():
    class Model(nn.Module):
        def __init__(self):
            self.layers = nn.List([Add(1), Add(2), Add(3)])

    model = Model()
    model.layers = model.layers[:2]

    assert isinstance(model.layers, nn.List)
    assert len(model.layers) == 2
    assert set(model.flat_parameter_dict()) == {
        'layers.0.value',
        'layers.1.value',
    }


def test_list_rejects_non_sequence_and_non_module_elements():
    with pytest.raises(TypeError, match='must be a sequence'):
        nn.List(Add(index) for index in range(2))
    with pytest.raises(TypeError, match=r'modules\[1\] must be a Module'):
        nn.List([Add(1), object()])


def test_sequential_accepts_one_module_sequence():
    modules = nn.Sequential([Add(1), Add(2)])

    output = jax.jit(modules)(jnp.asarray(3))

    assert output == 6
    assert len(modules) == 2
    assert list(modules) == list(modules.layers)
    assert modules[0] is modules.layers[0]


def test_sequential_slice_preserves_container_behavior():
    modules = nn.Sequential([Add(1), Add(2), Add(3)])[:2]

    assert isinstance(modules, nn.Sequential)
    assert len(modules) == 2
    assert jax.jit(modules)(jnp.asarray(3)) == 6


def test_sequential_rejects_non_sequence_and_non_module_elements():
    with pytest.raises(TypeError, match='must be a sequence'):
        nn.Sequential(Add(index) for index in range(2))
    with pytest.raises(TypeError, match=r'modules\[1\] must be a Module'):
        nn.Sequential([Add(1), object()])


def test_dict_exposes_mapping_operations_and_parameters():
    modules = nn.Dict({'left': Add(1), 'right': Add(2)})

    assert len(modules) == 2
    assert list(modules) == ['left', 'right']
    assert list(modules.keys()) == ['left', 'right']
    assert list(modules.values()) == list(modules.layers.values())
    assert list(modules.items()) == list(modules.layers.items())
    assert 'left' in modules
    assert modules['right'] is modules.layers['right']
    assert set(modules.flat_parameter_dict()) == {'left.value', 'right.value'}
    assert len(jax.tree.leaves(modules)) == 2


def test_train_and_eval_recursively_update_child_modules():
    model = nn.Dict(
        {
            'sequential': nn.Sequential([Add(1), Add(2)]),
            'list': nn.List([Double(), Double()]),
        }
    )

    assert model.training
    assert all(module.training for module in model.values())
    assert all(
        child.training
        for module in model.values()
        for child in module
    )

    assert model.eval() is model
    assert not model.training
    assert all(not module.training for module in model.values())
    assert all(
        not child.training
        for module in model.values()
        for child in module
    )

    assert model.train() is model
    assert model.training
    assert all(module.training for module in model.values())
@pytest.mark.parametrize(
    ('modules', 'message'),
    [
        ([Add(1)], 'must be a mapping'),
        ({1: Add(1)}, 'keys must be strings'),
        ({'': Add(1)}, 'must not be empty'),
        ({'nested.key': Add(1)}, "must not contain '.'"),
        ({'invalid': object()}, r"modules\['invalid'\] must be a Module"),
    ],
)
def test_dict_validates_keys_and_modules(modules, message):
    with pytest.raises((TypeError, ValueError), match=message):
        nn.Dict(modules)


def test_seq_stack_exposes_scan_controls_and_length():
    modules = nn.SeqStack(
        [Add(1), Add(2), Add(3)],
        reverse=True,
        unroll=2,
        split_transpose=True,
    )

    def apply(layer, carry):
        output = layer(carry)
        return output, output

    final, outputs = modules(apply, jnp.asarray(0))

    assert len(modules) == 3
    assert final == 6
    assert jnp.array_equal(outputs, jnp.asarray([6, 5, 3]))


def test_seq_stack_slices_quantized_parameters_loaded_after_stacking():
    layers = nn.SeqStack([
        nn.Linear(4, 4, bias=False, rngs=nn.Rngs(index))
        for index in range(2)
    ])
    parameter = layers.stacked.kernel
    parameter._value = qwix.quantize(
        parameter.value,
        'int4',
        channelwise_axes=(0, 2),
    )

    def apply(layer, carry):
        output = layer(carry)
        return output, None

    inputs = jnp.ones((1, 4), dtype=jnp.float32)
    output, _ = layers(apply, inputs)

    expected = inputs
    for weight in qwix.dequantize(parameter.value):
        expected = expected @ weight

    assert output.shape == (1, 4)
    assert jnp.allclose(output, expected)


def test_stack_exposes_vmap_axis_controls_and_broadcast_axes():
    modules = nn.Stack(
        [AxisAdd(1), AxisAdd(2), AxisAdd(3)],
        axis_name='layers',
    )

    output = modules(jnp.asarray(10), in_axes=None)

    assert len(modules) == 3
    assert jnp.array_equal(output, jnp.asarray([36, 36, 36]))


def test_seq_stack_groups_contiguous_static_configurations():
    modules = nn.SeqStack(
        [
            ConfiguredAdd(1, 'a'),
            ConfiguredAdd(2, 'a'),
            ConfiguredAdd(3, 'b'),
            ConfiguredAdd(4, 'b'),
            ConfiguredAdd(5, 'a'),
            ConfiguredAdd(6, 'a'),
        ]
    )

    def apply(layer, carry):
        output = layer(carry)
        return output, output

    final, outputs = modules(apply, jnp.asarray(0))

    assert modules.group_sizes == (2, 2, 2)
    assert final == 21
    assert jnp.array_equal(outputs, jnp.asarray([1, 3, 6, 10, 15, 21]))


def test_parallel_stack_reports_leaf_shape_mismatches():
    with pytest.raises(ValueError, match='same shape'):
        nn.Stack([Add(jnp.ones(2)), Add(jnp.ones(3))])


def test_stacks_derive_size_for_parameter_free_modules():
    modules = [Double(), Double(), Double()]
    parallel = nn.Stack(modules)
    sequential = nn.SeqStack(modules)

    parallel_output = parallel(jnp.asarray(2), in_axes=None)

    def apply(layer, carry):
        output = layer(carry)
        return output, output

    final, sequential_outputs = sequential(apply, jnp.asarray(1))

    assert jnp.array_equal(parallel_output, jnp.asarray([4, 4, 4]))
    assert final == 8
    assert jnp.array_equal(sequential_outputs, jnp.asarray([2, 4, 8]))


class Counter(nn.Module):
    def __init__(self, mode='a'):
        self.count = nn.Parameter(jnp.array(0.), trainable=False)
        self.mode = mode

    def __call__(self, x):
        self.count._value = self.count.value + 1
        return x + self.count.value


def _scan_apply(layer, carry):
    value = layer(carry)
    return value, value


@pytest.mark.parametrize('cls', [nn.Stack, nn.SeqStack])
def test_stacks_preserve_dynamic_updates_and_leave_originals_unchanged(cls):
    originals = [Counter(), Counter()]
    stack = cls(originals)
    if cls is nn.Stack:
        first = stack(jnp.array(0.), in_axes=None)
        second = stack(jnp.array(0.), in_axes=None)
        assert jnp.array_equal(first, jnp.array([1., 1.]))
        assert jnp.array_equal(second, jnp.array([2., 2.]))
    else:
        first, _ = stack(_scan_apply, jnp.array(0.))
        second, _ = stack(_scan_apply, jnp.array(0.))
        assert first == 2 and second == 4
    assert jnp.array_equal(stack.stacked.count.value, jnp.array([2., 2.]))
    assert all(layer.count.value == 0 for layer in originals)


@pytest.mark.parametrize('reverse', [False, True])
def test_grouped_seq_stack_preserves_state_and_original_output_order(reverse):
    stack = nn.SeqStack([Counter('a'), Counter('a'), Counter('b')], reverse=reverse)
    for call in (1, 2):
        carry, outputs = stack(_scan_apply, jnp.array(0.))
        expected = jnp.arange(1, 4, dtype=jnp.float32) * call
        assert carry == 3 * call
        assert jnp.array_equal(outputs, expected[::-1] if reverse else expected)
        assert all(jnp.all(group.stacked.count.value == call) for group in stack.groups)


@pytest.mark.parametrize('cls', [nn.Stack, nn.SeqStack])
def test_stacks_thread_state_through_jit_without_leaking(cls):
    @jax.jit
    def step(model, x):
        output = (model(x, in_axes=None) if cls is nn.Stack
                  else model(_scan_apply, x)[0])
        return output, model

    model = cls([Counter(), Counter()])
    with jax.checking_leaks():
        _, model = step(model, jnp.array(0.))
        _, model = step(model, jnp.array(0.))
    assert jnp.all(model.stacked.count.value == 2)


@pytest.mark.parametrize('cls', [nn.Stack, nn.SeqStack])
def test_stateless_stacks_can_be_closed_over_by_jit(cls):
    model = cls([Add(1.), Add(2.)])
    stored = model.stacked
    with jax.checking_leaks():
        jax.jit(lambda x: model(x, in_axes=None) if cls is nn.Stack
                else model(_scan_apply, x)[0])(jnp.array(0.))
    assert model.stacked is stored


@pytest.mark.parametrize('cls', [nn.Stack, nn.SeqStack])
def test_stacks_advance_owned_dropout_rngs(cls):
    @jax.jit
    def step(model, x):
        output = (model(x, in_axes=None) if cls is nn.Stack
                  else model(_scan_apply, x)[0])
        return output, model

    model = cls([nn.Dropout(0.5, rngs=nn.Rngs(i)) for i in range(2)])
    with jax.checking_leaks():
        first, model = step(model, jnp.ones(128))
        second, model = step(model, jnp.ones(128))
    assert not jnp.array_equal(first, second)


def test_seq_stack_threads_context_rng_in_carry():
    def apply(layer, carry):
        x, rngs = carry
        with nn.set_context_rng(rngs):
            output = layer(x)
        return (output, rngs), None

    model = nn.SeqStack([nn.Dropout(0.5), nn.Dropout(0.5)])
    @jax.jit
    def step(x, rngs):
        return model(apply, (x, rngs))[0]
    with jax.checking_leaks():
        first, rngs = step(jnp.ones(128), nn.Rngs(0))
        second, rngs = step(jnp.ones(128), rngs)
    assert not jnp.array_equal(first, second)


@pytest.mark.parametrize('cls', [nn.Stack, nn.SeqStack])
@pytest.mark.parametrize('axis_type', [AxisType.Auto, AxisType.Explicit])
def test_stack_sharding_metadata_tracks_layer_axis(cls, axis_type):
    mesh = Mesh(np.asarray(jax.devices()), ('tp',), axis_types=(axis_type,))
    with jax.set_mesh(mesh), map_logical_axis_names({'output': 'tp'}):
        width = 2 * mesh.size
        layers = [nn.Linear(width, width, rngs=nn.Rngs(i),
                            axis_names=('input', 'output')) for i in range(2)]
        model = cls(layers)
        assert model.stacked.kernel.axis_names == (None, 'input', 'output')
        assert model.stacked.kernel.partition_spec == P(None, None, 'tp')
        assert model.stacked.kernel.value.sharding.is_equivalent_to(
            NamedSharding(mesh, P(None, None, 'tp')), 3,
        )
        def apply(layer, x):
            assert layer.kernel.axis_names == ('input', 'output')
            assert layer.kernel.partition_spec == P(None, 'tp')
            y = layer(x)
            return y, None
        x = jnp.ones((1, width))
        if cls is nn.SeqStack:
            # Keep scan carry replicated under explicit meshes.
            def replicated_apply(layer, value):
                output, _ = apply(layer, value)
                if axis_type is AxisType.Explicit:
                    output = jax.sharding.reshard(output, P())
                return output, None
            result, _ = model(replicated_apply, x)
            expected = layers[1](layers[0](x))
        else:
            result = model(x, in_axes=None)
            expected = jnp.stack([layer(x) for layer in layers])
        assert jnp.allclose(result, expected, atol=1e-5)
        assert model.stacked.kernel.partition_spec == P(None, None, 'tp')
        assert layers[0].kernel.axis_names == ('input', 'output')


@pytest.mark.parametrize('cls', [nn.Stack, nn.SeqStack])
def test_stateful_captured_stacks_fail_without_mutating_stored_state(cls):
    model = cls([Counter(), Counter()])
    with pytest.raises(RuntimeError, match='Pass the container'):
        jax.jit(lambda x: model(x, in_axes=None) if cls is nn.Stack
                else model(_scan_apply, x)[0])(jnp.array(0.))
    assert jnp.all(model.stacked.count.value == 0)


@pytest.mark.parametrize('cls', [nn.Stack, nn.SeqStack])
@pytest.mark.parametrize('tracking', [False, True])
def test_stacks_support_batchnorm_nontracking_and_eval_modes(cls, tracking):
    layers = [nn.BatchNorm(2, momentum=1.0, track_running_stats=tracking)
              for _ in range(2)]
    x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    if tracking:
        for layer in layers:
            layer(x)  # BatchNorm currently updates statistics only eagerly.
            layer.eval()
    model = cls(layers)
    if cls is nn.Stack:
        result = model(x, in_axes=None)
        expected = jnp.stack([layer(x) for layer in layers])
    else:
        result, _ = model(_scan_apply, x)
        expected = layers[1](layers[0](x))
    assert jnp.allclose(result, expected, atol=1e-6)
    if tracking:
        assert jnp.allclose(model.stacked.running_mean.value,
                            jnp.stack([x.mean(0), x.mean(0)]))


@pytest.mark.parametrize('cls', [nn.Stack, nn.SeqStack])
def test_stacks_have_finite_parameter_gradients(cls):
    model = cls([nn.Linear(2, 2, rngs=nn.Rngs(i)) for i in range(2)])
    def objective(model):
        x = jnp.ones((3, 2))
        output = (model(x, in_axes=None) if cls is nn.Stack
                  else model(_scan_apply, x)[0])
        return output.sum()
    gradient = jax.jit(jax.grad(objective))(model)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    assert jnp.any(gradient.stacked.kernel.value != 0)


def test_stack_result_axis_does_not_move_state_axis():
    model = nn.Stack([Counter(), Counter(), Counter()])
    output = model(jnp.zeros(2), in_axes=None, out_axes=1)
    assert output.shape == (2, 3)
    assert model.stacked.count.shape == (3,)
    assert jnp.all(model.stacked.count.value == 1)


def test_stacks_reject_static_mutation():
    class ChangeConfig(nn.Module):
        def __init__(self):
            self.flag = False
        def __call__(self, x):
            self.flag = True
            return x
    model = nn.Stack([ChangeConfig(), ChangeConfig()])
    with pytest.raises(ValueError, match='static configuration'):
        model(jnp.ones(2))
    assert not model.stacked.flag


def test_grouped_scan_rejects_output_dtype_mismatch():
    model = nn.SeqStack([ConfiguredAdd(1, 'a'), ConfiguredAdd(2, 'b')])
    def apply(layer, carry):
        output = jnp.array(1, dtype=jnp.float32 if layer.mode == 'a' else jnp.int32)
        return carry, output
    with pytest.raises(ValueError, match='output shapes and dtypes'):
        model(apply, jnp.array(0))


class ParameterOutput(nn.Module):
    def __init__(self):
        with map_logical_axis_names({'input': 'tp'}):
            self.parameter = nn.Parameter(jnp.ones((2, 4)),
                                          axis_names=('input', 'output'),
                                          partition_spec=P('tp', None))
    def __call__(self, x):
        return {'parameter': self.parameter}


@pytest.mark.parametrize('axis', [0, 1, -1])
def test_stack_handles_parameter_output_metadata_locally(axis):
    model = nn.Stack([ParameterOutput() for _ in range(3)])
    result = model(jnp.array(0), in_axes=None, out_axes={'parameter': axis})
    parameter = result['parameter']
    index = axis if axis >= 0 else axis + 3
    names = ['input', 'output']
    names.insert(index, None)
    spec = ['tp', None]
    spec.insert(index, None)
    assert parameter.axis_names == tuple(names)
    assert parameter.partition_spec == P(*spec)
    assert model.stacked.parameter.axis_names == (None, 'input', 'output')
    assert model.stacked.parameter.partition_spec == P(None, 'tp', None)


def test_seq_stack_handles_parameter_output_metadata_locally():
    model = nn.SeqStack([ParameterOutput() for _ in range(3)])
    def apply(layer, carry):
        return carry, layer(carry)
    _, result = model(apply, jnp.array(0))
    assert result['parameter'].shape == (3, 2, 4)
    assert result['parameter'].axis_names == (None, 'input', 'output')
    assert result['parameter'].partition_spec == P(None, 'tp', None)
