import jax
import jax.numpy as jnp
import pytest

from taktiny import nn


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


def test_stack_exposes_vmap_axis_controls_and_broadcast_axes():
    modules = nn.Stack(
        [AxisAdd(1), AxisAdd(2), AxisAdd(3)],
        axis_name='layers',
    )

    output = modules(jnp.asarray(10), in_axes=None)

    assert len(modules) == 3
    assert jnp.array_equal(output, jnp.asarray([36, 36, 36]))


def test_stacks_report_static_configuration_and_leaf_shape_mismatches():
    with pytest.raises(ValueError, match='static configuration'):
        nn.SeqStack([ConfiguredAdd(1, 'a'), ConfiguredAdd(2, 'b')])
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
