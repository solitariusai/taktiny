import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.utils.spmd import map_logical_axis_names


def test_linear_matches_matrix_multiplication_with_bias():
    layer = nn.Linear(3, 2, rngs=nn.Rngs(0))
    kernel = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10
    bias = jnp.asarray([0.25, -0.5])
    layer.load_state_dict({'kernel': kernel, 'bias': bias})
    x = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) / 10

    output = jax.jit(layer)(x)
    expected = x @ kernel + bias

    assert output.shape == (2, 3, 2)
    assert jnp.allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_linear_supports_multi_axis_feature_shapes_without_bias():
    layer = nn.Linear(
        (2, 3),
        (2, 2),
        bias=False,
        rngs=nn.Rngs(0),
    )
    kernel = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 2, 2) / 20
    layer.load_state_dict({'kernel': kernel})
    x = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3) / 10

    output = layer(x)
    expected = jnp.einsum('bij,ijop->bop', x, kernel)

    assert layer.bias is None
    assert output.shape == (4, 2, 2)
    assert jnp.allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_linear_records_parameter_configuration():
    layer = nn.Linear(
        (2, 3),
        (4, 5),
        axis_names=('row', 'column', 'head', 'feature'),
        kernel_metadata={'kind': 'projection'},
        bias_metadata={'kind': 'offset'},
        rngs=nn.Rngs(0),
    )

    assert layer.kernel.shape == (2, 3, 4, 5)
    assert layer.bias.shape == (4, 5)
    assert layer.kernel.axis_names == ('row', 'column', 'head', 'feature')
    assert layer.bias.axis_names == ('head', 'feature')
    assert layer.kernel.metadata == {'kind': 'projection'}
    assert layer.bias.metadata == {'kind': 'offset'}
    assert layer.extra_repr() == '2×3 ➤ 4×5'


def test_linear_normalizes_and_validates_feature_shapes():
    layer = nn.Linear([2, 3], [4, 5], rngs=nn.Rngs(0))

    assert layer.in_features == (2, 3)
    assert layer.out_features == (4, 5)

    with pytest.raises(ValueError, match='in_features.*at least one dimension'):
        nn.Linear([], 4, rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='in_features.*positive integer'):
        nn.Linear([2, 0], 4, rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='out_features.*positive integer'):
        nn.Linear(2, [True], rngs=nn.Rngs(0))


def test_linear_applies_explicit_parameter_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('model',))
    kernel_spec = P(None, 'model')
    bias_spec = P('model')

    with jax.set_mesh(mesh):
        layer = nn.Linear(
            4,
            3,
            partition_spec=kernel_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == kernel_spec
    assert layer.bias.partition_spec == bias_spec
    assert layer.kernel.value.sharding.is_equivalent_to(
        NamedSharding(mesh, kernel_spec),
        layer.kernel.ndim,
    )
    assert layer.bias.value.sharding.is_equivalent_to(
        NamedSharding(mesh, bias_spec),
        layer.bias.ndim,
    )


def test_linear_maps_logical_axes_to_parameter_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('model',))
    kernel_spec = P(None, 'model')
    bias_spec = P('model')

    with jax.set_mesh(mesh), map_logical_axis_names({'output': 'model'}):
        layer = nn.Linear(
            4,
            3,
            axis_names=('input', 'output'),
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == kernel_spec
    assert layer.bias.partition_spec == bias_spec
    assert layer.kernel.value.sharding.is_equivalent_to(
        NamedSharding(mesh, kernel_spec),
        layer.kernel.ndim,
    )
    assert layer.bias.value.sharding.is_equivalent_to(
        NamedSharding(mesh, bias_spec),
        layer.bias.ndim,
    )


def test_linear_logical_axes_override_explicit_parameter_sharding():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ('data', 'model'))
    explicit_spec = P('data', None)
    logical_kernel_spec = P(None, 'model')
    logical_bias_spec = P('model')

    with jax.set_mesh(mesh), map_logical_axis_names({'output': 'model'}):
        layer = nn.Linear(
            4,
            3,
            axis_names=('input', 'output'),
            partition_spec=explicit_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == logical_kernel_spec
    assert layer.bias.partition_spec == logical_bias_spec
    assert layer.kernel.value.sharding.spec == logical_kernel_spec
    assert layer.bias.value.sharding.spec == logical_bias_spec


def test_linear_supports_quantized_kernels():
    layer = nn.Linear(
        4,
        3,
        bias=False,
        quant='int8',
        rngs=nn.Rngs(0),
    )
    x = jnp.arange(20, dtype=jnp.float32).reshape(5, 4) / 10

    output = jax.jit(layer)(x)
    expected = x @ qwix.dequantize(layer.kernel.value)

    assert isinstance(layer.kernel.value, qwix.QArray)
    assert jnp.allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_linear_out_sharding_covers_biased_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = nn.Linear(2, 3, rngs=nn.Rngs(0))
    apply = lambda value: layer(value, out_sharding=out_sharding)
    x = jnp.ones((2, 2))

    jaxpr = jax.make_jaxpr(apply)(x).jaxpr
    output = jax.jit(apply)(x)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)


def test_bilinear_matches_einsum_with_bias():
    layer = nn.Bilinear(3, 4, 2, rngs=nn.Rngs(0))
    kernel = jnp.arange(24, dtype=jnp.float32).reshape(3, 4, 2) / 24
    bias = jnp.asarray([0.25, -0.5])
    layer.load_state_dict({'kernel': kernel, 'bias': bias})
    x1 = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) / 10
    x2 = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10

    output = jax.jit(layer)(x1, x2)
    expected = jnp.einsum('...i,ijo,...j->...o', x1, kernel, x2) + bias

    assert output.shape == (2, 3, 2)
    assert jnp.allclose(output, expected, rtol=1e-5, atol=2e-5)


def test_bilinear_supports_multi_axis_feature_shapes_without_bias():
    layer = nn.Bilinear(
        (2, 3),
        (2, 2),
        (2, 2),
        bias=False,
        rngs=nn.Rngs(1),
    )
    kernel = jnp.arange(96, dtype=jnp.float32).reshape(2, 3, 2, 2, 2, 2)
    layer.load_state_dict({'kernel': kernel})
    x1 = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3) / 10
    x2 = jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2) / 10

    output = layer(x1, x2)
    expected = jnp.einsum('buv,uvijop,bij->bop', x1, kernel, x2)

    assert layer.bias is None
    assert output.shape == (4, 2, 2)
    assert jnp.allclose(output, expected, rtol=1e-5, atol=2e-5)
    assert layer.extra_repr() == '2×3, 2×2 ➤ 2×2'


def test_bilinear_records_parameter_configuration():
    layer = nn.Bilinear(
        3,
        4,
        2,
        axis_names=('left', 'right', 'output'),
        kernel_metadata={'kind': 'interaction'},
        bias_metadata={'kind': 'offset'},
        rngs=nn.Rngs(0),
    )

    assert layer.kernel.shape == (3, 4, 2)
    assert layer.bias.shape == (2,)
    assert layer.kernel.axis_names == ('left', 'right', 'output')
    assert layer.bias.axis_names == ('output',)
    assert layer.kernel.metadata == {'kind': 'interaction'}
    assert layer.bias.metadata == {'kind': 'offset'}


def test_bilinear_applies_explicit_parameter_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('model',))
    kernel_spec = P(None, None, 'model')
    bias_spec = P('model')

    with jax.set_mesh(mesh):
        layer = nn.Bilinear(
            3,
            4,
            2,
            partition_spec=kernel_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == kernel_spec
    assert layer.bias.partition_spec == bias_spec
    assert layer.kernel.value.sharding.is_equivalent_to(
        NamedSharding(mesh, kernel_spec),
        layer.kernel.ndim,
    )
    assert layer.bias.value.sharding.is_equivalent_to(
        NamedSharding(mesh, bias_spec),
        layer.bias.ndim,
    )


def test_bilinear_maps_logical_axes_to_parameter_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('model',))
    kernel_spec = P(None, None, 'model')
    bias_spec = P('model')

    with jax.set_mesh(mesh), map_logical_axis_names({'output': 'model'}):
        layer = nn.Bilinear(
            3,
            4,
            2,
            axis_names=('left', 'right', 'output'),
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == kernel_spec
    assert layer.bias.partition_spec == bias_spec
    assert layer.kernel.value.sharding.is_equivalent_to(
        NamedSharding(mesh, kernel_spec),
        layer.kernel.ndim,
    )
    assert layer.bias.value.sharding.is_equivalent_to(
        NamedSharding(mesh, bias_spec),
        layer.bias.ndim,
    )


def test_bilinear_logical_axes_override_explicit_parameter_sharding():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ('data', 'model'))
    explicit_spec = P('data', None, None)
    logical_kernel_spec = P(None, None, 'model')
    logical_bias_spec = P('model')

    with jax.set_mesh(mesh), map_logical_axis_names({'output': 'model'}):
        layer = nn.Bilinear(
            3,
            4,
            2,
            axis_names=('left', 'right', 'output'),
            partition_spec=explicit_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == logical_kernel_spec
    assert layer.bias.partition_spec == logical_bias_spec
    assert layer.kernel.value.sharding.spec == logical_kernel_spec
    assert layer.bias.value.sharding.spec == logical_bias_spec


def test_bilinear_supports_quantized_kernels():
    layer = nn.Bilinear(
        3,
        4,
        2,
        bias=False,
        quant='int8',
        rngs=nn.Rngs(0),
    )
    x1 = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10
    x2 = jnp.arange(20, dtype=jnp.float32).reshape(5, 4) / 10

    output = jax.jit(layer)(x1, x2)
    expected = jnp.einsum(
        'bi,ijo,bj->bo',
        x1,
        qwix.dequantize(layer.kernel.value),
        x2,
    )

    assert isinstance(layer.kernel.value, qwix.QArray)
    assert jnp.allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_bilinear_out_sharding_covers_biased_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = nn.Bilinear(2, 3, 4, rngs=nn.Rngs(0))
    apply = lambda left, right: layer(
        left,
        right,
        out_sharding=out_sharding,
    )
    x1 = jnp.ones((5, 2))
    x2 = jnp.ones((5, 3))

    jaxpr = jax.make_jaxpr(apply)(x1, x2).jaxpr
    output = jax.jit(apply)(x1, x2)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)


def test_bilinear_validates_input_contract():
    layer = nn.Bilinear((2, 3), 4, 5, rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match='fewer axes'):
        layer(jnp.ones((3,)), jnp.ones((4,)))
    with pytest.raises(ValueError, match='x1 trailing shape'):
        layer(jnp.ones((2, 2, 4)), jnp.ones((2, 4)))
    with pytest.raises(ValueError, match='x2 trailing shape'):
        layer(jnp.ones((2, 2, 3)), jnp.ones((2, 5)))
    with pytest.raises(ValueError, match='identical leading shapes'):
        layer(jnp.ones((2, 2, 3)), jnp.ones((3, 4)))


def test_bilinear_validates_feature_and_axis_name_shapes():
    with pytest.raises(ValueError, match='in1_features.*positive integer'):
        nn.Bilinear(0, 3, 4, rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='axis_names length'):
        nn.Bilinear(
            2,
            3,
            4,
            axis_names=('input', 'output'),
            rngs=nn.Rngs(0),
        )
