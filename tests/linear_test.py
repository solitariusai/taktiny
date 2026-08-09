import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.utils.typing import ShardMode


def test_linear_explicit_sharding_covers_biased_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = nn.Linear(
        2,
        3,
        bias=True,
        shard_mode=ShardMode.EXPLICIT,
        rngs=nn.Rngs(0),
    )
    apply = lambda value: layer(value, out_sharding=out_sharding)
    x = jnp.ones((2, 2))

    jaxpr = jax.make_jaxpr(apply)(x).jaxpr
    output = jax.jit(apply)(x)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)


def test_bilinear_matches_einsum_with_bias():
    layer = nn.Bilinear(3, 4, 2, rngs=nn.Rngs(0))
    layer.weight.value = jnp.arange(24, dtype=jnp.float32).reshape(3, 4, 2) / 24
    layer.bias.value = jnp.asarray([0.25, -0.5])
    x1 = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) / 10
    x2 = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10

    output = jax.jit(layer)(x1, x2)
    expected = jnp.einsum('...i,ijo,...j->...o', x1, layer.weight.value, x2)
    expected += layer.bias.value

    assert jnp.allclose(output, expected, rtol=1e-5, atol=2e-5)


def test_bilinear_supports_multi_axis_feature_shapes():
    layer = nn.Bilinear(
        (2, 3),
        (2, 2),
        (2, 2),
        bias=False,
        rngs=nn.Rngs(1),
    )
    x1 = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3)
    x2 = jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2)

    output = layer(x1, x2)
    expected = jnp.einsum(
        'buv,uvijop,bij->bop',
        x1,
        layer.weight.value,
        x2,
    )

    assert output.shape == (4, 2, 2)
    assert jnp.allclose(output, expected, rtol=1e-5, atol=2e-5)


def test_bilinear_records_parameter_axis_names():
    layer = nn.Bilinear(
        3,
        4,
        2,
        axis_names=('left', 'right', 'output'),
        rngs=nn.Rngs(0),
    )

    assert layer.weight.axis_names == ('left', 'right', 'output')
    assert layer.bias.axis_names == ('output',)
    assert layer.weight.input_axis_count == 2


def test_bilinear_supports_qwix_weight_storage():
    layer = nn.Bilinear(3, 4, 2, bias=False, rngs=nn.Rngs(0))
    layer.weight.value = qwix.quantize(
        layer.weight.value,
        'int8',
        channelwise_axes=(2,),
        tiled_axes={1: 2},
    )
    x1 = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10
    x2 = jnp.arange(20, dtype=jnp.float32).reshape(5, 4) / 10

    output = jax.jit(layer)(x1, x2)
    expected = jnp.einsum(
        'bi,ijo,bj->bo',
        x1,
        qwix.dequantize(layer.weight.value),
        x2,
    )

    assert jnp.allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_bilinear_explicit_sharding_covers_biased_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = nn.Bilinear(
        2,
        3,
        4,
        shard_mode=ShardMode.EXPLICIT,
        rngs=nn.Rngs(0),
    )
    apply = lambda left, right: layer(
        left,
        right,
        out_sharding=out_sharding,
    )

    jaxpr = jax.make_jaxpr(apply)(jnp.ones((5, 2)), jnp.ones((5, 3))).jaxpr
    output = jax.jit(apply)(jnp.ones((5, 2)), jnp.ones((5, 3)))

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)


def test_bilinear_validates_input_contract():
    layer = nn.Bilinear(2, 3, 4, rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match='x1 trailing shape'):
        layer(jnp.ones((5, 3)), jnp.ones((5, 3)))
    with pytest.raises(ValueError, match='identical leading shapes'):
        layer(jnp.ones((5, 2)), jnp.ones((4, 3)))
    with pytest.raises(ValueError, match='axis_names length'):
        nn.Bilinear(
            2,
            3,
            4,
            axis_names=('input', 'output'),
            rngs=nn.Rngs(0),
        )
