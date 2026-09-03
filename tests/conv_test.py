import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from taktiny import nn
from taktiny.utils.typing import ShardMode


def ones(key, shape, dtype):
    del key
    return jnp.ones(shape, dtype=dtype)


def test_conv_1d_unbatched():
    conv = nn.Conv(
        1,
        1,
        kernel_size=3,
        bias=False,
        initializer=ones,
        rngs=nn.Rngs(0),
    )
    x = jnp.asarray([[1.0], [2.0], [3.0], [4.0]])

    output = conv(x)

    assert output.shape == (2, 1)
    assert jnp.array_equal(output[:, 0], jnp.asarray([6.0, 9.0]))


@pytest.mark.parametrize(
    ('kernel_size', 'input_shape', 'expected_shape'),
    [
        ((3, 2), (2, 7, 6, 3), (2, 3, 5, 5)),
        ((2, 2, 2), (2, 5, 4, 3, 3), (2, 2, 3, 2, 5)),
    ],
)
def test_conv_supports_arbitrary_spatial_rank(
    kernel_size,
    input_shape,
    expected_shape,
):
    conv = nn.Conv(
        3,
        5,
        kernel_size=kernel_size,
        stride=(2,) + (1,) * (len(kernel_size) - 1),
        bias=False,
        rngs=nn.Rngs(0),
    )

    output = conv(jnp.ones(input_shape))

    assert output.shape == expected_shape


def test_conv_grouped_channels():
    conv = nn.Conv(
        4,
        4,
        kernel_size=1,
        groups=2,
        bias=False,
        initializer=ones,
        rngs=nn.Rngs(0),
    )
    x = jnp.asarray([[[1.0, 2.0, 3.0, 4.0]]])

    output = conv(x)

    assert jnp.array_equal(output, jnp.asarray([[[3.0, 3.0, 7.0, 7.0]]]))


def test_conv_reflect_padding_matches_explicit_lax_convolution():
    conv = nn.Conv(
        1,
        1,
        kernel_size=3,
        padding=1,
        padding_mode='reflect',
        bias=False,
        initializer=ones,
        rngs=nn.Rngs(0),
    )
    x = jnp.asarray([[[1.0], [2.0], [4.0]]])
    padded = jnp.pad(x, ((0, 0), (1, 1), (0, 0)), mode='reflect')
    expected = jax.lax.conv_general_dilated(
        padded,
        conv.weight.value,
        window_strides=(1,),
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )

    assert jnp.array_equal(conv(x), expected)


def test_conv_validates_input_channels():
    conv = nn.Conv(2, 4, kernel_size=(3, 3), rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match='expected 2 input channels'):
        conv(jnp.ones((1, 8, 8, 3)))


@pytest.mark.parametrize('module_type', [nn.Conv, nn.ConvTranspose])
def test_convolution_parameter_axis_names(module_type):
    layer = module_type(
        2,
        4,
        kernel_size=3,
        bias=True,
        axis_names=('kernel', 'input', 'output'),
        rngs=nn.Rngs(0),
    )

    assert layer.weight.axis_names == ('kernel', 'input', 'output')
    assert layer.bias.axis_names == ('output',)


def test_convolution_validates_parameter_axis_names():
    with pytest.raises(ValueError, match='axis_names length 2'):
        nn.Conv(
            2,
            4,
            kernel_size=3,
            axis_names=('input', 'output'),
            rngs=nn.Rngs(0),
        )


@pytest.mark.parametrize('module_type', [nn.Conv, nn.ConvTranspose])
def test_convolution_explicit_sharding_covers_final_output(module_type):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = module_type(
        1,
        2,
        kernel_size=1,
        bias=True,
        shard_mode=ShardMode.EXPLICIT,
        rngs=nn.Rngs(0),
    )
    apply = lambda value: layer(value, out_sharding=out_sharding)
    x = jnp.ones((1, 3, 1))

    jaxpr = jax.make_jaxpr(apply)(x).jaxpr
    output = jax.jit(apply)(x)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)
