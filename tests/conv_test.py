import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from taktiny import nn
from taktiny.utils.spmd import map_logical_axis_names


def ones(key, shape, dtype):
    del key
    return jnp.ones(shape, dtype=dtype)


def test_conv_1d_unbatched():
    conv = nn.Conv(
        1,
        1,
        kernel_size=3,
        bias=False,
        kernel_initializer=ones,
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
        kernel_initializer=ones,
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
        pad_mode='reflect',
        bias=False,
        kernel_initializer=ones,
        rngs=nn.Rngs(0),
    )
    x = jnp.asarray([[[1.0], [2.0], [4.0]]])
    padded = jnp.pad(x, ((0, 0), (1, 1), (0, 0)), mode='reflect')
    expected = jax.lax.conv_general_dilated(
        padded,
        conv.kernel.value,
        window_strides=(1,),
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )

    assert jnp.array_equal(conv(x), expected)


def test_conv_same_lower_places_odd_padding_at_start():
    same = nn.Conv(
        1,
        1,
        kernel_size=2,
        padding='SAME',
        bias=False,
        kernel_initializer=ones,
        rngs=nn.Rngs(0),
    )
    same_lower = nn.Conv(
        1,
        1,
        kernel_size=2,
        padding='same_lower',
        bias=False,
        kernel_initializer=ones,
        rngs=nn.Rngs(1),
    )
    x = jnp.asarray([[1.0], [2.0], [3.0]])

    assert same_lower.padding == 'SAME_LOWER'
    assert jnp.array_equal(same(x)[:, 0], jnp.asarray([3.0, 5.0, 3.0]))
    assert jnp.array_equal(
        same_lower(x)[:, 0],
        jnp.asarray([1.0, 3.0, 5.0]),
    )


def test_conv_validates_input_channels():
    conv = nn.Conv(2, 4, kernel_size=(3, 3), rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match='expected trailing input channels'):
        conv(jnp.ones((1, 8, 8, 3)))


def test_conv_requires_explicit_spatial_axes():
    conv = nn.Conv((2, 8), 4, kernel_size=1, rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match='expected an unbatched rank-3'):
        conv(jnp.ones((2, 8)))


def test_conv_rejects_empty_spatial_output():
    conv = nn.Conv((2, 8), 4, kernel_size=2, rngs=nn.Rngs(0))

    with pytest.raises(
        ValueError,
        match=r'spatial shape \(1,\).*effective kernel shape \(2,\)',
    ):
        conv(jnp.ones((1, 2, 8)))

    same_conv = nn.Conv(
        (2, 8),
        4,
        kernel_size=2,
        padding='SAME',
        rngs=nn.Rngs(1),
    )

    assert same_conv(jnp.ones((1, 2, 8))).shape == (1, 4)


def test_conv_supports_structured_input_and_output_channels():
    layer = nn.Conv(
        (2, 2),
        (2, 3),
        kernel_size=1,
        bias=False,
        kernel_initializer=ones,
        rngs=nn.Rngs(0),
    )
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 2, 2)

    output = jax.jit(layer)(x)
    expected = jnp.sum(x, axis=(-2, -1), keepdims=True)
    expected = jnp.broadcast_to(expected, (2, 2, 2, 3))

    assert layer.kernel.shape == (1, 2, 2, 2, 3)
    assert output.shape == (2, 2, 2, 3)
    assert jnp.array_equal(output, expected)
    assert layer.extra_repr() == '2×2 ➤ 2×3, k=1, s=1'


def test_conv_groups_partition_first_structured_channel_axis():
    layer = nn.Conv(
        (4, 2),
        (4, 3),
        kernel_size=1,
        groups=2,
        bias=False,
        kernel_initializer=ones,
        rngs=nn.Rngs(0),
    )
    x = jnp.arange(8, dtype=jnp.float32).reshape(1, 1, 4, 2)

    output = layer(x)
    first_group = jnp.sum(x[..., :2, :], axis=(-2, -1), keepdims=True)
    second_group = jnp.sum(x[..., 2:, :], axis=(-2, -1), keepdims=True)
    expected = jnp.concatenate(
        [
            jnp.broadcast_to(first_group, (1, 1, 2, 3)),
            jnp.broadcast_to(second_group, (1, 1, 2, 3)),
        ],
        axis=-2,
    )

    assert layer.kernel.shape == (1, 2, 2, 4, 3)
    assert output.shape == (1, 1, 4, 3)
    assert jnp.array_equal(output, expected)


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

    assert layer.kernel.axis_names == ('kernel', 'input', 'output')
    assert layer.bias.axis_names == ('output',)


def test_conv_records_structured_parameter_configuration():
    layer = nn.Conv(
        (2, 3),
        (4, 5),
        kernel_size=(3, 2),
        axis_names=(
            'height',
            'width',
            'input_group',
            'input_feature',
            'output_group',
            'output_feature',
        ),
        kernel_metadata={'kind': 'filter'},
        bias_metadata={'kind': 'offset'},
        rngs=nn.Rngs(0),
    )

    assert layer.kernel.shape == (3, 2, 2, 3, 4, 5)
    assert layer.bias.shape == (4, 5)
    assert layer.bias.axis_names == ('output_group', 'output_feature')
    assert layer.kernel.metadata == {'kind': 'filter'}
    assert layer.bias.metadata == {'kind': 'offset'}


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
        rngs=nn.Rngs(0),
    )
    apply = lambda value: layer(value, out_sharding=out_sharding)
    x = jnp.ones((1, 3, 1))

    jaxpr = jax.make_jaxpr(apply)(x).jaxpr
    output = jax.jit(apply)(x)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)


def test_conv_transpose_supports_structured_channels():
    layer = nn.ConvTranspose(
        [2, 2],
        [2, 3],
        kernel_size=[1],
        bias=False,
        kernel_initializer=ones,
        rngs=nn.Rngs(0),
    )
    x = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)

    output = jax.jit(layer)(x)
    expected = jnp.sum(x, axis=(-2, -1), keepdims=True)
    expected = jnp.broadcast_to(expected, (2, 2, 3))

    assert layer.kernel.shape == (1, 2, 2, 2, 3)
    assert output.shape == (2, 2, 3)
    assert jnp.array_equal(output, expected)
    assert layer.extra_repr() == '2×2 ➤ 2×3, k=1, s=1'


def test_conv_transpose_records_parameter_configuration():
    layer = nn.ConvTranspose(
        (2, 3),
        (4, 5),
        kernel_size=(3, 2),
        axis_names=(
            'height',
            'width',
            'input_group',
            'input_feature',
            'output_group',
            'output_feature',
        ),
        kernel_metadata={'kind': 'upsampling_filter'},
        bias_metadata={'kind': 'offset'},
        rngs=nn.Rngs(0),
    )

    assert layer.kernel.shape == (3, 2, 2, 3, 4, 5)
    assert layer.bias.shape == (4, 5)
    assert layer.kernel.axis_names == (
        'height',
        'width',
        'input_group',
        'input_feature',
        'output_group',
        'output_feature',
    )
    assert layer.bias.axis_names == ('output_group', 'output_feature')
    assert layer.kernel.metadata == {'kind': 'upsampling_filter'}
    assert layer.bias.metadata == {'kind': 'offset'}


def test_conv_transpose_applies_explicit_parameter_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('model',))
    kernel_spec = P(None, None, 'model')
    bias_spec = P('model')

    with jax.set_mesh(mesh):
        layer = nn.ConvTranspose(
            2,
            4,
            kernel_size=3,
            partition_spec=kernel_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == kernel_spec
    assert layer.bias.partition_spec == bias_spec
    assert layer.kernel.value.sharding.spec == kernel_spec
    assert layer.bias.value.sharding.spec == bias_spec


def test_conv_transpose_logical_axes_override_explicit_sharding():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ('data', 'model'))
    explicit_spec = P('data', None, None)
    logical_kernel_spec = P(None, None, 'model')
    logical_bias_spec = P('model')

    with jax.set_mesh(mesh), map_logical_axis_names({'output': 'model'}):
        layer = nn.ConvTranspose(
            2,
            4,
            kernel_size=3,
            axis_names=('kernel', 'input', 'output'),
            partition_spec=explicit_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == logical_kernel_spec
    assert layer.bias.partition_spec == logical_bias_spec
    assert layer.kernel.value.sharding.spec == logical_kernel_spec
    assert layer.bias.value.sharding.spec == logical_bias_spec


def test_conv_transpose_supports_quantized_kernels():
    rule = qwix.QuantizationRule(
        op_names=('conv_general_dilated',),
        weight_qtype='int8',
    )
    layer = nn.ConvTranspose(
        2,
        3,
        kernel_size=1,
        bias=False,
        quant=rule,
        rngs=nn.Rngs(0),
    )
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 4, 2) / 10

    output = jax.jit(layer)(x)
    expected = jax.lax.conv_transpose(
        x,
        qwix.dequantize(layer.kernel.value),
        strides=(1,),
        padding=((0, 0),),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )

    assert isinstance(layer.kernel.value, qwix.QArray)
    assert jnp.allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_conv_transpose_forwards_custom_convolution_options():
    calls = []

    def custom_convolution(**kwargs):
        calls.append(kwargs)
        return jax.lax.conv_general_dilated(**kwargs)

    layer = nn.ConvTranspose(
        2,
        3,
        kernel_size=1,
        stride=2,
        bias=False,
        dot_general=custom_convolution,
        precision=jax.lax.Precision.HIGH,
        preferred_element_type=jnp.float32,
        rngs=nn.Rngs(0),
    )

    output = layer(jnp.ones((1, 4, 2)))

    assert output.shape == (1, 7, 3)
    assert len(calls) == 1
    assert calls[0]['window_strides'] == (1,)
    assert calls[0]['lhs_dilation'] == (2,)
    assert calls[0]['precision'] == jax.lax.Precision.HIGH
    assert calls[0]['preferred_element_type'] == jnp.float32


def test_conv_applies_explicit_parameter_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('model',))
    kernel_spec = P(None, None, 'model')
    bias_spec = P('model')

    with jax.set_mesh(mesh):
        layer = nn.Conv(
            2,
            4,
            kernel_size=3,
            partition_spec=kernel_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == kernel_spec
    assert layer.bias.partition_spec == bias_spec
    assert layer.kernel.value.sharding.spec == kernel_spec
    assert layer.bias.value.sharding.spec == bias_spec


def test_conv_logical_axes_override_explicit_parameter_sharding():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ('data', 'model'))
    explicit_spec = P('data', None, None)
    logical_kernel_spec = P(None, None, 'model')
    logical_bias_spec = P('model')

    with jax.set_mesh(mesh), map_logical_axis_names({'output': 'model'}):
        layer = nn.Conv(
            2,
            4,
            kernel_size=3,
            axis_names=('kernel', 'input', 'output'),
            partition_spec=explicit_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.kernel.partition_spec == logical_kernel_spec
    assert layer.bias.partition_spec == logical_bias_spec
    assert layer.kernel.value.sharding.spec == logical_kernel_spec
    assert layer.bias.value.sharding.spec == logical_bias_spec


def test_conv_supports_quantized_kernels():
    rule = qwix.QuantizationRule(
        op_names=('conv_general_dilated',),
        weight_qtype='int8',
    )
    layer = nn.Conv(
        2,
        3,
        kernel_size=1,
        bias=False,
        quant=rule,
        rngs=nn.Rngs(0),
    )
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 4, 2) / 10

    output = jax.jit(layer)(x)
    expected = jax.lax.conv_general_dilated(
        x,
        qwix.dequantize(layer.kernel.value),
        window_strides=(1,),
        padding=((0, 0),),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )

    assert isinstance(layer.kernel.value, qwix.QArray)
    assert jnp.allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_conv_forwards_precision_to_custom_convolution():
    calls = []

    def custom_convolution(**kwargs):
        calls.append(kwargs)
        return jax.lax.conv_general_dilated(**kwargs)

    layer = nn.Conv(
        2,
        3,
        kernel_size=1,
        bias=False,
        dot_general=custom_convolution,
        precision=jax.lax.Precision.HIGH,
        preferred_element_type=jnp.float32,
        rngs=nn.Rngs(0),
    )

    output = layer(jnp.ones((1, 4, 2)))

    assert output.shape == (1, 4, 3)
    assert len(calls) == 1
    assert calls[0]['precision'] == jax.lax.Precision.HIGH
    assert calls[0]['preferred_element_type'] == jnp.float32
