import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from taktiny import nn
from taktiny.utils.spmd import map_logical_axis_names


def test_layer_norm_matches_reference_without_affine():
    x = jax.random.normal(jax.random.key(0), (2, 3, 4))
    layer = nn.LayerNorm(4, epsilon=1e-6, elementwise_affine=False)

    output = jax.jit(layer)(x)
    expected = (x - jnp.mean(x, axis=-1, keepdims=True)) * jax.lax.rsqrt(
        jnp.var(x, axis=-1, keepdims=True) + 1e-6
    )

    assert layer.scale is None
    assert layer.bias is None
    assert output.shape == x.shape
    assert jnp.allclose(output, expected, atol=1e-6)


def test_layer_norm_supports_nontrailing_axes_and_declared_order():
    x = jax.random.normal(jax.random.key(0), (2, 3, 4))
    layer = nn.LayerNorm(
        (4, 2),
        axes=(2, 0),
        axis_names=('feature', 'sample_feature'),
        scale_metadata={'kind': 'gain'},
        bias_metadata={'kind': 'offset'},
    )
    layer.load_state_dict({
        'scale': jnp.arange(1, 9, dtype=jnp.float32).reshape(4, 2),
        'bias': jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
    })

    output = layer(x)
    normalized = (x - jnp.mean(x, axis=(2, 0), keepdims=True)) * (
        jax.lax.rsqrt(jnp.var(x, axis=(2, 0), keepdims=True) + 1e-6)
    )
    expected = (
        normalized * layer.scale.value.T[:, None, :]
        + layer.bias.value.T[:, None, :]
    )

    assert layer.scale.axis_names == ('feature', 'sample_feature')
    assert layer.bias.axis_names == ('feature', 'sample_feature')
    assert layer.scale.metadata == {'kind': 'gain'}
    assert layer.bias.metadata == {'kind': 'offset'}
    assert jnp.allclose(output, expected, atol=1e-6)


def test_rms_norm_supports_multiple_dimensions_and_bfloat16_statistics():
    x = jax.random.normal(jax.random.key(0), (2, 3, 4)).astype(jnp.bfloat16)
    layer = nn.RMSNorm(
        (3, 4),
        epsilon=1e-6,
        elementwise_affine=False,
    )

    output = jax.jit(layer)(x)
    value = x.astype(jnp.float32)
    expected = value * jax.lax.rsqrt(
        jnp.mean(jnp.square(value), axis=(-2, -1), keepdims=True) + 1e-6
    )

    assert output.dtype == jnp.bfloat16
    assert layer.scale is None
    assert jnp.allclose(output.astype(jnp.float32), expected, atol=1e-2)


def test_rms_norm_supports_scale_and_bias():
    x = jax.random.normal(jax.random.key(1), (2, 3, 4))
    layer = nn.RMSNorm(4, epsilon=1e-6, bias=True)
    layer.load_state_dict({
        'scale': jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        'bias': jnp.asarray([-1.0, 0.0, 1.0, 2.0]),
    })

    output = jax.jit(layer)(x)
    expected = x * jax.lax.rsqrt(
        jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6
    )
    expected = expected * layer.scale.value + layer.bias.value

    assert jnp.allclose(output, expected, atol=1e-6)


def _group_norm_reference(
    x,
    *,
    groups,
    channel_axis,
    batch_axes,
    epsilon,
):
    channel_axis %= x.ndim
    canonical_batch_axes = tuple(axis % x.ndim for axis in batch_axes)
    spatial_axes = tuple(
        axis
        for axis in range(x.ndim)
        if axis not in canonical_batch_axes and axis != channel_axis
    )
    permutation = canonical_batch_axes + spatial_axes + (channel_axis,)
    transposed = jnp.transpose(x, permutation)
    grouped = transposed.reshape(
        *transposed.shape[:-1],
        groups,
        transposed.shape[-1] // groups,
    )
    group_axis = grouped.ndim - 2
    reduction_axes = tuple(
        axis
        for axis in range(len(canonical_batch_axes), grouped.ndim)
        if axis != group_axis
    )
    normalized = grouped - jnp.mean(grouped, axis=reduction_axes, keepdims=True)
    normalized *= jax.lax.rsqrt(
        jnp.var(grouped, axis=reduction_axes, keepdims=True) + epsilon
    )
    normalized = normalized.reshape(transposed.shape)
    inverse = tuple(sorted(range(x.ndim), key=permutation.__getitem__))
    return jnp.transpose(normalized, inverse)


@pytest.mark.parametrize(
    ('shape', 'channel_axes', 'batch_axes'),
    [
        ((2, 3, 5, 8), -1, 0),
        ((2, 8, 3, 5), 1, 0),
        ((3, 5, 8), -1, ()),
    ],
)
def test_group_norm_supports_channel_layouts_and_unbatched_inputs(
    shape,
    channel_axes,
    batch_axes,
):
    x = jax.random.normal(jax.random.key(0), shape)
    layer = nn.GroupNorm(
        4,
        8,
        elementwise_affine=False,
        channel_axes=channel_axes,
        batch_axes=batch_axes,
    )

    output = jax.jit(layer)(x)
    expected = _group_norm_reference(
        x,
        groups=4,
        channel_axis=channel_axes,
        batch_axes=(batch_axes,) if isinstance(batch_axes, int) else batch_axes,
        epsilon=1e-6,
    )

    assert output.shape == x.shape
    assert jnp.allclose(output, expected, atol=1e-6)


def test_group_norm_supports_nd_channels_and_groups():
    x = jax.random.normal(jax.random.key(2), (2, 5, 4, 6))
    layer = nn.GroupNorm(
        (2, 2),
        (4, 6),
        elementwise_affine=False,
    )

    output = jax.jit(layer)(x)
    grouped = output.reshape(2, 5, 2, 2, 2, 3)
    mean = jnp.mean(grouped, axis=(1, 3, 5))
    variance = jnp.var(grouped, axis=(1, 3, 5))

    assert output.shape == x.shape
    assert jnp.allclose(mean, 0.0, atol=2e-6)
    assert jnp.allclose(variance, 1.0, atol=2e-5)


def test_group_norm_is_differentiable():
    layer = nn.GroupNorm(2, 4)
    x = jax.random.normal(jax.random.key(0), (2, 3, 4))

    gradient = jax.jit(jax.grad(lambda value: jnp.sum(layer(value) ** 2)))(x)

    assert gradient.shape == x.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_batch_norm_uses_module_mode_and_updates_running_state():
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    layer = nn.BatchNorm(4, momentum=1.0)

    assert layer.is_training
    training_output = layer(x)
    mean = jnp.mean(x, axis=(0, 1))
    variance = jnp.var(x, axis=(0, 1))
    expected = (x - mean) * jax.lax.rsqrt(variance + 1e-6)

    assert jnp.allclose(training_output, expected, atol=1e-6)
    assert jnp.array_equal(layer.running_mean.value, mean)
    assert jnp.array_equal(layer.running_var.value, variance)
    assert int(layer.num_batches_tracked.value) == 1
    assert not layer.running_mean.trainable
    assert not layer.running_var.trainable

    layer.eval()
    assert not layer.is_training
    assert jnp.allclose(layer(x), training_output, atol=1e-6)

    layer.reset_running_stats()
    assert jnp.array_equal(layer.running_mean.value, jnp.zeros((4,)))
    assert jnp.array_equal(layer.running_var.value, jnp.ones((4,)))
    assert int(layer.num_batches_tracked.value) == 0


def test_batch_norm_supports_nd_features_in_declared_axis_order():
    x = jax.random.normal(jax.random.key(3), (2, 3, 4, 5))
    layer = nn.BatchNorm(
        (5, 3),
        axes=(3, 1),
        momentum=1.0,
        elementwise_affine=False,
    )

    output = layer(x)
    mean = jnp.mean(x, axis=(0, 2), keepdims=True)
    variance = jnp.var(x, axis=(0, 2), keepdims=True)
    expected = (x - mean) * jax.lax.rsqrt(variance + 1e-6)

    assert layer.running_mean.shape == (5, 3)
    assert jnp.array_equal(layer.running_mean.value, mean[0, :, 0, :].T)
    assert output.shape == x.shape
    assert jnp.allclose(output, expected, atol=1e-6)


def test_batch_norm_without_running_stats_is_jittable_in_both_modes():
    x = jax.random.normal(jax.random.key(0), (2, 4, 3, 5))
    layer = nn.BatchNorm(
        4,
        elementwise_affine=False,
        track_running_stats=False,
        axes=1,
    )

    output = jax.jit(layer)(x)
    layer.eval()
    evaluation_output = jax.jit(layer)(x)
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    variance = jnp.var(x, axis=(0, 2, 3), keepdims=True)
    expected = (x - mean) * jax.lax.rsqrt(variance + 1e-6)

    assert jnp.allclose(output, expected, atol=1e-6)
    assert jnp.allclose(evaluation_output, expected, atol=1e-6)


def test_batch_norm_rejects_running_state_mutation_during_jit():
    layer = nn.BatchNorm(4)

    with pytest.raises(TypeError, match='cannot be mutated while tracing'):
        jax.jit(layer)(jnp.ones((2, 3, 4)))


def test_normalization_supports_explicit_output_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    modules = (
        nn.LayerNorm(4),
        nn.RMSNorm(4),
        nn.BatchNorm(4, track_running_stats=False),
        nn.GroupNorm(2, 4),
    )
    x = jnp.ones((2, 3, 4))

    for module in modules:
        output = jax.jit(
            lambda value, current=module: current(
                value,
                out_sharding=out_sharding,
            )
        )(x)
        assert output.sharding.is_equivalent_to(out_sharding, output.ndim)


@pytest.mark.parametrize(
    'factory',
    [
        lambda **kwargs: nn.LayerNorm(4, **kwargs),
        lambda **kwargs: nn.RMSNorm(4, **kwargs),
        lambda **kwargs: nn.BatchNorm(4, **kwargs),
        lambda **kwargs: nn.GroupNorm(2, 4, **kwargs),
    ],
)
def test_normalization_parameters_support_logical_sharding_and_metadata(factory):
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ('data', 'model'))
    explicit_spec = P('data')
    logical_spec = P('model')

    with jax.set_mesh(mesh), map_logical_axis_names({'feature': 'model'}):
        layer = factory(
            bias=True,
            axis_names=('feature',),
            partition_spec=explicit_spec,
            scale_metadata={'kind': 'gain'},
            bias_metadata={'kind': 'offset'},
        )

    assert layer.scale.partition_spec == logical_spec
    assert layer.bias.partition_spec == logical_spec
    assert layer.scale.value.sharding.is_equivalent_to(
        NamedSharding(mesh, logical_spec),
        layer.scale.ndim,
    )
    assert layer.bias.value.sharding.is_equivalent_to(
        NamedSharding(mesh, logical_spec),
        layer.bias.ndim,
    )
    assert layer.scale.metadata == {'kind': 'gain'}
    assert layer.bias.metadata == {'kind': 'offset'}


@pytest.mark.parametrize(
    ('factory', 'match'),
    [
        (lambda: nn.LayerNorm(0), 'positive integer'),
        (lambda: nn.RMSNorm((2, 3), axes=-1), 'same number'),
        (lambda: nn.BatchNorm(4, momentum=1.5), 'momentum'),
        (lambda: nn.GroupNorm(3, 8), 'divisible'),
        (lambda: nn.GroupNorm((2, 2), 8), 'same number'),
    ],
)
def test_normalization_modules_validate_configuration(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_normalization_accepts_quantization_for_other_operation_types():
    layer = nn.LayerNorm(4, quant='int8')

    assert isinstance(layer.scale.value, jax.Array)
