import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from taktiny import nn
from taktiny.utils.typing import ShardMode


def test_layer_norm_matches_reference_and_supports_no_affine():
    x = jax.random.normal(jax.random.key(0), (2, 3, 4))
    layer = nn.LayerNorm(4, eps=1e-6, elementwise_affine=False)

    output = jax.jit(layer)(x)
    expected = (x - jnp.mean(x, axis=-1, keepdims=True)) * jax.lax.rsqrt(
        jnp.var(x, axis=-1, keepdims=True) + 1e-6
    )

    assert not hasattr(layer, 'weight')
    assert output.shape == x.shape
    assert jnp.allclose(output, expected, atol=1e-6)


def test_layer_norm_supports_non_trailing_axes_and_parameter_broadcasting():
    x = jax.random.normal(jax.random.key(0), (2, 3, 4))
    layer = nn.LayerNorm(
        (2, 4),
        axes=(0, 2),
        axis_names=('batch_feature', 'feature'),
    )
    layer.weight.value = jnp.arange(1, 9, dtype=jnp.float32).reshape(2, 4)
    layer.bias.value = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

    output = layer(x)
    normalized = (x - jnp.mean(x, axis=(0, 2), keepdims=True)) * (
        jax.lax.rsqrt(jnp.var(x, axis=(0, 2), keepdims=True) + 1e-5)
    )
    expected = (
        normalized * layer.weight.value[:, None, :]
        + layer.bias.value[:, None, :]
    )

    assert layer.weight.axis_names == ('batch_feature', 'feature')
    assert layer.bias.axis_names == ('batch_feature', 'feature')
    assert jnp.allclose(output, expected, atol=1e-6)


def test_layer_norm_preserves_declared_axis_order_when_broadcasting():
    x = jax.random.normal(jax.random.key(0), (2, 3, 4))
    layer = nn.LayerNorm(
        (4, 2),
        axes=(2, 0),
        bias=False,
    )
    layer.weight.value = jnp.arange(1, 9, dtype=jnp.float32).reshape(4, 2)

    output = layer(x)
    normalized = (x - jnp.mean(x, axis=(2, 0), keepdims=True)) * (
        jax.lax.rsqrt(jnp.var(x, axis=(2, 0), keepdims=True) + 1e-5)
    )
    expected = normalized * layer.weight.value.T[:, None, :]

    assert jnp.allclose(output, expected, atol=1e-6)


def test_rms_norm_supports_multiple_dimensions_and_bfloat16_statistics():
    x = jax.random.normal(jax.random.key(0), (2, 3, 4)).astype(jnp.bfloat16)
    layer = nn.RMSNorm((3, 4), epsilon=1e-6, with_scale=False)

    output = jax.jit(layer)(x)
    value = x.astype(jnp.float32)
    expected = value * jax.lax.rsqrt(
        jnp.mean(jnp.square(value), axis=(-2, -1), keepdims=True) + 1e-6
    )

    assert output.dtype == jnp.bfloat16
    assert not hasattr(layer, 'weight')
    assert jnp.allclose(output.astype(jnp.float32), expected, atol=1e-2)


def test_rms_norm_supports_learned_bias_and_parameter_axes():
    x = jax.random.normal(jax.random.key(1), (2, 3, 4))
    layer = nn.RMSNorm(
        4,
        epsilon=1e-6,
        bias=True,
        axis_names=('feature',),
    )
    layer.weight.value = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    layer.bias.value = jnp.asarray([-1.0, 0.0, 1.0, 2.0])

    output = jax.jit(layer)(x)
    expected = x * jax.lax.rsqrt(
        jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6
    )
    expected = expected * layer.weight.value + layer.bias.value

    assert layer.weight.axis_names == ('feature',)
    assert layer.bias.axis_names == ('feature',)
    assert jnp.allclose(output, expected, atol=1e-6)


def _group_norm_reference(
    x,
    *,
    groups,
    channel_axis,
    batch_axis,
    eps,
):
    channel_axis %= x.ndim
    if batch_axis is None:
        permutation = tuple(i for i in range(x.ndim) if i != channel_axis) + (
            channel_axis,
        )
    else:
        batch_axis %= x.ndim
        permutation = (
            batch_axis,
            *(i for i in range(x.ndim) if i not in (batch_axis, channel_axis)),
            channel_axis,
        )
    transposed = jnp.transpose(x, permutation)
    if batch_axis is None:
        transposed = transposed[None]
    grouped = transposed.reshape(
        *transposed.shape[:-1],
        groups,
        transposed.shape[-1] // groups,
    )
    group_axis = grouped.ndim - 2
    reduce_axes = tuple(
        axis for axis in range(1, grouped.ndim) if axis != group_axis
    )
    normalized = (grouped - jnp.mean(grouped, axis=reduce_axes, keepdims=True))
    normalized *= jax.lax.rsqrt(
        jnp.var(grouped, axis=reduce_axes, keepdims=True) + eps
    )
    normalized = normalized.reshape(transposed.shape)
    if batch_axis is None:
        normalized = normalized[0]
    inverse = tuple(sorted(range(x.ndim), key=permutation.__getitem__))
    return jnp.transpose(normalized, inverse)


@pytest.mark.parametrize(
    ('shape', 'channel_axis', 'batch_axis'),
    [
        ((2, 3, 5, 8), -1, 0),
        ((2, 8, 3, 5), 1, 0),
        ((3, 5, 8), -1, None),
    ],
)
def test_group_norm_supports_channel_layouts_and_unbatched_inputs(
    shape,
    channel_axis,
    batch_axis,
):
    x = jax.random.normal(jax.random.key(0), shape)
    layer = nn.GroupNorm(
        4,
        8,
        affine=False,
        channel_axis=channel_axis,
        batch_axis=batch_axis,
    )

    output = jax.jit(layer)(x)
    expected = _group_norm_reference(
        x,
        groups=4,
        channel_axis=channel_axis,
        batch_axis=batch_axis,
        eps=1e-5,
    )

    assert output.shape == x.shape
    assert jnp.allclose(output, expected, atol=1e-6)


def test_group_norm_is_differentiable():
    layer = nn.GroupNorm(2, 4)
    x = jax.random.normal(jax.random.key(0), (2, 3, 4))

    gradient = jax.jit(jax.grad(lambda value: jnp.sum(layer(value) ** 2)))(x)

    assert gradient.shape == x.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_batch_norm_training_statistics_and_running_state():
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    layer = nn.BatchNorm(4, momentum=1.0)

    training_output = layer(x, training=True, update_stats=True)
    mean = jnp.mean(x, axis=(0, 1))
    variance = jnp.var(x, axis=(0, 1))
    expected = (x - mean) * jax.lax.rsqrt(variance + 1e-5)

    assert jnp.allclose(training_output, expected, atol=1e-6)
    assert jnp.array_equal(layer.running_mean.value, mean)
    assert jnp.array_equal(layer.running_var.value, variance)
    assert int(layer.num_batches_tracked.value) == 1
    assert not layer.running_mean.trainable
    assert not layer.running_var.trainable
    layer.eval()
    assert not layer.training
    assert jnp.allclose(layer(x), training_output, atol=1e-6)

    layer.reset_running_stats()
    assert jnp.array_equal(layer.running_mean.value, jnp.zeros((4,)))
    assert jnp.array_equal(layer.running_var.value, jnp.ones((4,)))
    assert int(layer.num_batches_tracked.value) == 0


def test_batch_norm_uses_module_mode_and_allows_an_explicit_override():
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    layer = nn.BatchNorm(4, affine=False)
    expected_training = (x - jnp.mean(x, axis=(0, 1))) * jax.lax.rsqrt(
        jnp.var(x, axis=(0, 1)) + 1e-5
    )

    assert layer.training
    assert jnp.allclose(layer(x), expected_training, atol=1e-6)

    layer.eval()
    expected_eval = x * jax.lax.rsqrt(jnp.asarray(1.0 + 1e-5))
    assert jnp.allclose(layer(x), expected_eval, atol=1e-6)
    assert jnp.allclose(
        layer(x, training=True),
        expected_training,
        atol=1e-6,
    )


def test_batch_norm_supports_channel_first_and_jitted_training():
    x = jax.random.normal(jax.random.key(0), (2, 4, 3, 5))
    layer = nn.BatchNorm(
        4,
        affine=False,
        track_running_stats=False,
        channel_axis=1,
    )

    output = jax.jit(lambda value: layer(value, training=True))(x)
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    variance = jnp.var(x, axis=(0, 2, 3), keepdims=True)
    expected = (x - mean) * jax.lax.rsqrt(variance + 1e-5)

    assert output.shape == x.shape
    assert jnp.allclose(output, expected, atol=1e-6)


def test_batch_norm_rejects_state_mutation_during_jit():
    layer = nn.BatchNorm(4)
    apply = jax.jit(
        lambda value: layer(value, training=True, update_stats=True)
    )

    with pytest.raises(ValueError, match='cannot be mutated while tracing'):
        apply(jnp.ones((2, 3, 4)))


def test_normalization_parameter_axes_and_explicit_output_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    sharding = NamedSharding(mesh, P())
    modules = [
        nn.LayerNorm(
            4,
            axis_names=('feature',),
            shard_mode=ShardMode.EXPLICIT,
        ),
        nn.RMSNorm(
            4,
            axis_names=('feature',),
            shard_mode=ShardMode.EXPLICIT,
        ),
        nn.BatchNorm(
            4,
            axis_names=('feature',),
            shard_mode=ShardMode.EXPLICIT,
        ),
        nn.GroupNorm(
            2,
            4,
            axis_names=('feature',),
            shard_mode=ShardMode.EXPLICIT,
        ),
    ]
    x = jnp.ones((2, 3, 4))

    for module in modules:
        output = jax.jit(
            lambda value, current=module: current(
                value,
                out_sharding=sharding,
            )
        )(x)
        assert module.weight.axis_names == ('feature',)
        assert output.sharding.is_equivalent_to(sharding, output.ndim)


@pytest.mark.parametrize(
    ('factory', 'match'),
    [
        (lambda: nn.LayerNorm(0), 'positive integer'),
        (lambda: nn.RMSNorm((2, 3), axes=-1), 'same number'),
        (lambda: nn.BatchNorm(4, momentum=1.5), 'momentum'),
        (lambda: nn.GroupNorm(3, 8), 'divisible'),
    ],
)
def test_normalization_modules_validate_configuration(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()
