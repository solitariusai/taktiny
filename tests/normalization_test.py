import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.cosettes.layers import (
    AdaXNorm,
    SpatialNorm,
)
from taktiny.utils.typing import ShardMode


class CenterSpatialAxes(nn.Module):
    def __call__(self, value):
        return value - jnp.mean(value, axis=(1, 2), keepdims=True)


def test_adaptive_norm_supports_multiple_axes_and_structured_projection():
    layer = AdaXNorm(
        5,
        (2, 3),
        norm='layernorm',
        axes=(-2, -1),
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (4, 6, 8))
    conditioning = jnp.ones((4, 5))

    normalized, modulation = jax.jit(layer)(x, conditioning)

    assert normalized.shape == x.shape
    assert modulation.shape == (4, 2, 3)
    assert jnp.allclose(jnp.mean(normalized, axis=(-2, -1)), 0, atol=1e-6)
    assert jnp.allclose(jnp.var(normalized, axis=(-2, -1)), 1, atol=2e-5)


def test_adaptive_norm_accepts_a_custom_normalizer():
    layer = AdaXNorm(
        3,
        4,
        norm=CenterSpatialAxes(),
        activation=None,
        rngs=nn.Rngs(0),
    )
    x = jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape(2, 3, 4, 5)

    normalized, modulation = jax.jit(layer)(x, jnp.ones((2, 3)))

    assert normalized.shape == x.shape
    assert modulation.shape == (2, 4)
    assert jnp.allclose(jnp.mean(normalized, axis=(1, 2)), 0, atol=1e-6)


@pytest.mark.parametrize(
    'normalizer',
    [
        nn.LayerNorm(8),
        nn.RMSNorm(8),
        nn.GroupNorm(4, 8),
        nn.BatchNorm(8, track_running_stats=False),
    ],
)
def test_adaptive_norm_accepts_configured_normalization_modules(normalizer):
    layer = AdaXNorm(
        3,
        4,
        norm=normalizer,
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 5, 8))

    normalized, modulation = jax.jit(layer)(x, jnp.ones((2, 3)))

    assert normalized.shape == x.shape
    assert modulation.shape == (2, 4)
    assert jnp.all(jnp.isfinite(normalized))


def test_adaptive_norm_chunks_returns_equal_chunks():
    layer = AdaXNorm(
        5,
        28,
        norm='rmsnorm',
        rngs=nn.Rngs(0),
    )

    normalized, modulation = layer(
        jnp.ones((2, 3, 7)),
        jnp.ones((2, 5)),
    )
    chunks = tuple(jnp.split(modulation, 4, axis=-1))

    assert normalized.shape == (2, 3, 7)
    assert len(chunks) == 4
    assert all(chunk.shape == (2, 7) for chunk in chunks)


@pytest.mark.parametrize(
    ('norm', 'module_type'),
    [('layernorm', nn.LayerNorm), ('rmsnorm', nn.RMSNorm)],
)
def test_adaptive_norm_uses_parameter_free_nn_normalizer(norm, module_type):
    layer = AdaXNorm(5, 12, norm=norm, rngs=nn.Rngs(0))

    assert isinstance(layer.normalizer, module_type)
    assert layer.normalizer.normalized_shape is None
    assert not hasattr(layer.normalizer, 'weight')


def test_adaptive_norm_zero_init_zeros_modulation_projection():
    layer = AdaXNorm(
        5,
        12,
        zero_init=True,
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 3, 6))
    conditioning = jax.random.normal(jax.random.key(2), (2, 5))

    _, modulation = layer(x, conditioning)

    assert layer.zero_init is True
    assert jnp.all(layer.linear.weight.value == 0)
    assert jnp.all(modulation == 0)


def test_adaptive_norm_zero_init_requires_boolean():
    with pytest.raises(TypeError, match='zero_init must be a bool'):
        AdaXNorm(5, 12, zero_init=1, rngs=nn.Rngs(0))


@pytest.mark.parametrize(
    ('feature_shape', 'conditioning_shape'),
    [
        ((2, 7, 8), (2, 3, 4)),
        ((2, 4, 5, 8), (2, 2, 3, 4)),
        ((1, 3, 4, 5, 8), (1, 2, 2, 3, 4)),
    ],
)
def test_spatial_norm_supports_arbitrary_spatial_rank(
    feature_shape,
    conditioning_shape,
):
    layer = SpatialNorm(
        8,
        4,
        num_groups=4,
        rngs=nn.Rngs(0),
    )
    features = jax.random.normal(jax.random.key(1), feature_shape)
    conditioning = jax.random.normal(
        jax.random.key(2),
        conditioning_shape,
    )

    output = jax.jit(layer)(features, conditioning)
    feature_grads, conditioning_grads = jax.grad(
        lambda f, c: jnp.mean(layer(f, c) ** 2),
        argnums=(0, 1),
    )(features, conditioning)

    assert output.shape == feature_shape
    assert feature_grads.shape == feature_shape
    assert conditioning_grads.shape == conditioning_shape
    assert jnp.all(jnp.isfinite(output))
    assert jnp.all(jnp.isfinite(feature_grads))
    assert jnp.all(jnp.isfinite(conditioning_grads))


def test_spatial_norm_records_parameter_axis_names():
    layer = SpatialNorm(
        8,
        4,
        num_groups=4,
        axis_names=('condition', 'feature'),
        rngs=nn.Rngs(0),
    )

    assert layer.norm_layer.weight.axis_names == ('feature',)
    assert layer.norm_layer.bias.axis_names == ('feature',)
    assert layer.scale.weight.axis_names == ('condition', 'feature')
    assert layer.scale.bias.axis_names == ('feature',)
    assert layer.shift.weight.axis_names == ('condition', 'feature')


def test_adaptive_and_spatial_norm_support_explicit_output_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    sharding = NamedSharding(mesh, P())
    adaptive = AdaXNorm(
        4,
        8,
        norm='layernorm',
        shard_mode=ShardMode.EXPLICIT,
        rngs=nn.Rngs(0),
    )
    spatial = SpatialNorm(
        8,
        4,
        num_groups=4,
        shard_mode=ShardMode.EXPLICIT,
        rngs=nn.Rngs(1),
    )

    normalized, modulation = jax.jit(
        lambda x, c: adaptive(
            x,
            c,
            out_sharding=sharding,
            modulation_sharding=sharding,
        )
    )(jnp.ones((2, 3, 8)), jnp.ones((2, 4)))
    output = jax.jit(
        lambda x, c: spatial(
            x,
            c,
            out_sharding=sharding,
            modulation_sharding=sharding,
        )
    )(jnp.ones((2, 3, 8)), jnp.ones((2, 2, 4)))

    assert normalized.sharding.is_equivalent_to(sharding, normalized.ndim)
    assert modulation.sharding.is_equivalent_to(sharding, modulation.ndim)
    assert output.sharding.is_equivalent_to(sharding, output.ndim)


@pytest.mark.parametrize(
    ('factory', 'match'),
    [
        (lambda: AdaXNorm(4, 8, norm='unknown', rngs=nn.Rngs(0)),
         'unsupported norm'),
        (lambda: AdaXNorm(4, 0, rngs=nn.Rngs(0)), 'out_dim'),
        (lambda: SpatialNorm(10, 4, num_groups=4, rngs=nn.Rngs(0)),
         'divisible'),
    ],
)
def test_normalization_modules_validate_configuration(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()
