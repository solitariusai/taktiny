import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.utils.typing import ShardMode


def test_downsample_uses_scale_factor_as_divisor():
    layer = nn.Downsample(scale_factor=2, method='linear')
    x = jnp.arange(8, dtype=jnp.float32)[:, None]

    output = jax.jit(layer)(x)
    expected = jax.image.resize(
        x,
        (4, 1),
        method='linear',
        antialias=True,
    )

    assert output.shape == (4, 1)
    assert jnp.allclose(output, expected)


def test_downsample_supports_anisotropic_batched_inputs():
    layer = nn.Downsample(
        scale_factor=(2, 3),
        method='nearest',
    )
    x = jnp.arange(2 * 8 * 9 * 3, dtype=jnp.float32).reshape(
        2,
        8,
        9,
        3,
    )

    output = jax.jit(layer)(x)

    assert output.shape == (2, 4, 3, 3)


def test_downsample_supports_exact_and_preserved_sizes():
    layer = nn.Downsample(size=(3, None), method='cubic')
    x = jnp.ones((2, 7, 5, 4), dtype=jnp.float32)

    output = layer(x)

    assert output.shape == (2, 3, 5, 4)


def test_downsample_is_differentiable_and_jittable():
    layer = nn.Downsample(scale_factor=2)
    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4, 1)

    gradient = jax.jit(jax.grad(lambda value: jnp.sum(layer(value))))(x)

    assert gradient.shape == x.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_resampling_applies_explicit_output_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    sharding = NamedSharding(mesh, P())
    layer = nn.Downsample(
        scale_factor=2,
        shard_mode=ShardMode.EXPLICIT,
    )
    x = jnp.ones((8, 2), dtype=jnp.float32)

    output = jax.jit(
        lambda value: layer(value, out_sharding=sharding)
    )(x)

    assert output.sharding.is_equivalent_to(sharding, output.ndim)


def test_downsample_validates_configuration_and_target_size():
    with pytest.raises(ValueError, match='mutually exclusive'):
        nn.Downsample(size=4, scale_factor=2)
    with pytest.raises(ValueError, match='greater than or equal to 1'):
        nn.Downsample(scale_factor=0.5)
    with pytest.raises(ValueError, match='cannot exceed'):
        nn.Downsample(size=9)(jnp.ones((8, 2)))
    with pytest.raises(ValueError, match='finite and positive'):
        nn.Downsample(scale_factor=float('inf'))


def test_upsample_retains_existing_scale_semantics():
    layer = nn.Upsample(scale_factor=2, method='nearest')
    x = jnp.asarray([[1.0], [2.0]])

    output = jax.jit(layer)(x)

    assert jnp.array_equal(
        output[:, 0],
        jnp.asarray([1.0, 1.0, 2.0, 2.0]),
    )
