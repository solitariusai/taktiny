import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.layers import SinusoidalPositionalEmbedding
from taktiny.utils.typing import ShardMode


def ascending(key, shape, dtype):
    del key
    return jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)


def test_embedding_gathers_and_records_logical_axes():
    layer = nn.Embedding(
        5,
        3,
        initializer=ascending,
        axis_names=('vocab', 'embed'),
        rngs=nn.Rngs(0),
    )
    indices = jnp.asarray([[0, 2], [4, 1]])

    output = layer(indices)

    assert output.shape == (2, 2, 3)
    assert jnp.array_equal(output, layer.embedding.value[indices])
    assert layer.embedding.axis_names == ('vocab', 'embed')


def test_embedding_validates_logical_axes():
    with pytest.raises(ValueError, match='vocabulary and embedding axes'):
        nn.Embedding(
            5,
            3,
            axis_names=('embed',),
            rngs=nn.Rngs(0),
        )


def test_embedding_explicit_sharding_covers_gathered_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = nn.Embedding(
        5,
        3,
        shard_mode=ShardMode.EXPLICIT,
        rngs=nn.Rngs(0),
    )
    apply = lambda indices: layer(indices, out_sharding=out_sharding)
    indices = jnp.asarray([[0, 2]])

    jaxpr = jax.make_jaxpr(apply)(indices).jaxpr
    output = jax.jit(apply)(indices)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)


def test_sinusoidal_embedding_supports_arbitrary_position_shapes():
    layer = SinusoidalPositionalEmbedding(5)
    positions = jnp.asarray([[0.0, 1.0], [2.0, 3.0]])

    output = jax.jit(layer)(positions)

    assert output.shape == (2, 2, 5)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.array_equal(output[..., -1], jnp.zeros((2, 2)))


@pytest.mark.parametrize('embedding_dim', [1, 2, 3])
def test_sinusoidal_embedding_small_dimensions_are_finite(embedding_dim):
    layer = SinusoidalPositionalEmbedding(embedding_dim)

    output = layer(jnp.asarray(1.0))

    assert output.shape == (embedding_dim,)
    assert jnp.all(jnp.isfinite(output))


def test_sinusoidal_embedding_options_and_dtype():
    positions = jnp.asarray([0.0, 1.0])
    normal = SinusoidalPositionalEmbedding(4)(positions)
    flipped = SinusoidalPositionalEmbedding(
        4,
        flip_sin_to_cos=True,
        scale=2.0,
        dtype=jnp.bfloat16,
    )(positions)

    assert flipped.dtype == jnp.bfloat16
    assert jnp.array_equal(flipped[0], jnp.asarray([1, 1, 0, 0], jnp.bfloat16))
    assert not jnp.allclose(normal[1], flipped[1].astype(jnp.float32))


def test_sinusoidal_embedding_explicit_sharding_covers_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = SinusoidalPositionalEmbedding(
        8,
        shard_mode=ShardMode.EXPLICIT,
    )
    apply = lambda positions: layer(positions, out_sharding=out_sharding)
    positions = jnp.arange(4)

    jaxpr = jax.make_jaxpr(apply)(positions).jaxpr
    output = jax.jit(apply)(positions)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)
