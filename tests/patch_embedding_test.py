import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.layers import PatchEmbedding
from taktiny.utils.typing import ShardMode


def zeros(key, shape, dtype):
    del key
    return jnp.zeros(shape, dtype)


def test_patch_embedding_projects_and_flattens_2d_images():
    layer = PatchEmbedding(
        3,
        5,
        patch_size=2,
        rngs=nn.Rngs(0),
    )
    x = jnp.ones((2, 4, 6, 3))

    output = layer(x)
    projected = layer.projection(x)

    assert output.shape == (2, 6, 5)
    assert jnp.array_equal(output, projected.reshape(2, 6, 5))


def test_patch_embedding_can_preserve_an_unbatched_spatial_grid():
    layer = PatchEmbedding(
        3,
        5,
        patch_size=(2, 3),
        flatten=False,
        rngs=nn.Rngs(0),
    )

    output = layer(jnp.ones((4, 6, 3)))

    assert output.shape == (2, 2, 5)


def test_patch_embedding_supports_explicit_1d_patches():
    layer = PatchEmbedding(
        3,
        4,
        patch_size=(2,),
        rngs=nn.Rngs(0),
    )

    output = jax.jit(layer)(jnp.ones((2, 6, 3)))

    assert output.shape == (2, 3, 4)


def test_patch_embedding_applies_norm_and_dynamic_positions():
    embedding_dim = 4

    def positions(grid_shape):
        count = math.prod(grid_shape)
        return jnp.arange(count * embedding_dim).reshape(
            count,
            embedding_dim,
        )

    layer = PatchEmbedding(
        1,
        embedding_dim,
        patch_size=2,
        bias=False,
        norm=nn.LayerNorm(
            embedding_dim,
            elementwise_affine=False,
        ),
        position_embedding=positions,
        initializer=zeros,
        rngs=nn.Rngs(0),
    )

    output = layer(jnp.ones((2, 4, 4, 1)))
    expected = positions((2, 2))[None, ...]

    assert output.shape == (2, 4, embedding_dim)
    assert jnp.array_equal(output, jnp.broadcast_to(expected, output.shape))


def test_patch_embedding_rejects_position_shape_mismatch():
    layer = PatchEmbedding(
        1,
        4,
        patch_size=2,
        position_embedding=jnp.zeros((3, 4)),
        rngs=nn.Rngs(0),
    )

    with pytest.raises(ValueError, match='position_embedding must have shape'):
        layer(jnp.ones((1, 4, 4, 1)))


def test_patch_embedding_rejects_norm_that_changes_shape():
    layer = PatchEmbedding(
        1,
        4,
        patch_size=2,
        norm=lambda value: value[..., 0],
        rngs=nn.Rngs(0),
    )

    with pytest.raises(ValueError, match='norm must preserve shape'):
        layer(jnp.ones((1, 4, 4, 1)))


def test_patch_embedding_explicit_sharding_covers_final_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = PatchEmbedding(
        1,
        4,
        patch_size=2,
        shard_mode=ShardMode.EXPLICIT,
        rngs=nn.Rngs(0),
    )
    apply = lambda value: layer(value, out_sharding=out_sharding)
    x = jnp.ones((1, 4, 4, 1))

    jaxpr = jax.make_jaxpr(apply)(x).jaxpr
    output = jax.jit(apply)(x)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)
