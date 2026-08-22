import inspect

import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.cosettes.layers import FeedForward


def test_feed_forward_matches_direct_projection_path():
    layer = FeedForward(
        8,
        16,
        activation='gelu',
        dropout=0.0,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 3, 8))

    actual = jax.jit(layer)(x)
    expected = layer.output(layer.activation(layer.input(x)))

    assert actual.shape == x.shape
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_feed_forward_dropout_owns_rng_stream():
    layer = FeedForward(
        8,
        16,
        dropout=0.5,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    x = jnp.ones((8, 4, 8))

    first = layer(x)
    second = layer(x)
    layer.eval()
    evaluation = layer(x)

    assert 'key' not in inspect.signature(layer.__call__).parameters
    assert 'training' not in inspect.signature(layer.__call__).parameters
    assert not jnp.array_equal(first, second)
    assert jnp.all(jnp.isfinite(evaluation))
