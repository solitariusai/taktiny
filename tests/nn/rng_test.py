import jax
import jax.numpy as jnp
import pytest

from taktiny import nn


def test_split_key_returns_tuple_and_advances_stream():
    rngs = nn.Rngs(42)
    expected = jax.random.split(rngs.key, 4)

    keys = rngs.split_key(3)

    assert isinstance(keys, tuple)
    assert len(keys) == 3
    assert jnp.array_equal(rngs.key, expected[0])
    assert all(jnp.array_equal(key, expected[i + 1]) for i, key in enumerate(keys))


def test_split_rngs_returns_independent_streams():
    rngs = nn.Rngs(42)
    expected = jax.random.split(rngs.key, 3)

    streams = rngs.split_rngs(2)

    assert isinstance(streams, tuple)
    assert len(streams) == 2
    assert all(isinstance(stream, nn.Rngs) for stream in streams)
    assert jnp.array_equal(rngs.key, expected[0])
    assert all(jnp.array_equal(stream.key, expected[i + 1]) for i, stream in enumerate(streams))
    first_key = streams[1].key
    streams[0]()
    assert jnp.array_equal(streams[1].key, first_key)


@pytest.mark.parametrize('method', ['split_key', 'split_rngs'])
@pytest.mark.parametrize('count,error', [
    (0, ValueError),
    (-1, ValueError),
    (1.5, TypeError),
    (True, TypeError),
    ('2', TypeError),
])
def test_invalid_split_count_does_not_advance_stream(method, count, error):
    rngs = nn.Rngs(42)
    original_key = rngs.key

    with pytest.raises(error, match='num_splits must be a positive integer'):
        getattr(rngs, method)(count)

    assert jnp.array_equal(rngs.key, original_key)
