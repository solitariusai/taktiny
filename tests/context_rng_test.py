import asyncio

import jax
import jax.numpy as jnp
import pytest

from taktiny import nn


@pytest.fixture(autouse=True)
def isolated_context():
    with nn.set_context_rng(None):
        yield


def test_global_default_and_nested_exception_restoration():
    outer, inner = nn.Rngs(0), nn.Rngs(1)
    nn.set_context_rng(rngs=outer)
    outer_key = outer.key
    with pytest.raises(RuntimeError, match='test error'):
        with nn.set_context_rng(inner) as active:
            assert active is inner and nn.get_context_rng() is inner
            with nn.set_context_rng(nn.Rngs(2)):
                nn.get_context_rng()()
            assert nn.get_context_rng() is inner
            raise RuntimeError('test error')
    assert nn.get_context_rng() is outer
    assert jnp.array_equal(outer.key, outer_key)
    with nn.set_context_rng(None):
        with pytest.raises(ValueError, match='set_context_rng'):
            nn.get_context_rng()
    assert nn.get_context_rng() is outer


@pytest.mark.parametrize('cls', [nn.Dropout, nn.FeatureDropout,
                               nn.AlphaDropout, nn.FeatureAlphaDropout,
                               nn.StochasticDepth])
def test_dropout_variants_resolve_at_call_time_and_advance(cls):
    layer = cls(0.5)
    x = jnp.ones((32, 32))
    with nn.set_context_rng(nn.Rngs(42)):
        first, second = layer(x), layer(x)
    assert not jnp.array_equal(first, second)
    with nn.set_context_rng(nn.Rngs(42)):
        assert jnp.array_equal(first, layer(x))


def test_explicit_rng_wins_and_deterministic_calls_do_not_consume():
    default = nn.Rngs(0)
    x = jnp.ones((128,))
    with nn.set_context_rng(default):
        key = default.key
        nn.Dropout(0.5, rngs=nn.Rngs(1))(x)
        nn.Dropout(0)(x)
        nn.Dropout(1)(x)
        nn.Dropout(0.5).eval()(x)
        nn.AlphaDropout(0.5).eval()(x)
        assert jnp.array_equal(default.key, key)
        nn.Dropout(0.5)(x)
        assert not jnp.array_equal(default.key, key)


def test_missing_invalid_and_initialization_independence():
    with pytest.raises(ValueError, match='rngs is required'):
        nn.Dropout(0.5)(jnp.ones((8,)))
    with pytest.raises(TypeError, match='Rngs'):
        nn.set_context_rng(42)
    stream = nn.Rngs(0)
    with nn.set_context_rng(stream):
        key = stream.key
        nn.Linear(2, 3, rngs=nn.Rngs(1))
        assert jnp.array_equal(key, stream.key)


def test_jit_explicit_state_threading_and_no_tracer_leak():
    @jax.jit
    def step(x, rngs):
        with nn.set_context_rng(rngs):
            y = nn.Dropout(0.5)(x)
        return y, rngs

    with jax.checking_leaks():
        first, rngs = step(jnp.ones((128,)), nn.Rngs(0))
        second, rngs = step(jnp.ones((128,)), rngs)
    assert not jnp.array_equal(first, second)
    with pytest.raises(ValueError):
        nn.get_context_rng()


def test_jit_cannot_capture_or_rebind_outer_mutable_stream():
    stream = nn.Rngs(0)
    key = stream.key
    with nn.set_context_rng(stream):
        with pytest.raises(RuntimeError, match='JAX transformations'):
            jax.jit(lambda x: nn.Dropout(0.5)(x))(jnp.ones((8,)))

        @jax.jit
        def rebound(x):
            with nn.set_context_rng(stream):
                return nn.Dropout(0.5)(x)

        with pytest.raises(RuntimeError, match='JAX transformations'):
            rebound(jnp.ones((8,)))
    assert jnp.array_equal(key, stream.key)


def test_async_scoped_bindings_do_not_replace_each_other():
    async def worker(seed):
        stream = nn.Rngs(seed)
        with nn.set_context_rng(stream):
            await asyncio.sleep(0)
            assert nn.get_context_rng() is stream

    async def run():
        await asyncio.gather(worker(1), worker(2))

    asyncio.run(run())
    with pytest.raises(ValueError):
        nn.get_context_rng()


def test_scan_threads_rng_and_vmap_uses_independent_keys():
    def body(rngs, x):
        with nn.set_context_rng(rngs):
            y = nn.Dropout(0.5)(x)
        return rngs, y

    with jax.checking_leaks():
        _, outputs = jax.jit(lambda rngs, xs: jax.lax.scan(body, rngs, xs))(
            nn.Rngs(0), jnp.ones((3, 128)),
        )
    assert not jnp.array_equal(outputs[0], outputs[1])

    def sample(key, x):
        with nn.set_context_rng(nn.Rngs(key)):
            return nn.Dropout(0.5)(x)

    keys = jax.random.split(jax.random.key(0), 3)
    mapped = jax.jit(jax.vmap(sample))(keys, jnp.ones((3, 128)))
    assert not jnp.array_equal(mapped[0], mapped[1])
    gradient = jax.jit(jax.grad(lambda x, key: sample(key, x).sum()))(
        jnp.ones((128,)), keys[0],
    )
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0)
