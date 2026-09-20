import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from taktiny import nn
from taktiny.takt.optimizer import Optimizer


def assert_tree_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('tx', [
    optax.sgd(0.1),
    optax.adam(0.01),
    optax.chain(optax.clip_by_global_norm(1), optax.adamw(0.01)),
    optax.sgd(optax.exponential_decay(0.1, 2, 0.9), momentum=0.9),
    optax.masked(optax.adam(0.01), {'w': True, 'b': False}),
])
@pytest.mark.parametrize('compiled', [False, True])
def test_optimizer_matches_optax(tx, compiled):
    params = {'w': jnp.array([1., -2.], dtype=jnp.float32),
              'b': jnp.array(0.5, dtype=jnp.float32)}
    optimizer = Optimizer(params, tx)
    reference = params
    state = tx.init(params)
    traces = []

    def step(params, optimizer):
        traces.append(True)
        grads = jax.tree.map(lambda x: 2 * x, params)
        return optimizer.update(params, grads), optimizer

    if compiled:
        step = jax.jit(step)
    for _ in range(5):
        updates, state = tx.update(jax.tree.map(lambda x: 2 * x, reference), state, reference)
        reference = optax.apply_updates(reference, updates)
        params, optimizer = step(params, optimizer)
        assert_tree_close(params, reference)
        assert_tree_close(optimizer.state, state)
    if compiled:
        assert len(traces) == 1


def test_optimizer_extra_args():
    def update(grads, state, params=None, *, scale):
        return jax.tree.map(lambda g: -scale * g, grads), state

    tx = optax.GradientTransformationExtraArgs(lambda _: optax.EmptyState(), update)
    optimizer = Optimizer(jnp.ones(2), tx)

    @jax.jit
    def step(params, optimizer):
        return optimizer.update(params, jnp.ones(2), scale=0.25), optimizer

    params, optimizer = step(jnp.ones(2), optimizer)
    np.testing.assert_allclose(params, 0.75)


def test_optimizer_scan():
    params = jnp.ones(2)
    optimizer = Optimizer(params, optax.adam(0.01))

    def step(carry, _):
        params, optimizer = carry
        return (optimizer.update(params, 2 * params), optimizer), None

    (params, optimizer), _ = jax.jit(
        lambda p, o: jax.lax.scan(step, (p, o), None, length=5)
    )(params, optimizer)
    assert int(optimizer.state[0].count) == 5
    assert jnp.all(params < 1)


def test_optimizer_module_and_roundtrip():
    model = nn.Linear(2, 1, rngs=nn.Rngs(0))
    optimizer = Optimizer(model, optax.adam(0.01))
    leaves, structure = jax.tree.flatten(optimizer)
    assert leaves
    restored = jax.tree.unflatten(structure, leaves)
    assert restored.tx is optimizer.tx
    assert_tree_close(restored.state, optimizer.state)
    original_kernel = model.kernel.value

    @jax.jit
    def step(model, optimizer):
        loss, grad = jax.value_and_grad(lambda m: jnp.mean(m(jnp.ones((3, 2))) ** 2))(model)
        return optimizer.update(model, grad), optimizer, loss

    losses = []
    for _ in range(5):
        model, restored, loss = step(model, restored)
        losses.append(float(loss))
    assert losses[-1] < losses[0]
    assert not np.array_equal(model.kernel.value, original_kernel)
    assert int(restored.state[0].count) == 5
    assert int(optimizer.state[0].count) == 0
