import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taktiny.utils.transforms import scan


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('out_axis', [0, -1])
def test_multiple_carries_and_different_axes(reverse, out_axis):
    def body(hidden, cell, x, context):
        hidden = hidden * 0.5 + x + context
        cell = cell + hidden
        return hidden, cell, hidden + cell

    transformed = scan(body, in_axes=('carry', 'carry', 0, 1),
                       out_axes=('carry', 'carry', out_axis), reverse=reverse, unroll=2)
    xs = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    contexts = xs.T * 2
    initial = (jnp.zeros(3), jnp.ones(3))

    def reference(carry, values):
        hidden, cell, output = body(*carry, *values)
        return (hidden, cell), output

    final, history = jax.lax.scan(reference, initial, (xs, contexts.T), reverse=reverse)
    actual = jax.jit(transformed)(*initial, xs, contexts)
    for result, expected in zip(actual, (*final, jnp.moveaxis(history, 0, out_axis))):
        np.testing.assert_allclose(result, expected)


def test_interleaved_carries_broadcast_and_dropped_output():
    @scan(in_axes=(0, 'carry', None, 'carry'), out_axes=(0, 'carry', None, 'carry'))
    def step(x, total, factor, count, *, offset):
        total = total + x * factor + offset
        return total, total, x, count + 1

    history, total, discarded, count = step(jnp.arange(3), 0, 2, 0, offset=1)
    np.testing.assert_array_equal(history, [1, 4, 9])
    assert int(total) == 9
    assert int(count) == 3
    assert discarded is None


@pytest.mark.parametrize('length', [0, 3])
def test_carry_only_and_zero_length(length):
    @scan(in_axes=('carry', 'carry'), out_axes=('carry', 'carry'), length=length)
    def step(total, count):
        return total + 2, count + 1

    total, count = step(0, 0)
    assert int(total) == length * 2
    assert int(count) == length


def test_pytree_carry_and_input_axes():
    @scan(in_axes=('carry', {'x': -1, 'scale': None}),
          out_axes=('carry', {'values': -1, 'unused': None}))
    def step(state, inputs):
        value = state['value'] + inputs['x'] * inputs['scale']
        return {'value': value}, {'values': value, 'unused': value}

    state, history = step({'value': jnp.zeros(2)}, {'x': jnp.ones((2, 3)), 'scale': 2})
    np.testing.assert_array_equal(state['value'], [6, 6])
    np.testing.assert_array_equal(history['values'], [[2, 4, 6], [2, 4, 6]])
    assert history['unused'] is None


def test_grad_and_vmap():
    @scan(in_axes=('carry', 'carry', 0), out_axes=('carry', 'carry', 0))
    def step(total, count, x):
        return total + x, count + 1, total + x

    def loss(xs):
        return step(0., 0, xs)[0]

    xs = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    np.testing.assert_array_equal(jax.jit(jax.vmap(jax.grad(loss)))(xs), jnp.ones_like(xs))


@pytest.mark.parametrize('in_axes,out_axes', [
    (('carry', 0), ('carry', 'carry')),
    ((0,), ('carry',)),
    (('carry', 0), (0,)),
    ('carry', ('carry',)),
    (('carry',), 'carry'),
])
def test_invalid_carry_specs(in_axes, out_axes):
    with pytest.raises(ValueError, match='carry|sequences'):
        scan(lambda *args: args, in_axes=in_axes, out_axes=out_axes)


def test_invalid_calls_and_carry_invariants():
    decorator = scan(in_axes=('carry', 0), out_axes=('carry', 0))
    with pytest.raises(ValueError, match='positional arguments'):
        decorator(lambda c, x: (c, x))(0)
    with pytest.raises(ValueError, match='tuple of 2'):
        decorator(lambda c, x: c)(0, jnp.ones(3))
    with pytest.raises(TypeError, match='carry'):
        decorator(lambda c, x: (jnp.ones(2), x))(0., jnp.ones(3))
    with pytest.raises(ValueError, match='length'):
        scan(lambda c: (c,), in_axes=('carry',), out_axes=('carry',))(0)
    with pytest.raises(ValueError):
        scan(lambda c, x, y: (c, x + y), in_axes=('carry', 0, 0),
             out_axes=('carry', 0))(0, jnp.ones(2), jnp.ones(3))
