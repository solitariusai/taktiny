from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from taktiny import nn
import taktiny.utils.transforms as tt


@pytest.mark.parametrize('decorate', [False, True])
def test_vmap_matches_jax_and_preserves_name(decorate):
    def add(x, offset):
        return x + offset
    mapped = tt.vmap(in_axes=(0, None))(add) if decorate else tt.vmap(add, (0, None))
    x = jnp.arange(12).reshape(3, 4)
    assert jnp.array_equal(mapped(x, 2), jax.vmap(add, (0, None))(x, 2))
    assert mapped.__name__ == 'add'


def test_vmap_kwargs_and_unmapped_outputs_follow_jax():
    @tt.vmap(out_axes=(0, None))
    def function(x, *, offsets):
        return x + offsets, 7
    ys, constant = function(jnp.arange(3), offsets=jnp.arange(3))
    assert jnp.array_equal(ys, jnp.array([0, 2, 4]))
    assert constant == 7


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('in_axis,out_axis', [(0, 0), (1, 1), (-1, -1)])
def test_scan_axes_match_explicit_jax_moveaxis(reverse, in_axis, out_axis):
    def body(carry, x, factor, *, bias):
        carry = carry + x * factor + bias
        return carry, carry
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    moved = jnp.moveaxis(x, in_axis, 0)
    initial = jnp.zeros(moved.shape[1:])
    expected_carry, expected = jax.lax.scan(lambda c, v: body(c, v, 2, bias=1),
                                          initial, moved, reverse=reverse)
    scanned = tt.scan(body, in_axes=in_axis, out_axes=out_axis, reverse=reverse)
    carry, outputs = jax.jit(scanned)(initial, x, 2, bias=1)
    assert jnp.array_equal(carry, expected_carry)
    assert jnp.array_equal(outputs, jnp.moveaxis(expected, 0, out_axis))


def test_scan_pytree_prefixes_broadcast_subtrees_and_drop_outputs():
    @tt.scan(in_axes={'data': 1, 'config': None},
             out_axes={'history': -1, 'discard': None})
    def body(carry, x):
        a, b = x['data']
        carry = carry + a + b + x['config']['bias']
        return carry, {'history': (carry, carry * 2), 'discard': jnp.ones(10)}
    xs = {'data': (jnp.ones((2, 3)), jnp.ones((2, 3))), 'config': {'bias': 1}}
    carry, ys = body(jnp.zeros(2), xs)
    assert jnp.array_equal(carry, jnp.full(2, 9))
    assert ys['history'][0].shape == (2, 3)
    assert ys['discard'] is None


def test_scan_namedtuple_inputs_and_outputs():
    Pair = namedtuple('Pair', ['a', 'b'])
    @tt.scan(in_axes=Pair(0, None), out_axes=Pair(-1, 0))
    def body(carry, x):
        return carry + x.a, Pair(x.a + x.b, x.a)
    carry, ys = body(0, Pair(jnp.arange(3), 10))
    assert carry == 3
    assert isinstance(ys, Pair)
    assert ys.a.tolist() == [10, 11, 12]


@pytest.mark.parametrize('length', [0, 3])
def test_scan_no_mapped_inputs_and_discard_all_outputs(length):
    @tt.scan(in_axes=None, out_axes=None, length=length)
    def body(carry, increment):
        return carry + increment, {'unused': jnp.ones(4)}
    carry, ys = body(0, 2)
    assert carry == length * 2 and ys is None
    carry, ys = tt.scan(lambda c, _: (c + 1, None), length=length)(0)
    assert carry == length and ys is None


def test_scan_zero_length_with_mapped_inputs():
    carry, ys = tt.scan(lambda c, x: (c, x), in_axes=1, out_axes=-1)(
        7, jnp.zeros((2, 0)),
    )
    assert carry == 7 and ys.shape == (2, 0)


def test_scan_jit_gradient_and_explicit_rng_carry():
    def objective(x):
        return tt.scan(lambda c, y: (c + y, c + y))(0., x)[0]
    assert jnp.array_equal(jax.jit(jax.grad(objective))(jnp.ones(3)), jnp.ones(3))
    @tt.scan(length=3)
    def random_step(key, _):
        key, sample_key = jax.random.split(key)
        return key, jax.random.normal(sample_key, (4,))
    with jax.checking_leaks():
        key, first = jax.jit(random_step)(jax.random.key(0))
        _, second = jax.jit(random_step)(key)
    assert not jnp.array_equal(first, second)


@pytest.mark.parametrize('factory', [tt.vmap, tt.scan])
def test_transforms_reject_noncallables(factory):
    with pytest.raises(TypeError, match='callable'):
        factory(1)
    with pytest.raises(TypeError, match='callable'):
        factory()(1)


def test_scan_rejects_invalid_axes_lengths_and_prefixes():
    body = lambda c, x: (c, x)
    with pytest.raises(ValueError, match='length'):
        tt.scan(body, in_axes=None)(0, 1)
    with pytest.raises((ValueError, IndexError)):
        tt.scan(body, in_axes=2)(0, jnp.ones((2, 3)))
    with pytest.raises(TypeError, match='integers'):
        tt.scan(body, in_axes=True)(0, jnp.ones(3))
    with pytest.raises(ValueError, match='prefix'):
        tt.scan(body, in_axes={'missing': 0})(0, {'data': jnp.ones(3)})
    with pytest.raises(ValueError):
        tt.scan(body)(0, (jnp.ones(2), jnp.ones(3)))
    with pytest.raises(ValueError):
        tt.scan(body, length=4)(0, jnp.ones(3))
    with pytest.raises(ValueError, match='prefix'):
        tt.scan(body, out_axes={'missing': 0})(0, jnp.ones(3))


def test_generic_vmap_does_not_rewrite_module_metadata():
    @tt.vmap
    def make_module(value):
        class Holder(nn.Module):
            def __init__(self, value):
                self.parameter = nn.Parameter(value, axis_names=('feature',),
                                              partition_spec=P(None))
        return Holder(value)
    result = make_module(jnp.ones((3, 2)))
    assert result.parameter.shape == (3, 2)
    assert result.parameter.axis_names == ('feature',)
    assert result.parameter.partition_spec == P(None)


@jax.tree_util.register_pytree_node_class
class Box:
    def __init__(self, value):
        self.value = value
    def tree_flatten(self):
        return (self.value,), None
    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(children[0])


def test_scan_accepts_custom_pytree_state_inputs_and_outputs():
    @tt.scan(in_axes=1, out_axes=-1)
    def body(carry, x):
        next_carry = Box(carry.value + x.value)
        return next_carry, Box(next_carry.value)
    carry, outputs = jax.jit(body)(Box(jnp.zeros(2)), Box(jnp.ones((2, 3))))
    assert jnp.array_equal(carry.value, jnp.full(2, 3))
    assert outputs.value.shape == (2, 3)


def test_generic_scan_does_not_rewrite_module_metadata():
    class Holder(nn.Module):
        def __init__(self, value):
            self.parameter = nn.Parameter(value, axis_names=('feature',),
                                          partition_spec=P(None))
    @tt.scan
    def body(carry, x):
        return carry, Holder(x)
    _, result = body(0, jnp.ones((3, 2)))
    assert result.parameter.shape == (3, 2)
    assert result.parameter.axis_names == ('feature',)
    assert result.parameter.partition_spec == P(None)
