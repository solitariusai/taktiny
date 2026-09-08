import doctest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.nn import flatten


def test_flatten_preserves_dimensions_outside_selected_range():
    x = jnp.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    layer = nn.Flatten(start_axis=1, end_axis=-2)

    output = layer(x)

    assert output.shape == (2, 12, 5)
    assert jnp.array_equal(output, x.reshape(2, 12, 5))


def test_flatten_default_preserves_batch_dimension():
    output = nn.Flatten()(jnp.ones((2, 3, 4)))

    assert output.shape == (2, 12)


def test_flatten_supports_scalar_input():
    output = nn.Flatten(start_axis=0)(jnp.asarray(3.0))

    assert output.shape == (1,)
    assert output[0] == 3.0


def test_flatten_validates_dimension_range_and_order():
    x = jnp.ones((2, 3, 4))

    with pytest.raises(ValueError, match='start_axis=3 is out of range'):
        nn.Flatten(start_axis=3)(x)
    with pytest.raises(ValueError, match='before or equal'):
        nn.Flatten(start_axis=2, end_axis=1)(x)


def test_unflatten_expands_selected_dimension():
    x = jnp.arange(2 * 12 * 5).reshape(2, 12, 5)
    layer = nn.Unflatten(axis=1, unflattened_size=(3, 4))

    output = layer(x)

    assert output.shape == (2, 3, 4, 5)
    assert jnp.array_equal(output, x.reshape(2, 3, 4, 5))


def test_unflatten_infers_one_dimension():
    output = nn.Unflatten(-1, (2, -1))(jnp.ones((3, 12)))

    assert output.shape == (3, 2, 6)


def test_unflatten_validates_requested_shape():
    with pytest.raises(ValueError, match='cannot be unflattened'):
        nn.Unflatten(1, (2, 5))(jnp.ones((3, 12)))
    with pytest.raises(ValueError, match='only one'):
        nn.Unflatten(1, (-1, -1))
    with pytest.raises(ValueError, match='scalar'):
        nn.Unflatten(0, (1, 1))(jnp.asarray(3.0))


def test_flatten_and_unflatten_are_jittable():
    flatten = nn.Flatten(start_axis=1)
    unflatten = nn.Unflatten(axis=1, unflattened_size=(3, 4))
    transform = jax.jit(lambda value: unflatten(flatten(value)))
    x = jnp.arange(24).reshape(2, 3, 4)

    output = transform(x)

    assert output.shape == x.shape
    assert jnp.array_equal(output, x)


@pytest.mark.parametrize('factory', [
    lambda: nn.Flatten(True),
    lambda: nn.Flatten(end_axis=1.5),
    lambda: nn.Unflatten(False, 2),
    lambda: nn.Unflatten(0, True),
    lambda: nn.Unflatten(0, [2, False]),
    lambda: nn.Unflatten(0, '23'),
    lambda: nn.Unflatten(0, [2, 3.]),
])
def test_noninteger_configuration(factory):
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize('sizes', [[], [-2], [-1, -1]])
def test_invalid_unflatten_shape(sizes):
    with pytest.raises(ValueError):
        nn.Unflatten(0, sizes)


def test_zero_dimensions_and_inference():
    assert nn.Flatten()(jnp.empty((2, 0, 3))).shape == (2, 0)
    assert nn.Unflatten(1, (2, -1))(jnp.empty((3, 0))).shape == (3, 2, 0)
    assert nn.Unflatten(1, (0, 2))(jnp.empty((3, 0))).shape == (3, 0, 2)
    with pytest.raises(ValueError, match='known dimensions have size zero'):
        nn.Unflatten(0, (0, -1))(jnp.empty((0,)))
    with pytest.raises(ValueError, match='cannot be unflattened'):
        nn.Unflatten(1, (2, 4))(jnp.empty((0, 6)))
    with pytest.raises(ValueError, match='cannot be unflattened'):
        nn.Unflatten(0, (2, -1))(jnp.ones((5,)))


def test_axis_bounds_and_single_axis_identity():
    x = jnp.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(nn.Flatten(-1)(x), x)
    np.testing.assert_array_equal(nn.Unflatten(-1, 3)(x), x)
    for layer in [nn.Flatten(), nn.Unflatten(1, 1)]:
        with pytest.raises(ValueError, match='out of range'):
            layer(jnp.ones((3,)))
    with pytest.raises(ValueError, match='out of range'):
        nn.Flatten(0, -3)(x)


def test_unflatten_copies_shape_and_formats_repr():
    sizes = [2, 3]
    layer = nn.Unflatten(-1, sizes)
    sizes[0] = 7
    assert layer.unflattened_size == (2, 3)
    assert layer.extra_repr() == 'axis=-1, unflattened_size=2×3'


def test_transforms_and_dtype_preservation():
    layer = nn.Flatten(0)
    x = jnp.arange(24., dtype=jnp.float32).reshape(2, 3, 4)
    np.testing.assert_array_equal(jax.jit(jax.vmap(layer))(x), x.reshape(2, 12))
    inverse = nn.Unflatten(0, (2, 3, 4))
    gradient = jax.jit(jax.grad(lambda v: jnp.sum(inverse(layer(v)) ** 2)))(x)
    np.testing.assert_array_equal(gradient, 2 * x)
    for dtype in [jnp.int32, jnp.bool_, jnp.complex64]:
        assert inverse(layer(x.astype(dtype))).dtype == dtype


@pytest.mark.parametrize('axis_type', [AxisType.Auto, AxisType.Explicit])
@pytest.mark.parametrize('compiled', [False, True])
def test_output_sharding(axis_type, compiled):
    count = len(jax.devices())
    mesh = Mesh(np.asarray(jax.devices()), ('data',), axis_types=(axis_type,))
    flat_layout = NamedSharding(mesh, P('data'))
    expanded_layout = NamedSharding(mesh, P(None, 'data'))
    replicated_layout = NamedSharding(mesh, P(None, None))
    x = jnp.arange(count * 4.)
    with jax.set_mesh(mesh):
        x = jax.device_put(x, flat_layout)
        expand = lambda v: nn.Unflatten(0, (2, -1))(v, out_sharding=expanded_layout)
        expanded = (jax.jit(expand) if compiled else expand)(x)
        collapse = lambda v: nn.Flatten(0)(v, out_sharding=flat_layout)
        collapsed = (jax.jit(collapse) if compiled else collapse)(expanded)
        replicate = lambda v: nn.Unflatten(0, (2, -1))(v, out_sharding=replicated_layout)
        replicated = (jax.jit(replicate) if compiled else replicate)(x)
    np.testing.assert_array_equal(expanded, np.asarray(x).reshape(2, -1))
    np.testing.assert_array_equal(collapsed, x)
    np.testing.assert_array_equal(replicated, expanded)
    assert expanded.sharding.is_equivalent_to(expanded_layout, 2)
    assert collapsed.sharding.is_equivalent_to(flat_layout, 1)
    assert replicated.sharding.is_equivalent_to(replicated_layout, 2)


def test_docstring_examples():
    assert doctest.testmod(flatten).failed == 0
