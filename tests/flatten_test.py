import jax
import jax.numpy as jnp
import pytest

from taktiny import nn


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
