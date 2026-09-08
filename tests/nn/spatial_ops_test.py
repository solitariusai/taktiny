import doctest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.nn.modules import convolution


@pytest.mark.parametrize('padding', ['SAME', 'SAME_LOWER', ((1, 2), (2, 1))])
def test_fold_is_unfold_adjoint_with_padding(padding):
    x = jnp.arange(60., dtype=jnp.float32).reshape(5, 6, 2)
    unfold = nn.Unfold((2, 2), dilation=(2, 1), padding=padding)
    patches = jnp.ones_like(unfold(x))
    expected, = jax.linear_transpose(unfold, x)(patches)
    actual = nn.Fold((5, 6), (2, 2), dilation=(2, 1), padding=padding)(patches)
    np.testing.assert_array_equal(actual, expected)


def test_maxpool_nan_and_tie_indices():
    x = jnp.array([2., 2., jnp.nan, 1.])[:, None]
    values, indices = nn.MaxPool(2, return_indices=True)(x)
    np.testing.assert_array_equal(indices[:, 0], [0, 2])
    np.testing.assert_array_equal(values, nn.MaxPool(2)(x))


def test_maxunpool_rejects_noninteger_and_drops_invalid_indices():
    pool = nn.MaxUnpool(1)
    x = jnp.array([3., 4., 5.])[:, None]
    with pytest.raises(TypeError, match='integer'):
        pool(x, x)
    actual = pool(x, jnp.array([-1, 1, 20])[:, None], output_size=3)
    np.testing.assert_array_equal(actual[:, 0], [0., 4., 0.])


@pytest.mark.parametrize('cls', [nn.MaxPool, nn.AvgPool])
def test_small_input_requires_padding(cls):
    pool = cls(4)
    padded = cls(4, padding='SAME')
    x = jnp.ones((1, 2))
    with pytest.raises(ValueError):
        pool(x)
    assert padded(x).shape == (1, 2)


@pytest.mark.parametrize('p', [0.5, 1., 2., 3.])
def test_lppool_zero_gradient_is_finite(p):
    pool = nn.LPPool(p, 2)
    gradient = jax.grad(lambda x: jnp.sum(pool(x)))(jnp.zeros((4, 2)))
    np.testing.assert_array_equal(gradient, 0)


@pytest.mark.parametrize('factory', [
    lambda: nn.MaxPool(True),
    lambda: nn.AvgPool(2, divisor_override=True),
    lambda: nn.MaxPool(2, return_indices=1),
    lambda: nn.AdaptiveAvgPool((True,)),
    lambda: nn.LPPool(float('inf'), 2),
    lambda: nn.FractionalMaxPool(2, output_ratio=float('nan')),
    lambda: nn.FractionalMaxPool(2, output_size=1, random_samples=(1.,)),
    lambda: nn.Padding((1, -1)),
    lambda: nn.Padding(True),
])
def test_invalid_configuration(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_fractional_rng_context_and_compiled_state():
    x = jnp.arange(100.)[:, None]
    pool = nn.FractionalMaxPool(2, output_size=1)
    explicit = nn.FractionalMaxPool(2, output_size=1, rngs=nn.Rngs(42))
    with nn.set_context_rng(nn.Rngs(42)):
        for _ in range(3):
            np.testing.assert_array_equal(pool(x), explicit(x))

    @jax.jit
    def apply(layer, inputs):
        return layer, layer(inputs)

    compiled = nn.FractionalMaxPool(2, output_size=1, rngs=nn.Rngs(7))
    reference = nn.FractionalMaxPool(2, output_size=1, rngs=nn.Rngs(7))
    for _ in range(3):
        compiled, output = apply(compiled, x)
        np.testing.assert_array_equal(output, reference(x))


def test_fractional_fixed_samples_nd_indices():
    x = jnp.arange(6 * 6 * 2.).reshape(6, 6, 2)
    pool = nn.FractionalMaxPool((2, 2), output_size=(3, 3),
                               random_samples=(.5, .5), return_indices=True)
    values, indices = jax.jit(pool)(x)
    expected_indices = jnp.array([[7, 9, 11], [19, 21, 23], [31, 33, 35]])
    np.testing.assert_array_equal(indices[..., 0], expected_indices)
    np.testing.assert_array_equal(values, x.reshape(-1, 2)[expected_indices])


def test_fractional_explicit_rng_precedence_and_fixed_samples():
    x = jnp.arange(30.)[:, None]
    context_rng = nn.Rngs(1)
    initial = jax.random.key_data(context_rng.key)
    owned = nn.FractionalMaxPool(2, output_size=1, rngs=nn.Rngs(2))
    reference = nn.FractionalMaxPool(2, output_size=1, rngs=nn.Rngs(2))
    fixed = nn.FractionalMaxPool(2, output_size=1, random_samples=(.5,))
    with nn.set_context_rng(context_rng):
        np.testing.assert_array_equal(owned(x), reference(x))
        np.testing.assert_array_equal(fixed(x), fixed(x))
    np.testing.assert_array_equal(jax.random.key_data(context_rng.key), initial)


def test_adaptive_preserved_dimension_and_shape_repr():
    x = jnp.arange(24.).reshape(4, 3, 2)
    for cls in [nn.AdaptiveMaxPool, nn.AdaptiveAvgPool]:
        assert cls((None, 2))(x).shape == (4, 2, 2)
    assert 'k=2×3' in nn.MaxPool([2, 3]).extra_repr()
    assert 'output_size=*×2' in nn.AdaptiveAvgPool((None, 2)).extra_repr()


def test_lppool_ceil_partial_window():
    x = jnp.array([3., 4., 2.])[:, None]
    np.testing.assert_allclose(nn.LPPool(2., 2, ceil_mode=True)(x)[:, 0], [5., 2.])
    with pytest.raises(ValueError):
        nn.LPPool(2., 4)(x)


@pytest.mark.parametrize('axis_type', [AxisType.Auto, AxisType.Explicit])
@pytest.mark.parametrize('kind', [
    'Unfold', 'Fold', 'MaxPool', 'MaxUnpool', 'AvgPool',
    'FractionalMaxPool', 'LPPool', 'AdaptiveMaxPool', 'AdaptiveAvgPool', 'Padding',
])
def test_spatial_output_sharding(kind, axis_type):
    mesh = Mesh(np.asarray(jax.devices()), ('data',), axis_types=(axis_type,))
    sharding = NamedSharding(mesh, P('data', None, None))
    x = jnp.arange(len(jax.devices()) * 8., dtype=jnp.float32).reshape(-1, 8, 1)
    constructors = {
        'Unfold': lambda: nn.Unfold(2),
        'Fold': lambda: nn.Fold(8, 2),
        'MaxPool': lambda: nn.MaxPool(2, return_indices=True),
        'MaxUnpool': lambda: nn.MaxUnpool(2),
        'AvgPool': lambda: nn.AvgPool(2),
        'FractionalMaxPool': lambda: nn.FractionalMaxPool(
            2, output_size=3, random_samples=(.5,), return_indices=True),
        'LPPool': lambda: nn.LPPool(2., 2),
        'AdaptiveMaxPool': lambda: nn.AdaptiveMaxPool(3, return_indices=True),
        'AdaptiveAvgPool': lambda: nn.AdaptiveAvgPool(3),
        'Padding': lambda: nn.Padding(((0, 0), (1, 1), (0, 0))),
    }
    layer = constructors[kind]()
    args = (x,)
    if kind == 'Fold':
        args = (nn.Unfold(2)(x),)
    elif kind == 'MaxUnpool':
        args = nn.MaxPool(2, return_indices=True)(x)
    expected = layer(*args)
    with jax.set_mesh(mesh):
        actual = layer(*args, out_sharding=sharding)
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, reference)
        assert result.sharding.is_equivalent_to(sharding, result.ndim)


def test_spatial_docstring_examples():
    runner = doctest.DocTestRunner()
    for name in ['Unfold', 'Fold', 'MaxPool', 'MaxUnpool', 'AvgPool',
                 'FractionalMaxPool', 'LPPool', 'AdaptiveMaxPool',
                 'AdaptiveAvgPool', 'Padding']:
        for test in doctest.DocTestFinder().find(getattr(convolution, name), name):
            runner.run(test)
    assert runner.summarize().failed == 0
