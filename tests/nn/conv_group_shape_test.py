from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taktiny import nn


@pytest.mark.parametrize('cls', [nn.Conv, nn.ConvTranspose])
@pytest.mark.parametrize('groups', [(4, 8), (4, 2), (2, 2), [2, 2]])
@pytest.mark.parametrize('batched', [False, True])
def test_structured_groups_match_independent_convolutions(cls, groups, batched):
    channels = (4, 8)
    output_channels = (8, 8)
    layer = cls(channels, output_channels, 3, groups=groups, padding='SAME', rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 5, *channels) if batched else (5, *channels))
    expected = jnp.zeros((*x.shape[:-2], *output_channels))
    for index in product(*(range(g) for g in groups)):
        ins = tuple(slice(i * (n // g), (i + 1) * (n // g)) for i, n, g in zip(index, channels, groups))
        outs = tuple(slice(i * (n // g), (i + 1) * (n // g)) for i, n, g in zip(index, output_channels, groups))
        small = cls(
            tuple(n // g for n, g in zip(channels, groups)),
            tuple(n // g for n, g in zip(output_channels, groups)),
            3, padding='SAME', rngs=nn.Rngs(2),
        )
        kernel_slice = ((slice(None),) + ins + (slice(None),) * 2 if cls is nn.ConvTranspose
                        else (slice(None),) * 3 + outs)
        small.load_state_dict({'kernel': layer.kernel.value[kernel_slice], 'bias': layer.bias.value[outs]})
        expected = expected.at[(Ellipsis, *outs)].set(small(x[(Ellipsis, *ins)]))
    actual = jax.jit(layer)(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    grad = jax.jit(jax.grad(lambda m: jnp.sum(m(x) ** 2)))(layer)
    assert grad.kernel.shape == layer.kernel.shape
    assert jnp.all(jnp.isfinite(grad.kernel.value))


@pytest.mark.parametrize('cls', [nn.Conv, nn.ConvTranspose])
@pytest.mark.parametrize('groups', [(), (2,), (2, 2, 1), (0, 2), (-1, 2), (True, 2), (3, 2), (2, 3)])
def test_invalid_group_shapes(cls, groups):
    with pytest.raises(ValueError, match='groups|channel'):
        cls((4, 8), (4, 8), 1, groups=groups, rngs=nn.Rngs(0))


@pytest.mark.parametrize('cls', [nn.Conv, nn.ConvTranspose])
def test_groups_validate_output_dimensions(cls):
    with pytest.raises(ValueError, match='out_channels'):
        cls((4, 8), (3, 8), 1, groups=(2, 2), rngs=nn.Rngs(0))


@pytest.mark.parametrize('cls', [nn.Conv, nn.ConvTranspose])
def test_structured_depthwise_2d_strided(cls):
    layer = cls((2, 4), (2, 4), (2, 3), stride=2, padding='SAME',
                groups=(2, 4), bias=False, rngs=nn.Rngs(0))
    x = jnp.zeros((1, 4, 6, 2, 4)).at[..., 1, 2].set(1)
    y = jax.jit(layer)(x)
    spatial = (8, 12) if cls is nn.ConvTranspose else (2, 3)
    assert y.shape == (1, *spatial, 2, 4)
    assert jnp.any(y[..., 1, 2] != 0)
    assert jnp.all(y.at[..., 1, 2].set(0) == 0)


@pytest.mark.parametrize('cls', [nn.Conv, nn.ConvTranspose])
def test_structured_groups_with_quantized_kernel(cls):
    kwargs = dict(groups=(2, 2), padding='SAME', bias=False)
    dense = cls((4, 8), (4, 8), 3, rngs=nn.Rngs(0), **kwargs)
    quantized = cls((4, 8), (4, 8), 3, rngs=nn.Rngs(0), quant='int8', **kwargs)
    x = jax.random.normal(jax.random.key(1), (1, 5, 4, 8))
    np.testing.assert_allclose(jax.jit(quantized)(x), dense(x), atol=0.04, rtol=0.04)
