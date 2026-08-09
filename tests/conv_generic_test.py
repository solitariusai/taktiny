import math

import jax
import jax.numpy as jnp

from taktiny import nn


def ones(key, shape, dtype):
    del key
    return jnp.ones(shape, dtype=dtype)


def ascending(key, shape, dtype):
    del key
    return jnp.arange(1, math.prod(shape) + 1, dtype=dtype).reshape(shape)


def test_conv_transpose_1d():
    layer = nn.ConvTranspose(
        1,
        1,
        kernel_size=3,
        stride=2,
        bias=False,
        initializer=ones,
        rngs=nn.Rngs(0),
    )

    output = layer(jnp.asarray([[1.0], [2.0]]))

    assert jnp.array_equal(
        output[:, 0],
        jnp.asarray([1.0, 1.0, 3.0, 2.0, 2.0]),
    )


def test_conv_transpose_uses_cross_correlation_adjoint_orientation():
    layer = nn.ConvTranspose(
        1,
        1,
        kernel_size=3,
        bias=False,
        initializer=ascending,
        rngs=nn.Rngs(0),
    )

    output = layer(jnp.asarray([[1.0], [2.0]]))

    assert jnp.array_equal(output[:, 0], jnp.asarray([1.0, 4.0, 7.0, 6.0]))


def test_conv_transpose_groups_and_output_padding():
    layer = nn.ConvTranspose(
        2,
        2,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        groups=2,
        bias=False,
        initializer=ones,
        rngs=nn.Rngs(0),
    )

    output = layer(jnp.ones((1, 2, 2)))

    assert output.shape == (1, 4, 2)
    assert jnp.array_equal(output[..., 0], output[..., 1])


def test_unfold_and_fold_are_overlap_add_pair():
    unfold = nn.Unfold(kernel_size=2, stride=1)
    fold = nn.Fold(output_size=4, kernel_size=2, stride=1)
    x = jnp.asarray([[1.0], [2.0], [3.0], [4.0]])

    patches = unfold(x)
    output = fold(patches)

    assert jnp.array_equal(
        patches,
        jnp.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
    )
    assert jnp.array_equal(
        output[:, 0],
        jnp.asarray([1.0, 4.0, 6.0, 4.0]),
    )


def test_max_pool_indices_feed_max_unpool():
    pool = nn.MaxPool(kernel_size=2, stride=2, return_indices=True)
    unpool = nn.MaxUnpool(kernel_size=2, stride=2)
    x = jnp.asarray([[1.0], [3.0], [2.0], [4.0]])

    values, indices = pool(x)
    output = unpool(values, indices)

    assert jnp.array_equal(values[:, 0], jnp.asarray([3.0, 4.0]))
    assert jnp.array_equal(indices[:, 0], jnp.asarray([1, 3]))
    assert jnp.array_equal(output[:, 0], jnp.asarray([0.0, 3.0, 0.0, 4.0]))


def test_average_pool_padding_divisors():
    x = jnp.asarray([[2.0], [4.0]])
    include = nn.AvgPool(
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=True,
    )
    exclude = nn.AvgPool(
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False,
    )

    assert jnp.allclose(include(x)[:, 0], jnp.asarray([2.0, 2.0]))
    assert jnp.allclose(exclude(x)[:, 0], jnp.asarray([3.0, 3.0]))


def test_average_pool_ceil_extension_is_not_counted_as_padding():
    pool = nn.AvgPool(
        kernel_size=3,
        stride=2,
        padding=1,
        ceil_mode=True,
        count_include_pad=True,
    )
    x = jnp.asarray([[1.0], [2.0], [3.0], [4.0]])

    output = pool(x)

    assert jnp.allclose(output[:, 0], jnp.asarray([1.0, 3.0, 2.0]))


def test_lp_pool():
    pool = nn.LPPool(norm_type=2, kernel_size=2, stride=2)
    x = jnp.asarray([[3.0], [4.0], [5.0], [12.0]])

    assert jnp.allclose(pool(x)[:, 0], jnp.asarray([5.0, 13.0]))


def test_fractional_max_pool_is_reproducible():
    pool = nn.FractionalMaxPool(
        kernel_size=2,
        output_size=3,
        return_indices=True,
        random_samples=(0.5,),
    )
    x = jnp.arange(1, 7, dtype=jnp.float32)[:, None]

    values, indices = pool(x)

    assert jnp.array_equal(values[:, 0], jnp.asarray([2.0, 4.0, 6.0]))
    assert jnp.array_equal(indices[:, 0], jnp.asarray([1, 3, 5]))


def test_adaptive_pooling_values_and_indices():
    x = jnp.arange(1, 7, dtype=jnp.float32)[:, None]
    max_pool = nn.AdaptiveMaxPool(3, return_indices=True)
    avg_pool = nn.AdaptiveAvgPool(3)

    maxima, indices = max_pool(x)
    averages = avg_pool(x)

    assert jnp.array_equal(maxima[:, 0], jnp.asarray([2.0, 4.0, 6.0]))
    assert jnp.array_equal(indices[:, 0], jnp.asarray([1, 3, 5]))
    assert jnp.allclose(averages[:, 0], jnp.asarray([1.5, 3.5, 5.5]))


def test_padding_modes():
    x = jnp.asarray([1, 2, 3])

    assert jnp.array_equal(
        nn.Padding((1, 2), value=9)(x),
        jnp.asarray([9, 1, 2, 3, 9, 9]),
    )
    assert jnp.array_equal(
        nn.Padding((1, 1), mode='circular')(x),
        jnp.asarray([3, 1, 2, 3, 1]),
    )


def test_upsample_is_rank_generic_and_jittable():
    layer = nn.Upsample(scale_factor=2, method='nearest')
    x = jnp.asarray([[1.0], [2.0]])

    output = jax.jit(layer)(x)

    assert jnp.array_equal(
        output[:, 0],
        jnp.asarray([1.0, 1.0, 2.0, 2.0]),
    )
