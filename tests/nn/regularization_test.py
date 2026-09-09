# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for regularization layers (Dropout, FeatureDropout, AlphaDropout, StochasticDepth)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn


def test_dropout_respects_train_and_eval_modes():
    layer = nn.Dropout(0.5, rngs=nn.Rngs(0))
    x = jnp.ones((128,), dtype=jnp.float32)

    training_output = layer(x)
    assert set(np.asarray(jnp.unique(training_output))) <= {0.0, 2.0}

    layer.eval()
    evaluation_output = layer(x)
    assert jnp.array_equal(evaluation_output, x)

    layer.train()
    re_trained_output = layer(x)
    assert set(np.asarray(jnp.unique(re_trained_output))) <= {0.0, 2.0}


def test_dropout_p_boundary_conditions():
    x = jnp.ones((8,), dtype=jnp.float32)

    # p=0 is always identity, even in training mode without RNG
    zero_p = nn.Dropout(0.0)
    assert jnp.array_equal(zero_p(x), x)

    # p=1 is always zeros without RNG
    one_p = nn.Dropout(1.0)
    assert jnp.array_equal(one_p(x), jnp.zeros_like(x))


def test_dropout_requires_rng_for_stochastic_training():
    x = jnp.ones((4,), dtype=jnp.float32)

    # Without explicit or context RNG, training raises an error
    with pytest.raises((ValueError, RuntimeError), match='rng'):
        nn.Dropout(0.5)(x)

    # In eval mode, no RNG is needed
    assert jnp.array_equal(nn.Dropout(0.5).eval()(x), x)


def test_dropout_context_rng_support():
    dropout = nn.Dropout(0.5)
    x = jnp.ones((128,), dtype=jnp.float32)

    with nn.set_context_rng(nn.Rngs(42)):
        out1 = dropout(x)
        out2 = dropout(x)

    assert set(np.asarray(jnp.unique(out1))) <= {0.0, 2.0}
    # Successive calls consume subsequent keys
    assert not jnp.array_equal(out1, out2)


def test_dropout_advances_explicit_rngs_stream():
    layer = nn.Dropout(0.5, rngs=nn.Rngs(0))
    x = jnp.ones((128,), dtype=jnp.float32)

    first = layer(x)
    second = layer(x)

    assert not jnp.array_equal(first, second)


def test_dropout_broadcast_axes_shares_mask_values():
    layer = nn.Dropout(
        0.5,
        broadcast_axes=(1, 2),
        rngs=nn.Rngs(0),
    )
    x = jnp.ones((8, 4, 5, 3), dtype=jnp.float32)

    output = jax.jit(layer)(x)
    # Output along broadcast axes (1, 2) must be identical
    assert jnp.all(output == output[:, :1, :1, :])


@pytest.mark.parametrize(
    ('shape', 'channel_axis', 'batch_axis'),
    [
        ((4, 3, 5, 8), -1, 0),
        ((4, 8, 3, 5), 1, 0),
        ((3, 5, 8), -1, None),
    ],
)
def test_feature_dropout_supports_arbitrary_layouts(
    shape,
    channel_axis,
    batch_axis,
):
    layer = nn.FeatureDropout(
        0.5,
        channel_axis=channel_axis,
        batch_axis=batch_axis,
        rngs=nn.Rngs(1),
    )
    x = jnp.ones(shape, dtype=jnp.float32)

    output = jax.jit(layer)(x)

    canonical_channel = channel_axis % len(shape)
    canonical_batch = None if batch_axis is None else batch_axis % len(shape)
    spatial_axes = tuple(
        axis
        for axis in range(len(shape))
        if axis not in (canonical_channel, canonical_batch)
    )
    for axis in spatial_axes:
        assert jnp.all(output == jnp.expand_dims(jnp.take(output, 0, axis=axis), axis))


def test_alpha_dropout_approximately_preserves_unit_normal_statistics():
    x = jax.random.normal(jax.random.key(0), (200_000,))
    layer = nn.AlphaDropout(0.3, rngs=nn.Rngs(1))

    output = jax.jit(layer)(x)

    assert output.dtype == x.dtype
    assert jnp.isclose(jnp.mean(output), 0.0, atol=1e-2)
    assert jnp.isclose(jnp.var(output), 1.0, atol=2e-2)

    # In eval mode it acts as identity
    layer.eval()
    assert jnp.array_equal(layer(x[:10]), x[:10])


def test_feature_alpha_dropout_shares_spatial_mask():
    layer = nn.FeatureAlphaDropout(0.5, rngs=nn.Rngs(0))
    x = jnp.ones((8, 4, 5, 6), dtype=jnp.float32)

    output = jax.jit(layer)(x)

    assert jnp.all(output == output[:, :1, :1, :])


@pytest.mark.parametrize('mode', ['batch', 'row'])
def test_stochastic_depth_drops_complete_residual_branches(mode):
    layer = nn.StochasticDepth(0.5, mode=mode, rngs=nn.Rngs(0))
    x = jnp.ones((32, 4, 5), dtype=jnp.float32)

    output = jax.jit(layer)(x)

    if mode == 'batch':
        assert jnp.all(output == output.reshape(-1)[0])
    else:
        assert jnp.all(output == output[:, :1, :1])
    assert set(np.asarray(jnp.unique(output))) <= {0.0, 2.0}

    # Eval mode is identity
    layer.eval()
    assert jnp.array_equal(layer(x), x)


def test_dropout_preserves_dtypes_and_has_finite_gradients():
    layer = nn.Dropout(0.25, rngs=nn.Rngs(0))
    x = jnp.ones((64,), dtype=jnp.bfloat16)

    output = layer(x)
    gradient = jax.grad(
        lambda val: jnp.sum(layer(val).astype(jnp.float32))
    )(x)

    assert output.dtype == jnp.bfloat16
    assert gradient.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(gradient))


def test_dropout_explicit_sharding_covers_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    sharding = NamedSharding(mesh, P())
    layer = nn.Dropout(0.5, rngs=nn.Rngs(0))

    output = jax.jit(
        lambda value: layer(
            value,
            out_sharding=sharding,
        )
    )(jnp.ones((16,), dtype=jnp.float32))

    assert output.sharding.is_equivalent_to(sharding, output.ndim)


@pytest.mark.parametrize(
    ('factory', 'error', 'match'),
    [
        (lambda: nn.Dropout(-0.1), ValueError, r'\[0, 1\]'),
        (lambda: nn.Dropout(1.1), ValueError, r'\[0, 1\]'),
        (lambda: nn.AlphaDropout(1.0), ValueError, r'\[0, 1\)'),
        (lambda: nn.StochasticDepth(0.5, 'invalid'), ValueError, 'mode'),
        (lambda: nn.Dropout(0.5, rngs=object()), TypeError, 'Rngs'),
    ],
)
def test_regularization_modules_validate_configuration(factory, error, match):
    with pytest.raises(error, match=match):
        factory()


def test_feature_dropout_validates_axes_and_dropout_validates_dtype():
    with pytest.raises(ValueError, match='must be different'):
        nn.FeatureDropout(0.5, channel_axis=0, batch_axis=0)(
            jnp.ones((2, 3), dtype=jnp.float32),
        )
    with pytest.raises(TypeError, match='floating-point or complex'):
        nn.Dropout(0.5, rngs=nn.Rngs(0))(
            jnp.ones((4,), dtype=jnp.int32),
        )


def test_extra_repr_strings():
    assert 'p=0.5' in nn.Dropout(0.5).extra_repr()
    assert 'channel_axis' in nn.FeatureDropout(0.5, channel_axis=1).extra_repr()
    assert 'p=0.3' in nn.AlphaDropout(0.3).extra_repr()
    assert 'channel_axis' in nn.FeatureAlphaDropout(0.3).extra_repr()
    assert "mode='row'" in nn.StochasticDepth(0.5, mode='row').extra_repr()
