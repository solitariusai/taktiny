import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.layers import (
    Attention,
    ConditionEmbedding,
    FrequencyEmbedding,
    ProjectionEmbedding,
    SinusoidalPositionalEmbedding,
    TokenResampler,
)
from taktiny.utils.typing import ShardMode


class Scale(nn.Module):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, value):
        return value * self.factor


def ones(key, shape, dtype):
    del key
    return jnp.ones(shape, dtype)


def test_sinusoidal_frequency_embedding_matches_positional_embedding():
    options = {
        'max_period': 1_000.0,
        'frequency_shift': 0.5,
        'flip_sin_to_cos': True,
        'scale': 2.0,
        'dtype': jnp.bfloat16,
    }
    frequency = FrequencyEmbedding(7, **options)
    positional = SinusoidalPositionalEmbedding(7, **options)
    values = jnp.asarray([[0.0, 1.0], [2.0, 3.0]])

    actual = jax.jit(frequency)(values)
    expected = positional(values)

    assert actual.shape == (2, 2, 7)
    assert actual.dtype == jnp.bfloat16
    assert jnp.array_equal(actual, expected)


def test_gaussian_frequency_embedding_matches_stored_basis():
    layer = FrequencyEmbedding(
        5,
        kind='gaussian',
        scale=0.5,
        trainable=False,
        axis_names=('frequency',),
        rngs=nn.Rngs(0),
    )
    values = jnp.asarray([0.25, 1.5])
    angles = (
        values[:, None]
        * layer.frequencies.value[None, :]
        * 0.5
        * 2.0
        * math.pi
    )
    expected = jnp.concatenate(
        (
            jnp.sin(angles),
            jnp.cos(angles),
            jnp.zeros((2, 1)),
        ),
        axis=-1,
    )

    actual = jax.jit(layer)(values)

    assert jnp.allclose(actual, expected)
    assert not layer.frequencies.trainable
    assert layer.frequencies.axis_names == ('frequency',)


def test_frequency_embedding_supports_log_input_and_gradients():
    layer = FrequencyEmbedding(
        6,
        kind='gaussian',
        log_input=True,
        trainable=True,
        rngs=nn.Rngs(0),
    )
    values = jnp.asarray([1.0, 2.0, 4.0])

    gradient = jax.jit(jax.grad(lambda x: jnp.sum(layer(x))))(values)

    assert layer.frequencies.trainable
    assert gradient.shape == values.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_projection_embedding_applies_components_in_order():
    first = nn.Linear(
        3,
        4,
        bias=False,
        initializer=ones,
        rngs=nn.Rngs(0),
    )
    second = nn.Linear(
        4,
        2,
        bias=False,
        initializer=ones,
        rngs=nn.Rngs(1),
    )
    norm = nn.LayerNorm(2, elementwise_affine=False)
    layer = ProjectionEmbedding(
        first,
        activation='silu',
        output_projection=second,
        norm=norm,
    )
    x = jnp.asarray([[1.0, -2.0, 3.0]])

    actual = jax.jit(layer)(x)
    expected = norm(second(jax.nn.silu(first(x))))

    assert actual.shape == (1, 2)
    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize('fusion', ['sum', 'concat', 'stack'])
def test_condition_embedding_fuses_named_branches(fusion):
    layer = ConditionEmbedding(
        {'time': Scale(2), 'label': Scale(3)},
        fusion=fusion,
        axis=-1,
    )
    time = jnp.ones((2, 4))
    label = jnp.ones((2, 4)) * 2

    actual = jax.jit(layer)(time=time, label=label)

    if fusion == 'sum':
        expected = time * 2 + label * 3
    elif fusion == 'concat':
        expected = jnp.concatenate((time * 2, label * 3), axis=-1)
    else:
        expected = jnp.stack((time * 2, label * 3), axis=-1)
    assert jnp.array_equal(actual, expected)


def test_condition_embedding_supports_custom_fusion_and_projection():
    projection = nn.Linear(
        4,
        3,
        bias=False,
        initializer=ones,
        rngs=nn.Rngs(0),
    )
    layer = ConditionEmbedding(
        {'left': Scale(2), 'right': Scale(4)},
        fusion=lambda outputs: outputs['right'] - outputs['left'],
        projection=projection,
    )
    conditions = {
        'left': jnp.ones((2, 4)),
        'right': jnp.ones((2, 4)),
    }

    actual = layer(conditions)
    expected = projection(jnp.ones((2, 4)) * 2)

    assert jnp.array_equal(actual, expected)


def test_condition_embedding_rejects_incorrect_condition_names():
    layer = ConditionEmbedding({'time': Scale(1), 'label': Scale(1)})

    with pytest.raises(ValueError, match='missing'):
        layer(time=jnp.ones((2, 4)))
    with pytest.raises(ValueError, match='unexpected'):
        layer(
            time=jnp.ones((2, 4)),
            label=jnp.ones((2, 4)),
            image=jnp.ones((2, 4)),
        )


def _token_resampler(*, residual=False, **kwargs):
    attention = Attention(
        8,
        2,
        4,
        bias=False,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    return TokenResampler(
        3,
        8,
        attention,
        residual=residual,
        rngs=nn.Rngs(1),
        **kwargs,
    )


def test_token_resampler_matches_direct_cross_attention():
    layer = _token_resampler()
    x = jax.random.normal(jax.random.key(2), (2, 5, 8))
    queries = jnp.broadcast_to(layer.queries.value, (2, 3, 8))
    q = layer.attention.q_proj(queries)
    k = layer.attention.k_proj(x)
    v = layer.attention.v_proj(x)
    expected = layer.attention.o_proj(Attention.apply(q, k, v))

    actual = jax.jit(layer)(x)

    assert actual.shape == (2, 3, 8)
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_token_resampler_mask_hides_changed_source_tokens():
    layer = _token_resampler()
    x = jax.random.normal(jax.random.key(2), (2, 5, 8))
    changed = x.at[:, -1].add(1_000)
    mask = jnp.asarray(
        [
            [True, True, True, True, False],
            [True, True, True, True, False],
        ]
    )

    original = layer(x, attention_mask=mask)
    modified = layer(changed, attention_mask=mask)

    assert jnp.allclose(original, modified, rtol=1e-5, atol=1e-5)


def test_token_resampler_supports_unbatched_inputs_and_gradients():
    layer = _token_resampler(residual=True)
    x = jax.random.normal(jax.random.key(2), (5, 8))

    output = layer(x)
    gradient = jax.jit(jax.grad(lambda value: jnp.sum(layer(value))))(x)

    assert output.shape == (3, 8)
    assert gradient.shape == x.shape
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize(
    'make_layer,input_value',
    [
        (
            lambda mode: FrequencyEmbedding(
                8,
                shard_mode=mode,
            ),
            jnp.arange(4),
        ),
        (
            lambda mode: ProjectionEmbedding(
                Scale(2),
                shard_mode=mode,
            ),
            jnp.ones((2, 4)),
        ),
        (
            lambda mode: ConditionEmbedding(
                {'value': Scale(2)},
                shard_mode=mode,
            ),
            {'value': jnp.ones((2, 4))},
        ),
        (
            lambda mode: _token_resampler(shard_mode=mode),
            jnp.ones((2, 5, 8)),
        ),
    ],
)
def test_generic_embedding_explicit_sharding(make_layer, input_value):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = make_layer(ShardMode.EXPLICIT)
    apply = lambda value: layer(value, out_sharding=out_sharding)

    jaxpr = jax.make_jaxpr(apply)(input_value).jaxpr
    output = jax.jit(apply)(input_value)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)
