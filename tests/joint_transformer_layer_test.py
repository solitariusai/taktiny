import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.layers import JointTransformerLayer
from taktiny.utils.typing import ShardMode


def _layer(**kwargs):
    options = {
        'hidden_size': 8,
        'context_size': 6,
        'num_heads': 2,
        'intermediate_size': 16,
        'context_intermediate_size': 12,
        'conditioning_size': 5,
        'head_dim': 4,
        'dropout': 0.0,
        'dtype': 'float32',
        'rngs': nn.Rngs(0),
    }
    options.update(kwargs)
    return JointTransformerLayer(**options)


def _inputs():
    return (
        jax.random.normal(jax.random.key(1), (2, 3, 8)),
        jax.random.normal(jax.random.key(2), (2, 2, 6)),
        jax.random.normal(jax.random.key(3), (2, 5)),
    )


def _modulate(value, shift, scale):
    return value * (1 + scale[:, None, :]) + shift[:, None, :]


def _manual(layer, hidden, context, conditioning):
    hidden_base, hidden_modulation = layer.norm1(hidden, conditioning)
    hidden_groups = jnp.split(
        hidden_modulation,
        9 if layer.dual_attention else 6,
        axis=-1,
    )
    (
        hidden_shift_attention,
        hidden_scale_attention,
        hidden_gate_attention,
        hidden_shift_feed,
        hidden_scale_feed,
        hidden_gate_feed,
        *hidden_dual_groups,
    ) = hidden_groups

    context_base, context_modulation = layer.norm1_context(
        context,
        conditioning,
    )
    context_groups = jnp.split(
        context_modulation,
        2 if layer.context_pre_only else 6,
        axis=-1,
    )

    hidden_attention_input = _modulate(
        hidden_base,
        hidden_shift_attention,
        hidden_scale_attention,
    )
    if layer.context_pre_only:
        context_scale_attention, context_shift_attention = context_groups
    else:
        (
            context_shift_attention,
            context_scale_attention,
            context_gate_attention,
            context_shift_feed,
            context_scale_feed,
            context_gate_feed,
        ) = context_groups
    context_attention_input = _modulate(
        context_base,
        context_shift_attention,
        context_scale_attention,
    )

    hidden_attention, context_attention = layer.attn(
        hidden_attention_input,
        context_attention_input,
    )
    hidden = hidden + hidden_gate_attention[:, None, :] * hidden_attention

    if layer.dual_attention:
        hidden_shift_dual, hidden_scale_dual, hidden_gate_dual = (
            hidden_dual_groups
        )
        dual_input = _modulate(
            hidden_base,
            hidden_shift_dual,
            hidden_scale_dual,
        )
        dual_attention, _ = layer.attn2(dual_input)
        hidden = hidden + hidden_gate_dual[:, None, :] * dual_attention

    hidden_feed_input = _modulate(
        layer.norm2(hidden),
        hidden_shift_feed,
        hidden_scale_feed,
    )
    hidden = hidden + hidden_gate_feed[:, None, :] * layer.ff(
        hidden_feed_input,
        out_sharding=None,
    )

    if layer.context_pre_only:
        return None, hidden

    context = context + context_gate_attention[:, None, :] * context_attention
    context_feed_input = _modulate(
        layer.norm2_context(context),
        context_shift_feed,
        context_scale_feed,
    )
    context = context + context_gate_feed[:, None, :] * layer.ff_context(
        context_feed_input,
        out_sharding=None,
    )
    return context, hidden


@pytest.mark.parametrize(
    'options',
    [
        {},
        {'context_pre_only': True},
        {'dual_attention': True},
        {'norm': 'rmsnorm'},
    ],
)
def test_joint_transformer_layer_matches_direct_equations(options):
    layer = _layer(**options)
    hidden, context, conditioning = _inputs()

    actual_context, actual_hidden = layer(hidden, context, conditioning)
    expected_context, expected_hidden = _manual(
        layer,
        hidden,
        context,
        conditioning,
    )

    assert jnp.allclose(actual_hidden, expected_hidden, rtol=1e-5, atol=1e-5)
    if expected_context is None:
        assert actual_context is None
    else:
        assert jnp.allclose(
            actual_context,
            expected_context,
            rtol=1e-5,
            atol=1e-5,
        )


def test_joint_transformer_layer_is_jittable_and_differentiable():
    layer = _layer()
    hidden, context, conditioning = _inputs()

    output_context, output_hidden = jax.jit(layer)(
        hidden,
        context,
        conditioning,
    )

    def loss(hidden_value, context_value, conditioning_value):
        output_context, output_hidden = layer(
            hidden_value,
            context_value,
            conditioning_value,
        )
        return jnp.mean(jnp.square(output_hidden)) + jnp.mean(
            jnp.square(output_context)
        )

    gradients = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(
        hidden,
        context,
        conditioning,
    )

    assert output_hidden.shape == hidden.shape
    assert output_context.shape == context.shape
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


def test_joint_mask_hides_changed_context_tokens_from_hidden_stream():
    layer = _layer()
    hidden, context, conditioning = _inputs()
    changed_context = context.at[:, -1, :].add(100)
    total_length = hidden.shape[1] + context.shape[1]
    mask = jnp.ones((total_length, total_length), dtype=jnp.bool_)
    mask = mask.at[:, -1].set(False)

    _, original = layer(
        hidden,
        context,
        conditioning,
        attention_mask=mask,
    )
    _, changed = layer(
        hidden,
        changed_context,
        conditioning,
        attention_mask=mask,
    )

    assert jnp.allclose(original, changed, rtol=1e-5, atol=1e-5)


def test_joint_transformer_layer_dropout_owns_rng_and_uses_module_mode():
    layer = _layer(dropout=0.5)
    inputs = _inputs()

    first = layer(*inputs)
    second = layer(*inputs)
    layer.eval()
    evaluation = layer(*inputs)

    assert 'key' not in inspect.signature(layer.__call__).parameters
    assert 'training' not in inspect.signature(layer.__call__).parameters
    assert not jnp.array_equal(first[0], second[0])
    assert not jnp.array_equal(first[1], second[1])
    assert jnp.all(jnp.isfinite(evaluation[0]))
    assert jnp.all(jnp.isfinite(evaluation[1]))


def test_joint_transformer_layer_applies_each_output_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    sharding = NamedSharding(mesh, P())
    layer = _layer(shard_mode=ShardMode.EXPLICIT)

    context, hidden = jax.jit(
        lambda *values: layer(
            *values,
            out_shardings=(sharding, sharding),
        )
    )(*_inputs())

    assert context.sharding.is_equivalent_to(sharding, context.ndim)
    assert hidden.sharding.is_equivalent_to(sharding, hidden.ndim)


def test_joint_transformer_layer_validates_configuration_and_inputs():
    with pytest.raises(ValueError, match='divisible'):
        _layer(hidden_size=7, head_dim=None)
    with pytest.raises(ValueError, match='positive integer'):
        _layer(context_intermediate_size=0)
    with pytest.raises(ValueError, match="norm must be"):
        _layer(norm='groupnorm')

    layer = _layer()
    hidden, context, conditioning = _inputs()
    with pytest.raises(ValueError, match='share a batch size'):
        layer(hidden, context[:1], conditioning)
    with pytest.raises(ValueError, match='conditioning must have shape'):
        layer(hidden, context, conditioning[:, :-1])
    with pytest.raises(ValueError, match='exactly two values'):
        layer(
            hidden,
            context,
            conditioning,
            out_shardings=(None,),
        )
