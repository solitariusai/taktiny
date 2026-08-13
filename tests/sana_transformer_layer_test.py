import jax
import jax.numpy as jnp
import pytest

from taktiny import layers as ly
from taktiny import nn
from taktiny.cosettes.transformers._ordinario import (
    ConditionalTransformerLayer,
)
from taktiny.cosettes.transformers.sana import (
    SanaLinearAttention,
    SanaTransformerLayer,
)
from taktiny.maestro.config import ModelConfig


def _config(**overrides) -> ModelConfig:
    values = {
        'num_attention_heads': 2,
        'attention_head_dim': 4,
        'num_cross_attention_heads': 2,
        'cross_attention_head_dim': 4,
        'cross_attention_dim': 8,
        'mlp_ratio': 2.0,
        'dropout': 0.0,
        'attention_bias': False,
        'norm_elementwise_affine': False,
        'norm_eps': 1e-6,
        'dtype': 'float32',
    }
    values.update(overrides)
    return ModelConfig(**values)


def _inputs():
    return (
        jax.random.normal(jax.random.key(1), (2, 6, 8)),
        jax.random.normal(jax.random.key(2), (2, 3, 8)),
        jax.random.normal(jax.random.key(3), (2, 48)),
    )


def test_sana_layer_declares_conditional_topology():
    layer = SanaTransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=3)

    assert isinstance(layer, ConditionalTransformerLayer)
    assert isinstance(layer.norm1, nn.LayerNorm)
    assert not layer.norm1.elementwise_affine
    assert isinstance(layer.attn1, SanaLinearAttention)
    assert isinstance(layer.attn2, ly.Attention)
    assert isinstance(layer.norm2, nn.LayerNorm)
    assert isinstance(layer.ff, ly.GLUMBConv)
    assert layer.scale_shift_table.shape == (6, 8)
    assert layer.scale_shift_table.axis_names == ('modulation', 'embed')
    assert layer.layer_idx == 3

    assert not layer.attn1.q_proj.has_bias
    assert layer.attn1.o_proj.has_bias
    assert layer.attn2.q_proj.has_bias
    assert layer.attn2.o_proj.has_bias
    assert layer.ff.conv_inverted.has_bias
    assert not layer.ff.conv_point.has_bias


def test_sana_linear_attention_matches_feature_map_equation():
    attention = SanaLinearAttention(
        8,
        2,
        4,
        bias=False,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(4), (2, 5, 8))

    actual, cache = attention(x)
    q = jax.nn.relu(attention.q_proj(x)).astype(jnp.float32)
    k = jax.nn.relu(attention.k_proj(x)).astype(jnp.float32)
    v = attention.v_proj(x).astype(jnp.float32)
    key_value = jnp.einsum('bthv,bthd->bhvd', v, k)
    numerator = jnp.einsum('bhvd,bshd->bshv', key_value, q)
    denominator = jnp.einsum('bthd,bshd->bsh', k, q)[..., None]
    expected = numerator / (denominator + 1e-15)
    expected = attention.o_proj(expected)

    assert cache is None
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_sana_layer_matches_modulation_and_residual_equations():
    layer = SanaTransformerLayer(_config(), rngs=nn.Rngs(0))
    x, enc_x, temb = _inputs()

    actual = layer(x, enc_x, temb, height=2, width=3)
    modulation = (
        layer.scale_shift_table.value[None, :, :]
        + temb.reshape(2, 6, 8)
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        modulation[:, index, :] for index in range(6)
    )
    normalized = layer.norm1(x)
    normalized = normalized * (1.0 + scale_msa[:, None, :])
    normalized = normalized + shift_msa[:, None, :]
    self_attention, _ = layer.attn1(normalized)
    expected = x + gate_msa[:, None, :] * self_attention
    cross_attention, _ = layer.attn2(expected, context=enc_x)
    expected = expected + cross_attention
    normalized = layer.norm2(expected)
    normalized = normalized * (1.0 + scale_mlp[:, None, :])
    normalized = normalized + shift_mlp[:, None, :]
    feed = layer.ff(normalized, height=2, width=3)
    expected = expected + gate_mlp[:, None, :] * feed

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    compiled = jax.jit(
        lambda hidden, context, condition: layer(
            hidden,
            context,
            condition,
            height=2,
            width=3,
        )
    )
    assert jnp.allclose(compiled(x, enc_x, temb), actual)


def test_sana_across_heads_qk_norm_keeps_flat_checkpoint_weights():
    layer = SanaTransformerLayer(
        _config(qk_norm='rms_norm_across_heads'),
        rngs=nn.Rngs(0),
    )

    assert layer.attn1.q_norm.weight.shape == (8,)
    assert layer.attn1.k_norm.weight.shape == (8,)
    assert layer.attn2.q_norm.weight.shape == (8,)
    assert layer.attn2.k_norm.weight.shape == (8,)
    assert jnp.all(jnp.isfinite(layer(*_inputs(), height=2, width=3)))


def test_sana_layer_requires_the_token_grid_shape():
    layer = SanaTransformerLayer(_config(), rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match=r'height \* width'):
        layer(*_inputs(), height=1, width=5)
