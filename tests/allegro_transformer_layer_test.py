import jax
import jax.numpy as jnp

from taktiny.cosettes import layers as ly
from taktiny import nn
from taktiny.cosettes.transformers.ordinario import (
    ConditionalTransformerLayer,
)
from taktiny.cosettes.transformers.allegro import (
    AllegroTransformerLayer,
)
from taktiny.maestro.config import ModelConfig


def _config(**overrides) -> ModelConfig:
    values = {
        'num_attention_heads': 2,
        'attention_head_dim': 4,
        'cross_attention_dim': 10,
        'dropout': 0.0,
        'attention_bias': True,
        'activation_fn': 'gelu-approximate',
        'norm_elementwise_affine': False,
        'norm_eps': 1e-6,
        'dtype': 'float32',
    }
    values.update(overrides)
    return ModelConfig(**values)


def _inputs():
    return (
        jax.random.normal(jax.random.key(1), (2, 6, 8)),
        jax.random.normal(jax.random.key(2), (2, 3, 10)),
        jax.random.normal(jax.random.key(3), (2, 48)),
    )


def test_allegro_layer_declares_conditional_topology():
    layer = AllegroTransformerLayer(
        _config(),
        rngs=nn.Rngs(0),
        layer_idx=4,
    )

    assert isinstance(layer, ConditionalTransformerLayer)
    assert isinstance(layer.norm1, nn.LayerNorm)
    assert isinstance(layer.attn1, ly.AttentionLegacy)
    assert isinstance(layer.attn2, ly.AttentionLegacy)
    assert layer.norm_cross is None
    assert isinstance(layer.norm2, nn.LayerNorm)
    assert isinstance(layer.ff, ly.FeedForward)
    assert layer.layer_idx == 4
    assert layer.ff.input.weight.shape == (8, 32)
    assert layer.ff.output.weight.shape == (32, 8)

    assert layer.attn1.q_proj.has_bias
    assert layer.attn1.k_proj.has_bias
    assert layer.attn1.v_proj.has_bias
    assert layer.attn2.q_proj.has_bias
    assert layer.attn2.k_proj.has_bias
    assert layer.attn2.v_proj.has_bias


def test_allegro_layer_matches_modulation_and_residual_equations():
    layer = AllegroTransformerLayer(_config(), rngs=nn.Rngs(0))
    x, enc_x, temb = _inputs()

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
    expected = expected + gate_mlp[:, None, :] * layer.ff(normalized)

    actual = layer(x, enc_x, temb)

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    compiled = jax.jit(layer)
    assert jnp.allclose(compiled(x, enc_x, temb), actual)


def test_allegro_layer_supports_self_attention_without_context():
    layer = AllegroTransformerLayer(_config(), rngs=nn.Rngs(0))
    x, _, temb = _inputs()

    output = layer(x, None, temb)

    assert output.shape == x.shape
    assert jnp.all(jnp.isfinite(output))


def test_conditional_layer_supports_tokenwise_modulation():
    layer = AllegroTransformerLayer(_config(), rngs=nn.Rngs(0))
    x, enc_x, temb = _inputs()
    tokenwise_temb = jnp.broadcast_to(
        temb[:, None, :],
        (x.shape[0], x.shape[1], temb.shape[-1]),
    )

    batchwise = layer(x, enc_x, temb)
    tokenwise = layer(x, enc_x, tokenwise_temb)

    assert jnp.allclose(tokenwise, batchwise, rtol=1e-5, atol=1e-5)


def test_allegro_layer_honors_non_affine_norm_configuration():
    layer = AllegroTransformerLayer(
        _config(norm_elementwise_affine=False),
        rngs=nn.Rngs(0),
    )

    assert not layer.norm1.elementwise_affine
    assert not layer.norm2.elementwise_affine
