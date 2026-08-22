import jax
import jax.numpy as jnp

from taktiny.cosettes import layers as ly
from taktiny import nn
from taktiny.cosettes.transformers.ordinario import JointTransformerLayer
from taktiny.cosettes.transformers.sd import SD3TransformerLayer
from taktiny.maestro.config import ModelConfig


def _config(**overrides) -> ModelConfig:
    values = {
        'num_layers': 3,
        'num_attention_heads': 2,
        'attention_head_dim': 4,
        'caption_projection_dim': 8,
        'intermediate_size': 32,
        'dual_attention_layers': (1,),
        'qk_norm': 'rms_norm',
        'dtype': 'float32',
    }
    values.update(overrides)
    return ModelConfig(**values)


def _inputs():
    return (
        jax.random.normal(jax.random.key(1), (2, 4, 8)),
        jax.random.normal(jax.random.key(2), (2, 3, 8)),
        jax.random.normal(jax.random.key(3), (2, 8)),
    )


def test_sd3_layer_declares_mmdit_components_and_topology():
    regular = SD3TransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=0)
    dual = SD3TransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=1)
    final = SD3TransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=2)

    assert isinstance(regular.norm1, ly.AdaXNorm)
    assert isinstance(regular.norm1_context, ly.AdaXNorm)
    assert isinstance(regular.attn, ly.JointAttention)
    assert isinstance(regular.norm2, nn.LayerNorm)
    assert isinstance(regular.ff, ly.FeedForward)
    assert dual.dual_attention
    assert isinstance(dual.attn2, ly.AttentionLegacy)
    assert final.context_pre_only
    assert final.norm2_context is None
    assert final.ff_context is None


def test_sd3_layer_alias_call_matches_joint_transformer_equations():
    layer = SD3TransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=0)
    x, enc_x, temb = _inputs()

    actual_enc, actual_x = layer(x=x, enc_x=enc_x, temb=temb)
    expected_enc, expected_x = JointTransformerLayer.__call__(
        layer,
        x,
        enc_x,
        temb,
    )

    assert jnp.allclose(actual_x, expected_x, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(actual_enc, expected_enc, rtol=1e-5, atol=1e-5)


def test_sd3_layer_is_jittable_and_uses_approximate_gelu():
    layer = SD3TransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=1)
    enc_x, x = jax.jit(layer)(*_inputs())
    activation_input = jnp.asarray([-2.0, 0.0, 2.0])

    assert enc_x.shape == (2, 3, 8)
    assert x.shape == (2, 4, 8)
    assert jnp.allclose(
        layer.ff.activation(activation_input),
        jax.nn.gelu(activation_input, approximate=True),
    )


def test_sd3_final_layer_returns_only_hidden_stream():
    layer = SD3TransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=2)
    enc_x, x = layer(*_inputs())

    assert enc_x is None
    assert x.shape == (2, 4, 8)
