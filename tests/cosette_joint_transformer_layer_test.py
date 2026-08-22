import jax
import jax.numpy as jnp

from taktiny.cosettes import layers as ly
from taktiny import nn
from taktiny.cosettes.transformers.ordinario import JointTransformerLayer
from taktiny.maestro.config import ModelConfig


def _config() -> ModelConfig:
    return ModelConfig(
        num_layers=3,
        num_attention_heads=2,
        attention_head_dim=4,
        caption_projection_dim=8,
        intermediate_size=16,
        dual_attention_layers=(1,),
        norm_eps=1e-6,
        dtype='float32',
    )


def _inputs():
    return (
        jax.random.normal(jax.random.key(1), (2, 4, 8)),
        jax.random.normal(jax.random.key(2), (2, 3, 8)),
        jax.random.normal(jax.random.key(3), (2, 8)),
    )


def test_joint_transformer_cosette_derives_layer_topology_from_config():
    first = JointTransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=0)
    dual = JointTransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=1)
    final = JointTransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=2)

    assert first.hidden_size == 8
    assert first.context_size == 8
    assert not first.dual_attention
    assert dual.dual_attention
    assert isinstance(dual.attn2, ly.AttentionLegacy)
    assert final.context_pre_only
    assert final.ff_context is None


def test_joint_transformer_cosette_runs_and_jits():
    layer = JointTransformerLayer(_config(), rngs=nn.Rngs(0), layer_idx=0)
    context, hidden = jax.jit(layer)(*_inputs())

    assert context.shape == (2, 3, 8)
    assert hidden.shape == (2, 4, 8)
    assert jnp.all(jnp.isfinite(context))
    assert jnp.all(jnp.isfinite(hidden))


def test_joint_transformer_cosette_accepts_component_subclasses():
    class CustomJointAttention(ly.JointAttention):
        pass

    class CustomFeedForward(ly.FeedForward):
        pass

    layer = JointTransformerLayer(
        _config(),
        rngs=nn.Rngs(0),
        layer_idx=0,
        joint_attention=CustomJointAttention,
        mlp=CustomFeedForward,
        context_mlp=CustomFeedForward,
    )

    assert isinstance(layer.attn, CustomJointAttention)
    assert isinstance(layer.ff, CustomFeedForward)
    assert isinstance(layer.ff_context, CustomFeedForward)
