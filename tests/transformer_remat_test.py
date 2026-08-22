import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.cosettes.transformers.ordinario import TransformerCausalLM, TransformerModel
from taktiny.maestro.config import ModelConfig


class RematTestLayer(nn.Module):
    def __init__(self, config, rngs, layer_idx=None, **kwargs):
        del layer_idx, kwargs
        self.proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype='float32',
            rngs=rngs,
        )

    def __call__(self, x, **kwargs):
        return jax.nn.gelu(self.proj(x)), None


class PositionTestLayer(nn.Module):
    def __init__(self, config, rngs, layer_idx=None, **kwargs):
        del config, rngs, layer_idx, kwargs

    def __call__(self, x, position_ids=None, **kwargs):
        del kwargs
        return x + position_ids[..., None], None


class IdentityNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __call__(self, x, **kwargs):
        del kwargs
        return x


class RematTransformerModel(TransformerModel):
    _layer_type = RematTestLayer


class PositionTransformerModel(TransformerModel):
    _layer_type = PositionTestLayer
    _norm = IdentityNorm


class PositionCausalLM(TransformerCausalLM):
    _model_type = PositionTransformerModel


@pytest.mark.parametrize('stack_type', ['list', 'stack'])
def test_transformer_remat_preserves_forward_and_backward(stack_type):
    config = ModelConfig(
        num_hidden_layers=2,
        vocab_size=8,
        hidden_size=4,
        num_attention_heads=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        dtype='float32',
        stack_type=stack_type,
    )
    model = RematTransformerModel(config, rngs=nn.Rngs(0))
    input_ids = jnp.asarray([[1, 2, 3]])

    expected = model(input_ids).x
    model.enable_remat()
    actual = model(input_ids).x

    assert jnp.allclose(actual, expected)

    def loss(candidate):
        return jnp.sum(candidate(input_ids).x)

    gradients = jax.grad(loss)(model)
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree.leaves(gradients)
    )
    differentiated = str(jax.make_jaxpr(jax.grad(loss))(model))
    assert 'remat' in differentiated


def test_causal_lm_enable_remat_forwards_to_transformer_model():
    causal_lm = object.__new__(TransformerCausalLM)
    causal_lm.model = type(
        'RematTarget',
        (),
        {
            'enable_remat': lambda self: setattr(self, 'enabled', True),
            'enabled': False,
        },
    )()

    causal_lm.enable_remat()

    assert causal_lm.model.enabled is True


def test_transformer_model_forwards_per_token_position_ids():
    config = ModelConfig(
        num_hidden_layers=1,
        vocab_size=8,
        hidden_size=2,
        num_attention_heads=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        dtype='float32',
        stack_type='stack',
    )
    model = PositionTransformerModel(config, rngs=nn.Rngs(0))
    input_ids = jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    position_ids = jnp.asarray([[0, 1, 0], [0, 1, 2]])

    output = model(input_ids, position_ids=position_ids).x

    hidden_states = model.token_embedding(input_ids)
    expected = jnp.broadcast_to(position_ids[..., None], hidden_states.shape)
    expected = hidden_states + expected
    assert jnp.array_equal(output, expected)


def test_transformer_causal_lm_accepts_position_ids():
    config = ModelConfig(
        num_hidden_layers=1,
        vocab_size=8,
        hidden_size=2,
        num_attention_heads=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        dtype='float32',
        stack_type='stack',
    )
    model = PositionCausalLM(config, rngs=nn.Rngs(0))
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    position_ids = jnp.asarray([[0, 1, 0]])

    logits = model(input_ids, position_ids=position_ids).logits
    hidden_states = model.model.token_embedding(input_ids)
    expected_hidden = jnp.broadcast_to(
        position_ids[..., None],
        hidden_states.shape,
    ) + hidden_states
    expected = model.lm_head(expected_hidden)

    assert jnp.allclose(logits, expected)
