import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.cosettes.common import TransformerCausalLM, TransformerModel
from taktiny.maestro.config import ModelConfig


class RematTestLayer(nn.Module):
    def __init__(self, config, rngs, layer_idx=None):
        self.layer_idx = layer_idx
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
    def __init__(self, config, rngs, layer_idx=None):
        del config, rngs
        self.layer_idx = layer_idx

    def __call__(self, x, position_idx=None, **kwargs):
        del kwargs
        return x + position_idx[..., None], None


@pytest.mark.parametrize('use_list', [True, False])
def test_transformer_remat_preserves_forward_and_backward(use_list):
    config = ModelConfig(
        num_hidden_layers=2,
        vocab_size=8,
        hidden_size=4,
        dtype='float32',
    )
    model = TransformerModel(
        config,
        rngs=nn.Rngs(0),
        module=RematTestLayer,
        embedding=nn.Embedding,
        use_list=use_list,
    )
    input_ids = jnp.asarray([[1, 2, 3]])

    expected, _ = model(input_ids)
    model.enable_remat()
    actual, _ = model(input_ids)

    assert jnp.allclose(actual, expected)

    def loss(candidate):
        output, _ = candidate(input_ids)
        return jnp.sum(output)

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
        dtype='float32',
    )
    model = TransformerModel(
        config,
        rngs=nn.Rngs(0),
        module=PositionTestLayer,
        embedding=nn.Embedding,
    )
    hidden_states = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    position_ids = jnp.asarray([[0, 1, 0], [0, 1, 2]])

    output, _ = model(hidden_states, position_idx=position_ids)

    expected = jnp.broadcast_to(position_ids[..., None], hidden_states.shape)
    assert jnp.array_equal(output, expected)


def test_transformer_causal_lm_accepts_position_ids():
    config = ModelConfig(
        num_hidden_layers=1,
        vocab_size=8,
        hidden_size=2,
        dtype='float32',
    )
    model = TransformerCausalLM(
        config,
        rngs=nn.Rngs(0),
        decoder=PositionTestLayer,
    )
    hidden_states = jnp.zeros((1, 3, 2), dtype=jnp.float32)
    position_ids = jnp.asarray([[0, 1, 0]])

    logits, _ = model(hidden_states, position_ids=position_ids)
    expected_hidden = jnp.broadcast_to(
        position_ids[..., None],
        hidden_states.shape,
    )
    expected = model.lm_head(expected_hidden)

    assert jnp.allclose(logits, expected)
