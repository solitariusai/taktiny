import jax
import pytest

from taktiny import nn
from taktiny.maestro.config import ModelConfig
from taktiny.maestro.opus.llama import Llama


@pytest.fixture
def qkv():
    def make(
        *,
        query_length=8,
        key_length=8,
        query_heads=4,
        key_heads=2,
        head_dim=16,
        dtype='float32',
    ):
        shapes = (
            (2, query_length, query_heads, head_dim),
            (2, key_length, key_heads, head_dim),
            (2, key_length, key_heads, head_dim),
        )
        return tuple(
            jax.random.normal(jax.random.key(index), shape, dtype=dtype)
            for index, shape in enumerate(shapes)
        )

    return make


@pytest.fixture
def tiny_llama():
    def make(*, head_dim=16, dtype='float32'):
        hidden_size = 4 * head_dim
        config = ModelConfig(
            num_hidden_layers=1,
            vocab_size=64,
            hidden_size=hidden_size,
            intermediate_size=2 * hidden_size,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=head_dim,
            max_position_embeddings=512,
            rope_theta=10_000.0,
            rope_scaling=None,
            rms_norm_eps=1e-6,
            hidden_act='silu',
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            tie_word_embeddings=False,
            eos_token_id=None,
            pad_token_id=0,
            dtype=dtype,
        )
        return Llama(config, rngs=nn.Rngs(0), use_list=False)

    return make
