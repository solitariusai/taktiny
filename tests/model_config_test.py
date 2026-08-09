import jax.numpy as jnp

from taktiny import nn
from taktiny.layers import RotaryEmbedding
from taktiny.maestro.config import ModelConfig
from taktiny.maestro.opus.gemma import Gemma2, Gemma3
from taktiny.maestro.opus.llama import Llama
from taktiny.maestro.opus.qwen import Qwen3
from taktiny.cosettes.transformers.qwen import Qwen3DecoderLayer


def test_nested_model_config_supports_mapping_get():
    config = ModelConfig(
        rope_scaling={
            'rope_type': 'llama3',
            'factor': 8.0,
            'original_max_position_embeddings': 8192,
        }
    )

    assert config.rope_scaling.get('rope_type') == 'llama3'
    assert config.rope_scaling.get('factor', 1.0) == 8.0
    assert config.rope_scaling.get('low_freq_factor', 1.0) == 1.0


def test_rotary_embedding_accepts_nested_model_config():
    config = ModelConfig(
        rope_scaling={
            'rope_type': 'llama3',
            'factor': 8.0,
            'low_freq_factor': 1.0,
            'high_freq_factor': 4.0,
            'original_max_position_embeddings': 8192,
        }
    )
    rotary = RotaryEmbedding(8, rope_scaling=config.rope_scaling)
    q = jnp.ones((1, 4, 2, 8), dtype=jnp.float32)
    k = jnp.ones((1, 4, 1, 8), dtype=jnp.float32)

    rotated_q, rotated_k = rotary(q, k)

    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    assert jnp.all(jnp.isfinite(rotated_q))
    assert jnp.all(jnp.isfinite(rotated_k))


def test_scanned_llama_accepts_nested_rope_scaling_config():
    config = ModelConfig(
        num_hidden_layers=2,
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        rope_theta=500000.0,
        rope_scaling={
            'rope_type': 'llama3',
            'factor': 8.0,
            'low_freq_factor': 1.0,
            'high_freq_factor': 4.0,
            'original_max_position_embeddings': 8192,
        },
        rms_norm_eps=1e-5,
        dtype='float32',
    )
    model = Llama(config, rngs=nn.Rngs(0), use_list=False)

    logits, context = model(jnp.asarray([[1, 2, 3]], dtype=jnp.int32))

    assert logits.shape == (1, 3, config.vocab_size)
    assert context is None
    assert jnp.all(jnp.isfinite(logits))


def test_qwen3_uses_bias_free_attention_with_qk_norms():
    config = ModelConfig(
        num_hidden_layers=2,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        hidden_act='silu',
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        dtype='float32',
    )
    model = Qwen3(config, rngs=nn.Rngs(0), use_list=False)
    layer = model.model.layers.stacked
    attention = layer.self_attn

    assert isinstance(layer, Qwen3DecoderLayer)
    assert attention.q_norm.eps == config.rms_norm_eps
    assert attention.k_norm.eps == config.rms_norm_eps
    assert not hasattr(attention.q_proj, 'bias')
    assert not hasattr(attention.k_proj, 'bias')
    assert not hasattr(attention.v_proj, 'bias')
    assert not hasattr(attention.o_proj, 'bias')
    assert model.tied_word_embeddings

    logits, context = model(jnp.asarray([[1, 2, 3]], dtype=jnp.int32))

    assert logits.shape == (1, 3, config.vocab_size)
    assert context is None
    assert jnp.all(jnp.isfinite(logits))


def test_scanned_gemma2_matches_unrolled_alternating_attention():
    config = ModelConfig(
        num_hidden_layers=2,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        hidden_act='gelu_pytorch_tanh',
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        query_pre_attn_scalar=4,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        sliding_window=16,
        dtype='float32',
    )
    unrolled = Gemma2(config, rngs=nn.Rngs(0), use_list=True)
    scanned = Gemma2(config, rngs=nn.Rngs(0), use_list=False)
    input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    expected, _ = unrolled(input_ids)
    actual, _ = scanned(input_ids)

    assert isinstance(scanned.model.layers, nn.SeqStack)
    assert not scanned.model.use_list
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_scanned_gemma3_matches_unrolled_local_global_attention():
    config = ModelConfig(
        num_hidden_layers=6,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        rope_local_base_freq=10_000.0,
        rope_parameters=None,
        sliding_window=4,
        sliding_window_pattern=6,
        rms_norm_eps=1e-6,
        hidden_act='gelu_pytorch_tanh',
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        query_pre_attn_scalar=4,
        dtype='float32',
    )
    unrolled = Gemma3(config, rngs=nn.Rngs(0), use_list=True)
    scanned = Gemma3(config, rngs=nn.Rngs(0), use_list=False)
    input_ids = jnp.asarray([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32)

    expected, _ = unrolled(input_ids)
    actual, _ = scanned(input_ids)

    assert isinstance(scanned.model.layers, nn.SeqStack)
    assert not scanned.model.use_list
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
