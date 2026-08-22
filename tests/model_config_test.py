import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.cosettes.layers import RotaryEmbedding
from taktiny.cosettes.transformers.gemma import GemmaRMSNorm
from taktiny.maestro.config import ModelConfig
from taktiny.maestro.opus.gemma import (
    Gemma,
    Gemma2,
    Gemma2Model,
    Gemma3,
)
from taktiny.maestro.opus.llama import Llama
from taktiny.maestro.opus.qwen import Qwen2, Qwen3
from taktiny.cosettes.transformers.qwen import (
    Qwen2DecoderLayer,
    Qwen3DecoderLayer,
)


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
    model = Llama(config, rngs=nn.Rngs(0), stack_type='stack')

    output = model(jnp.asarray([[1, 2, 3]], dtype=jnp.int32))

    assert output.logits.shape == (1, 3, config.vocab_size)
    assert output.kv_cache is None
    assert jnp.all(jnp.isfinite(output.logits))


@pytest.mark.parametrize('stack_type', ['list', 'stack'])
def test_gemma_projects_normalized_hidden_states_with_tied_embedding(stack_type):
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
        dtype='float32',
        eos_token_id=2,
        pad_token_id=0,
    )
    model = Gemma(config, rngs=nn.Rngs(0), stack_type=stack_type)
    input_ids = jnp.asarray([[1, 3, 4]], dtype=jnp.int32)
    position_ids = jnp.arange(input_ids.shape[1])[None, :]

    model_output = model.model(
        input_ids,
        position_ids=position_ids,
        is_causal=True,
        kernel='dot_product',
    )
    output = model(
        input_ids,
        position_ids=position_ids,
        is_causal=True,
        kernel='dot_product',
    )
    expected = jnp.einsum(
        '...d,vd->...v',
        model_output.x,
        model.model.token_embedding.embedding.value,
    )

    assert isinstance(model.model.norm, GemmaRMSNorm)
    assert model.lm_head is None
    assert jnp.array_equal(
        model._lm_weight(),
        model.model.token_embedding.embedding.value.T,
    )
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert output.kv_cache is None
    assert jnp.allclose(output.logits, expected)

    replacement = jnp.full_like(
        model.model.token_embedding.embedding.value,
        7,
    )
    model.model.token_embedding.embedding.value = replacement
    assert jnp.array_equal(model._lm_weight(), replacement.T)


def test_gemma_updates_tuple_kv_cache():
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
        dtype='float32',
        eos_token_id=None,
        pad_token_id=0,
    )
    model = Gemma(config, rngs=nn.Rngs(0), stack_type='stack')
    input_ids = jnp.asarray([[1, 3, 4]], dtype=jnp.int32)

    cache_shape = (
        config.num_hidden_layers,
        input_ids.shape[0],
        8,
        config.num_key_value_heads,
        config.head_dim,
    )
    kv_cache = (
        jnp.zeros(cache_shape, dtype=jnp.float32),
        jnp.zeros(cache_shape, dtype=jnp.float32),
    )
    position_ids = jnp.arange(input_ids.shape[1])[None, :]
    output = model(
        input_ids,
        kv_cache=kv_cache,
        position_ids=position_ids,
        cache_position=position_ids,
        is_causal=True,
        kernel='dot_product',
    )

    assert output.logits.shape == (1, 3, config.vocab_size)
    assert isinstance(output.kv_cache, tuple)
    assert output.kv_cache[0].shape == cache_shape
    assert output.kv_cache[1].shape == cache_shape


def test_gemma_preserves_quantization_stored_in_config():
    config = ModelConfig(
        num_hidden_layers=1,
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
        dtype='float32',
        quant='int4',
    )

    model = Gemma(config, rngs=nn.Rngs(0), stack_type='stack')

    assert model.config.quant == 'int4'
    assert model.model.token_embedding.embedding.quantization == 'int4'
    assert (
        model.model.layers.stacked.attention.q_proj.weight.quantization
        == 'int4'
    )


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
    model = Qwen3(config, rngs=nn.Rngs(0), stack_type='stack')
    layer = model.model.layers.stacked
    attention = layer.attention

    assert isinstance(layer, Qwen3DecoderLayer)
    assert attention.q_norm.eps == config.rms_norm_eps
    assert attention.k_norm.eps == config.rms_norm_eps
    assert not hasattr(attention.q_proj, 'bias')
    assert not hasattr(attention.k_proj, 'bias')
    assert not hasattr(attention.v_proj, 'bias')
    assert not hasattr(attention.o_proj, 'bias')
    assert model.config.tie_word_embeddings

    output = model(jnp.asarray([[1, 2, 3]], dtype=jnp.int32))

    assert output.logits.shape == (1, 3, config.vocab_size)
    assert output.kv_cache is None
    assert jnp.all(jnp.isfinite(output.logits))


def test_scanned_qwen2_matches_unrolled_mixed_window_attention():
    config = ModelConfig(
        num_hidden_layers=4,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        hidden_act='silu',
        attention_bias=True,
        attention_dropout=0.0,
        use_sliding_window=True,
        sliding_window=2,
        max_window_layers=2,
        dtype='float32',
    )
    unrolled = Qwen2(config, rngs=nn.Rngs(0), stack_type='list')
    scanned = Qwen2(config, rngs=nn.Rngs(0), stack_type='stack')
    input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    expected = unrolled(
        input_ids,
        is_causal=True,
        kernel='dot_product',
    ).logits
    actual = scanned(
        input_ids,
        is_causal=True,
        kernel='dot_product',
    ).logits
    layer = scanned.model.layers.stacked

    assert isinstance(layer, Qwen2DecoderLayer)
    assert layer.sliding_pattern == (False, False, True, True)
    assert layer.attention.window_size == 2
    assert hasattr(layer.attention.q_proj, 'bias')
    assert hasattr(layer.attention.k_proj, 'bias')
    assert hasattr(layer.attention.v_proj, 'bias')
    assert not hasattr(layer.attention.o_proj, 'bias')
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


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
    unrolled = Gemma2(config, rngs=nn.Rngs(0), stack_type='list')
    scanned = Gemma2(config, rngs=nn.Rngs(0), stack_type='stack')
    input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    expected = unrolled(
        input_ids,
        is_causal=True,
        kernel='dot_product',
    ).logits
    actual = scanned(
        input_ids,
        is_causal=True,
        kernel='dot_product',
    ).logits

    assert isinstance(scanned.model, Gemma2Model)
    assert isinstance(scanned.model.layers, nn.SeqStack)
    assert scanned.model.layers.group_sizes == (2,)
    assert scanned.model.layers.stacked.attention.window_size == 16
    assert isinstance(
        scanned.model.layers.stacked.attention.window_size,
        int,
    )
    assert scanned.model.layers.stacked.sliding_pattern == (True, False)
    assert scanned.model.layers.stacked.attention.scaling == 0.5
    assert scanned.model.layers.stacked.attention.softcap == 50.0
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
    unrolled = Gemma3(config, rngs=nn.Rngs(0), stack_type='list')
    scanned = Gemma3(config, rngs=nn.Rngs(0), stack_type='stack')
    input_ids = jnp.asarray([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32)

    expected = unrolled(
        input_ids,
        is_causal=True,
        kernel='dot_product',
    ).logits
    actual = scanned(
        input_ids,
        is_causal=True,
        kernel='dot_product',
    ).logits

    assert isinstance(scanned.model.layers, nn.SeqStack)
    assert scanned.config.layer_types == [
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'full_attention',
    ]
    assert scanned.model.sliding_pattern == (
        True,
        True,
        True,
        True,
        True,
        False,
    )
    assert scanned.model.rotary_embedding.base == 1_000_000.0
    assert scanned.model.local_rotary_embedding.base == 10_000.0
    assert isinstance(
        scanned.model.layers.stacked.attention.q_norm,
        GemmaRMSNorm,
    )
    assert isinstance(
        scanned.model.layers.stacked.attention.k_norm,
        GemmaRMSNorm,
    )
    assert isinstance(
        scanned.model.layers.stacked.attention.window_size,
        int,
    )
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_gemma3_fills_missing_serialized_config_defaults():
    config = ModelConfig(
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        dtype='float32',
    )

    model = jax.eval_shape(
        lambda: Gemma3(config, rngs=nn.Rngs(0), stack_type='stack')
    )

    assert model.config.vocab_size == 262_208
    assert model.model.token_embedding.embedding.shape == (262_208, 8)
    assert model.model.layers.stacked.attention.q_norm.weight.shape == (1, 6)
    assert model.model.layers.stacked.attention.k_norm.weight.shape == (1, 6)
    assert model.config.layer_types == ['sliding_attention']
    assert Gemma3._default_config.hidden_size == 2304
    assert Gemma3._default_config.layer_types is None


def test_model_config_defaults_are_not_mutated_by_overrides():
    defaults = ModelConfig(
        hidden_size=8,
        nested={'value': 1, 'preserved': 2},
    )
    overrides = ModelConfig(
        hidden_size=16,
        nested={'value': 3},
    )

    merged = defaults.with_overrides(overrides)
    merged.nested.preserved = 99

    assert merged.hidden_size == 16
    assert merged.nested.value == 3
    assert defaults.hidden_size == 8
    assert defaults.nested.value == 1
    assert defaults.nested.preserved == 2


def test_local_load_config_missing_file_raises_file_not_found():
    missing = '/nonexistent/taktiny/checkpoint'

    with pytest.raises(FileNotFoundError):
        ModelConfig.load_config(missing, local=True)


def test_local_load_config_honors_custom_filename(tmp_path):
    (tmp_path / 'custom.json').write_text('{"vocab_size": 16}')

    config = ModelConfig.load_config(
        str(tmp_path),
        filename='custom.json',
        local=True,
    )

    assert config.vocab_size == 16


def test_local_from_pretrained_missing_config_raises_clear_error():
    missing = '/nonexistent/taktiny/checkpoint'

    with pytest.raises(FileNotFoundError):
        Llama.from_pretrained(missing, local=True)
