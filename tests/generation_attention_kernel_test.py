import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.maestro.config import ModelConfig
from taktiny.maestro.opus.llama import Llama


def _model(*, use_list=False, **config_overrides):
    values = {
        'num_hidden_layers': 1,
        'vocab_size': 32,
        'hidden_size': 16,
        'intermediate_size': 32,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'head_dim': 4,
        'max_position_embeddings': 64,
        'rope_theta': 10_000.0,
        'rope_scaling': None,
        'rms_norm_eps': 1e-6,
        'hidden_act': 'silu',
        'attention_bias': False,
        'attention_dropout': 0.0,
        'mlp_bias': False,
        'tie_word_embeddings': False,
        'eos_token_id': None,
        'pad_token_id': 0,
        'dtype': 'float32',
    }
    values.update(config_overrides)
    return Llama(
        ModelConfig(**values),
        rngs=nn.Rngs(0),
        use_list=use_list,
    )


def test_generation_attention_kernel_auto_policy(monkeypatch):
    model = _model()

    monkeypatch.setattr('jax.default_backend', lambda: 'tpu')
    assert model._resolve_generation_attention_kernels('auto') == (
        'flash',
        'ragged',
    )


def test_generation_attention_kernel_auto_respects_model_features(monkeypatch):
    monkeypatch.setattr('jax.default_backend', lambda: 'tpu')
    softcapped = _model(attn_logit_softcapping=50.0)
    sliding = _model(sliding_window=16)
    inactive_sliding = _model(
        sliding_window=16,
        use_sliding_window=False,
    )

    assert softcapped._resolve_generation_attention_kernels('auto') == (
        'dot_product',
        'dot_product',
    )
    assert sliding._resolve_generation_attention_kernels('auto') == (
        'flash',
        'dot_product',
    )
    assert inactive_sliding._resolve_generation_attention_kernels('auto') == (
        'flash',
        'ragged',
    )


def test_generation_attention_kernel_accepts_phase_mapping_and_aliases():
    model = _model()

    assert model._resolve_generation_attention_kernels(
        {
            'prefill': 'flash_attention',
            'decode': 'ragged_attention',
        }
    ) == ('flash', 'ragged')
    assert model._resolve_generation_attention_kernels('jax') == (
        'dot_product',
        'dot_product',
    )


@pytest.mark.parametrize(
    ('kernel', 'message'),
    [
        ('unknown', 'unsupported attention kernel'),
        ({'prefill': 'ragged'}, 'decode-only'),
        ({'decode': 'ring'}, 'prebuilt topology-specific kernel'),
        ({'prefill': 'flash', 'extra': 'flash'}, 'unknown attention_kernel'),
    ],
)
def test_generation_attention_kernel_rejects_invalid_policies(kernel, message):
    model = _model()

    with pytest.raises((TypeError, ValueError), match=message):
        model._resolve_generation_attention_kernels(kernel)


def test_flash_prefill_and_ragged_decode_match_dot_product_generation():
    model = _model()
    input_ids = jnp.asarray(
        [[0, 0, 1, 2], [3, 4, 5, 6]],
        dtype=jnp.int32,
    )
    attention_mask = jnp.asarray(
        [[0, 0, 1, 1], [1, 1, 1, 1]],
        dtype=jnp.bool_,
    )
    generation_args = {
        'max_new_tokens': 3,
        'temperature': 0.0,
        'top_k': 0,
        'attention_mask': attention_mask,
    }

    expected = model.generate(
        input_ids,
        attention_kernel='dot_product',
        **generation_args,
    )
    actual = model.generate(
        input_ids,
        attention_kernel={
            'prefill': 'flash',
            'decode': 'ragged',
        },
        **generation_args,
    )
    automatic = model.generate(
        input_ids,
        attention_kernel='auto',
        **generation_args,
    )
    streamed = jnp.concatenate(
        tuple(
            model.stream_generate(
                input_ids,
                attention_kernel={
                    'prefill': 'flash',
                    'decode': 'ragged',
                },
                **generation_args,
            )
        ),
        axis=1,
    )

    assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(automatic, expected)
    assert jnp.array_equal(streamed, expected[:, input_ids.shape[1]:])


def test_generation_attention_kernel_auto_uses_dense_decode_off_tpu(
    monkeypatch,
):
    model = _model()
    monkeypatch.setattr('jax.default_backend', lambda: 'gpu')

    assert model._resolve_generation_attention_kernels('auto') == (
        'flash',
        'dot_product',
    )

    with pytest.raises(ValueError, match='requires a TPU backend'):
        model._resolve_generation_attention_kernels(
            {'prefill': 'flash', 'decode': 'ragged'}
        )


def test_generation_uses_current_sliced_layer_count():
    model = _model(num_hidden_layers=3, use_list=True)
    model.model.layers = model.model.layers[:2]

    output = model.generate(
        jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
        max_new_tokens=2,
        temperature=0.0,
        top_k=0,
        attention_kernel='dot_product',
    )

    assert isinstance(model.model.layers, nn.List)
    assert output.shape == (1, 5)
