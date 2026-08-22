import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taktiny import nn
from taktiny.maestro.config import ModelConfig
from taktiny.maestro.opus.gemma import Gemma


def _tiny_gemma(*, stack_type: str = 'stack') -> Gemma:
    config = ModelConfig(
        num_hidden_layers=1,
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        hidden_act='gelu_pytorch_tanh',
        dtype='float32',
        eos_token_id=None,
        pad_token_id=0,
    )
    return Gemma(config, rngs=nn.Rngs(0), stack_type=stack_type)


def test_new_causal_lm_sampling_applies_repetition_penalty():
    model = _tiny_gemma()
    logits = jnp.asarray([[2.0, 1.5, 0.0]], dtype=jnp.float32)
    seen_tokens = jnp.asarray([[True, False, False]])

    token = model._sample(
        logits,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        key=jax.random.key(0),
        seen_tokens=seen_tokens,
        repetition_penalty=2.0,
    )

    np.testing.assert_array_equal(token, np.asarray([[1]]))


def test_new_causal_lm_resolves_supported_attention_kernels():
    model = _tiny_gemma()

    assert model._canonical_attention_kernel(' JAX ') == 'dot_product'
    assert model._canonical_attention_kernel('flash_attention') == 'flash'
    assert model._resolve_generation_attention_kernels('auto') == (
        'dot_product',
        'dot_product',
    )
    assert model._resolve_generation_attention_kernels(
        {'prefill': 'flash', 'decode': 'standard'}
    ) == ('flash', 'dot_product')

    with pytest.raises(ValueError, match='unsupported attention kernel'):
        model._canonical_attention_kernel('ragged')
    with pytest.raises(ValueError, match='phase keys'):
        model._resolve_generation_attention_kernels({'other': 'flash'})


@pytest.mark.parametrize('stack_type', ['stack', 'list'])
def test_new_causal_lm_generates_from_padded_batched_prompts(stack_type):
    model = _tiny_gemma(stack_type=stack_type)
    input_ids = jnp.asarray(
        [[0, 1, 2], [3, 4, 5]],
        dtype=jnp.int32,
    )
    attention_mask = jnp.asarray(
        [[0, 1, 1], [1, 1, 1]],
        dtype=jnp.int32,
    )

    output = model.generate(
        input_ids,
        max_new_tokens=3,
        temperature=0.0,
        attention_mask=attention_mask,
    )

    assert output.shape == (2, 6)
    np.testing.assert_array_equal(output[:, :3], input_ids)


def test_new_causal_lm_stream_matches_generate():
    model = _tiny_gemma()
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    generation_args = {
        'max_new_tokens': 3,
        'temperature': 0.0,
        'seed': 7,
    }

    expected = model.generate(input_ids, **generation_args)
    streamed = list(model.stream_generate(input_ids, **generation_args))
    actual = jnp.concatenate([input_ids, *streamed], axis=1)

    np.testing.assert_array_equal(actual, expected)


def test_new_causal_lm_forwards_tokens_to_streamer():
    class Streamer:
        def __init__(self):
            self.values = []
            self.ended = False

        def put(self, value):
            self.values.append(np.asarray(value))

        def end(self):
            self.ended = True

    model = _tiny_gemma()
    input_ids = jnp.asarray([[1, 2]], dtype=jnp.int32)
    streamer = Streamer()

    output = model.generate(
        input_ids,
        max_new_tokens=2,
        temperature=0.0,
        streamer=streamer,
    )

    assert streamer.ended
    assert [value.shape for value in streamer.values] == [(1, 2), (1, 1), (1, 1)]
    np.testing.assert_array_equal(
        np.concatenate(streamer.values, axis=1),
        output,
    )
