import jax
import jax.numpy as jnp
import pytest

from taktiny.layers import Attention


pytestmark = [
    pytest.mark.tpu,
    pytest.mark.skipif(
        jax.default_backend() != 'tpu',
        reason='requires a JAX TPU backend',
    ),
]


def test_tpu_flash_attention_matches_dot_product(qkv):
    query, key, value = qkv(dtype='bfloat16')

    expected = jax.jit(
        lambda q, k, v: Attention.apply(
            q,
            k,
            v,
            kernel='dot_product',
            is_causal=True,
        )
    )(query, key, value)
    actual = jax.jit(
        lambda q, k, v: Attention.apply(
            q,
            k,
            v,
            kernel='flash',
            is_causal=True,
            block_q=4,
            block_kv=4,
        )
    )(query, key, value)

    actual.block_until_ready()
    assert all(device.platform == 'tpu' for device in actual.devices())
    assert jnp.allclose(actual, expected, rtol=2e-2, atol=2e-2)


def test_tpu_ragged_decode_matches_prefix_masked_attention(qkv):
    query, key, value = qkv(
        query_length=1,
        key_length=256,
        head_dim=128,
        dtype='bfloat16',
    )
    lengths = jnp.asarray([256, 137], dtype=jnp.int32)
    prefix_mask = (
        jnp.arange(key.shape[1])[None, None, None, :]
        < lengths[:, None, None, None]
    )

    expected = jax.jit(
        lambda q, k, v, mask: Attention.apply(
            q,
            k,
            v,
            kernel='dot_product',
            mask=mask,
        )
    )(query, key, value, prefix_mask)
    actual = Attention.apply(
        query,
        key,
        value,
        kernel='ragged',
        lengths=lengths,
        block_size=256,
    )

    actual.block_until_ready()
    assert all(device.platform == 'tpu' for device in actual.devices())
    assert jnp.allclose(actual, expected, rtol=3e-2, atol=3e-2)


def test_tpu_auto_generation_uses_flash_prefill_and_ragged_decode(
    tiny_llama,
):
    model = tiny_llama(head_dim=128, dtype='bfloat16')
    input_ids = jnp.tile(
        jnp.arange(254, dtype=jnp.int32)[None, :] % model.vocab_size,
        (2, 1),
    )
    attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
    attention_mask = attention_mask.at[0, :2].set(False)
    generation_args = {
        'max_new_tokens': 2,
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
        attention_kernel='auto',
        **generation_args,
    )

    actual.block_until_ready()
    assert model._resolve_generation_attention_kernels('auto') == (
        'flash',
        'ragged',
    )
    assert jnp.array_equal(actual, expected)
