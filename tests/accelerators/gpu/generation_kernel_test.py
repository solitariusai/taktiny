import jax
import jax.numpy as jnp
import pytest

from taktiny.cosettes.layers import AttentionLegacy


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        jax.default_backend() != 'gpu',
        reason='requires a JAX GPU backend',
    ),
]


def test_gpu_flash_attention_matches_dot_product(qkv):
    query, key, value = qkv()

    expected = jax.jit(
        lambda q, k, v: AttentionLegacy.apply(
            q,
            k,
            v,
            kernel='dot_product',
            is_causal=True,
        )
    )(query, key, value)
    actual = jax.jit(
        lambda q, k, v: AttentionLegacy.apply(
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
    assert all(device.platform == 'gpu' for device in actual.devices())
    assert jnp.allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_gpu_auto_generation_uses_flash_prefill_and_dense_decode(tiny_llama):
    model = tiny_llama()
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
        attention_kernel='auto',
        **generation_args,
    )

    actual.block_until_ready()
    assert model._resolve_generation_attention_kernels('auto') == (
        'flash',
        'dot_product',
    )
    assert jnp.array_equal(actual, expected)


def test_gpu_generation_rejects_tpu_ragged_kernel(tiny_llama):
    model = tiny_llama()

    with pytest.raises(ValueError, match='requires a TPU backend'):
        model.generate(
            jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
            max_new_tokens=2,
            attention_kernel={'prefill': 'flash', 'decode': 'ragged'},
        )
