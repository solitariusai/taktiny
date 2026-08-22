import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp


_MODULE_PATH = (
    Path(__file__).parents[1]
    / 'src/taktiny/cosettes/kernels/attention/flash_attention.py'
)
_SPEC = importlib.util.spec_from_file_location(
    'taktiny_boundary_flash_attention',
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
flash_attention_boundary_masked = _MODULE.flash_attention


def _qkv(*, query_length=7, key_length=7, value_dim=8):
    query = jax.random.normal(
        jax.random.key(0),
        (2, query_length, 4, 8),
    )
    key = jax.random.normal(
        jax.random.key(1),
        (2, key_length, 2, 8),
    )
    value = jax.random.normal(
        jax.random.key(2),
        (2, key_length, 2, value_dim),
    )
    return query, key, value


def test_boundary_flash_matches_packed_causal_attention():
    query, key, value = _qkv()
    boundaries = jnp.asarray([0, 3, 7], dtype=jnp.int32)
    positions = jnp.arange(7)
    documents = jnp.sum(
        positions[:, None] >= boundaries[None, 1:],
        axis=-1,
    )
    mask = (
        (documents[:, None] == documents[None, :])
        & (positions[None, :] <= positions[:, None])
    )

    actual = flash_attention_boundary_masked(
        query,
        key,
        value,
        boundary_ids=boundaries,
        is_causal=True,
        block_q=3,
        block_kv=4,
    )
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=mask,
    )

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_boundary_flash_supports_callable_cross_document_visibility():
    query, key, value = _qkv(query_length=5, key_length=5)
    boundaries = jnp.asarray([0, 2, 5], dtype=jnp.int32)
    connectivity = jnp.asarray([[True, True], [False, True]])

    def visibility(query_positions, key_positions, query_docs, key_docs):
        del query_positions, key_positions
        return connectivity[query_docs, key_docs]

    actual = jax.jit(
        lambda q, k, v: flash_attention_boundary_masked(
            q,
            k,
            v,
            mask=visibility,
            boundary_ids=boundaries,
            respect_boundaries=False,
            block_q=3,
            block_kv=2,
        )
    )(query, key, value)
    positions = jnp.arange(5)
    documents = jnp.sum(
        positions[:, None] >= boundaries[None, 1:],
        axis=-1,
    )
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=connectivity[documents[:, None], documents[None, :]],
    )

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_boundary_flash_handles_arbitrary_masks_and_fully_masked_rows():
    query, key, value = _qkv(
        query_length=5,
        key_length=7,
        value_dim=6,
    )
    mask = jnp.asarray(
        [
            [True, True, False, False, False, False, False],
            [True, True, True, False, False, False, False],
            [False, False, False, False, False, False, False],
            [True, False, True, False, True, False, False],
            [True, True, True, True, True, True, True],
        ]
    )

    output, statistics = flash_attention_boundary_masked(
        query,
        key,
        value,
        mask=mask,
        block_q=3,
        block_kv=4,
        save_residuals=True,
    )
    gradients = jax.grad(
        lambda q: jnp.sum(
            flash_attention_boundary_masked(
                q,
                key,
                value,
                mask=mask,
                block_q=3,
                block_kv=4,
            )
        )
    )(query)

    assert output.shape == (2, 5, 4, 6)
    assert statistics['logsumexp'].shape == (2, 5, 4)
    assert statistics['max_logits'].shape == (2, 5, 4)
    assert jnp.all(output[:, 2] == 0)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.all(jnp.isfinite(gradients))


def test_boundary_flash_supports_per_batch_decode_offsets():
    query, key, value = _qkv(query_length=1, key_length=7)
    offsets = jnp.asarray([2, 6], dtype=jnp.int32)

    actual = flash_attention_boundary_masked(
        query,
        key,
        value,
        is_causal=True,
        query_offset=offsets,
        block_q=2,
        block_kv=4,
    )
    key_positions = jnp.arange(7)
    mask = key_positions[None, None, None, :] <= offsets[:, None, None, None]
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=mask,
    )

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_boundary_flash_pads_broadcast_generation_mask():
    query, key, value = _qkv(query_length=2, key_length=66)
    mask = jnp.ones((2, 1, 1, 66), dtype=jnp.bool_)

    actual = jax.jit(
        lambda q, k, v, m: flash_attention_boundary_masked(
            q,
            k,
            v,
            mask=m,
            is_causal=True,
            block_q=128,
            block_kv=128,
        )
    )(query, key, value, mask)
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=mask,
        is_causal=True,
    )

    assert actual.shape == query.shape
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
