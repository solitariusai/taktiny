import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taktiny import nn
from taktiny.cosettes.kernels.attention.flash_attention import (
    flash_attention_block_masked,
)
from taktiny.cosettes.kernels.attention.tokamax_splash import ring_attention_kernel
from taktiny.cosettes.kernels.attention.tokamax_splash import (
    splash_attention_mask,
)
from taktiny.cosettes.kernels.ragged import ragged_gather
from taktiny.cosettes.kernels.ragged.ragged_gather_reduce import (
    ragged_gather_reduce as ragged_gather_reduce_v1,
)
from taktiny.cosettes.kernels.ragged.ragged_gather_reduce_v2 import (
    ragged_gather_reduce as ragged_gather_reduce_v2,
)
from taktiny.cosettes.layers.attention import AttentionLegacy
from taktiny.cosettes.layers.ffn import FusedGateMLP, MoEFFN


def _qkv(*, query_heads=4, key_heads=2, query_length=4, key_length=4):
    query = jax.random.normal(
        jax.random.key(0),
        (2, query_length, query_heads, 8),
    )
    key = jax.random.normal(
        jax.random.key(1),
        (2, key_length, key_heads, 8),
    )
    value = jax.random.normal(
        jax.random.key(2),
        (2, key_length, key_heads, 8),
    )
    return query, key, value


@pytest.mark.parametrize(
    ('kernel', 'kernel_kwargs'),
    [
        ('flash', {'block_q': 2, 'block_kv': 2}),
        ('splash', {}),
    ],
)
def test_attention_kernel_entry_matches_dot_product_gqa(
    kernel,
    kernel_kwargs,
):
    query, key, value = _qkv(query_length=3, key_length=5)
    mask = jnp.asarray(
        [
            [
                [
                    [True, True, True, False, False],
                    [True, True, True, True, False],
                    [True, True, True, True, True],
                ]
            ],
            [
                [
                    [True, False, False, False, False],
                    [True, True, False, False, False],
                    [True, True, True, False, False],
                ]
            ],
        ]
    )

    expected = AttentionLegacy.apply(
        query,
        key,
        value,
        kernel='dot_product',
        mask=mask,
        scale=0.25,
        is_causal=True,
    )
    actual = AttentionLegacy.apply(
        query,
        key,
        value,
        kernel=kernel,
        mask=mask,
        scale=0.25,
        is_causal=True,
        **kernel_kwargs,
    )

    assert actual.shape == query.shape
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_attention_derives_packed_boundaries_from_position_ids():
    attention = AttentionLegacy(
        hidden_size=8,
        num_heads=2,
        head_dim=4,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    hidden_states = jax.random.normal(jax.random.key(4), (2, 4, 8))
    position_ids = jnp.asarray([
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    packed_segments = jnp.cumsum(position_ids == 0, axis=-1)
    causal = jnp.tril(jnp.ones((4, 4), dtype=jnp.bool_))
    dense_mask = (
        packed_segments[:, None, :, None]
        == packed_segments[:, None, None, :]
    ) & causal

    actual, _ = attention(
        hidden_states,
        position_idx=position_ids,
        is_causal=True,
    )
    query = attention.q_proj(hidden_states)
    key = attention.k_proj(hidden_states)
    value = attention.v_proj(hidden_states)
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=dense_mask,
    )
    expected = attention.o_proj(expected)

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ('kernel', 'kernel_kwargs'),
    [
        ('flash_attention', {'block_q': 2, 'block_kv': 2}),
        ('splash_attention', {}),
    ],
)
def test_attention_kernel_entry_is_jittable(kernel, kernel_kwargs):
    query, key, value = _qkv()
    mask = jnp.tril(jnp.ones((4, 4), dtype=jnp.bool_))
    apply = jax.jit(
        lambda q, k, v: AttentionLegacy.apply(
            q,
            k,
            v,
            kernel=kernel,
            mask=mask,
            scale=0.25,
            **kernel_kwargs,
        )
    )

    actual = apply(query, key, value)
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=mask,
        scale=0.25,
    )

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_flash_kernel_preserves_gqa_heads_and_residual_shapes():
    query, key, value = _qkv()
    query = query.transpose(0, 2, 1, 3) * 0.25
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    mask = jnp.tril(jnp.ones((4, 4), dtype=jnp.bool_))

    actual, statistics = flash_attention_block_masked(
        query,
        key,
        value,
        segment_ids=None,
        block_kv=2,
        block_q=2,
        mask=mask,
        mask_value=-1e9,
        save_residuals=True,
    )
    expected = jax.nn.dot_product_attention(
        query.transpose(0, 2, 1, 3),
        key.transpose(0, 2, 1, 3),
        value.transpose(0, 2, 1, 3),
        mask=mask,
        scale=1.0,
    ).transpose(0, 2, 1, 3)

    assert actual.shape == (2, 4, 4, 8)
    assert statistics['logsumexp'].shape == (2, 4, 4)
    assert statistics['max_logits'].shape == (2, 4, 4)
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_flash_kernel_softcap_matches_dense_reference():
    cap = 1.0
    query = jnp.asarray(
        [[[[10.0], [10.0]]]],
        dtype=jnp.float32,
    )
    key = jnp.asarray(
        [[[[10.0], [0.0]]]],
        dtype=jnp.float32,
    )
    value = jnp.asarray(
        [[[[1.0], [3.0]]]],
        dtype=jnp.float32,
    )
    mask = jnp.asarray(
        [[True, True], [True, False]],
        dtype=jnp.bool_,
    )

    actual = flash_attention_block_masked(
        query,
        key,
        value,
        segment_ids=None,
        block_kv=2,
        block_q=1,
        mask=mask,
        mask_value=-1e9,
        cap=cap,
    )
    logits = jnp.einsum('bhqd,bhkd->bhqk', query, key)
    logits = jnp.tanh(logits / cap) * cap
    logits = jnp.where(mask[None, None], logits, -1e9)
    expected = jnp.einsum(
        'bhqk,bhkd->bhqd',
        jax.nn.softmax(logits, axis=-1),
        value,
    )

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_splash_entry_supports_per_batch_per_head_masks():
    query, key, value = _qkv()
    row = jnp.arange(4)[:, None]
    column = jnp.arange(4)[None, :]
    masks = jnp.stack(
        [
            jnp.stack(
                [
                    column <= row,
                    column >= row,
                    jnp.ones((4, 4), dtype=jnp.bool_),
                    jnp.eye(4, dtype=jnp.bool_),
                ]
            ),
            jnp.stack(
                [
                    jnp.eye(4, dtype=jnp.bool_),
                    column <= row,
                    column >= row,
                    jnp.ones((4, 4), dtype=jnp.bool_),
                ]
            ),
        ]
    )

    actual = AttentionLegacy.apply(
        query,
        key,
        value,
        kernel='splash',
        mask=masks,
        scale=0.25,
    )
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=masks,
        scale=0.25,
    )

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('key_heads', [4, 2])
def test_ragged_entry_matches_prefix_masked_decode(key_heads):
    query, key, value = _qkv(
        key_heads=key_heads,
        query_length=1,
        key_length=8,
    )
    lengths = jnp.asarray([8, 5], dtype=jnp.int32)
    prefix_mask = (
        jnp.arange(8)[None, None, None, :]
        < lengths[:, None, None, None]
    )

    actual = AttentionLegacy.apply(
        query,
        key,
        value,
        kernel='ragged',
        lengths=lengths,
        scale=0.25,
        block_size=4,
        interpret=True,
    )
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=prefix_mask,
        scale=0.25,
    )

    assert actual.shape == query.shape
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_ragged_entry_rejects_prefill_and_non_prefix_masks():
    query, key, value = _qkv()

    with pytest.raises(ValueError, match='decode kernel'):
        AttentionLegacy.apply(
            query,
            key,
            value,
            kernel='ragged',
        )

    with pytest.raises(ValueError, match='uses lengths'):
        AttentionLegacy.apply(
            query[:, :1],
            key,
            value,
            kernel='ragged',
            mask=jnp.ones((1, 4), dtype=jnp.bool_),
        )


def test_ring_entry_calls_prebuilt_kernel_in_bhld_layout():
    query, key, value = _qkv()

    def ring_kernel(q, k, v, segment_ids):
        assert q.shape == (4, 4, 8)
        assert k.shape == (2, 4, 8)
        assert v.shape == (2, 4, 8)
        assert segment_ids is None
        return q

    actual = AttentionLegacy.apply(
        query,
        key,
        value,
        kernel='ring',
        ring_kernel=ring_kernel,
        scale=0.25,
    )

    assert jnp.allclose(actual, query * 0.25)


def test_ring_entry_requires_prebuilt_kernel():
    query, key, value = _qkv()

    with pytest.raises(ValueError, match='ring_kernel is required'):
        AttentionLegacy.apply(query, key, value, kernel='ring')


@pytest.mark.parametrize(
    'mask',
    [
        splash_attention_mask.FullMask((128, 128)),
        splash_attention_mask.CausalMask((128, 128)),
    ],
)
def test_ring_kernel_builder_creates_valid_pytree_metadata(mask):
    kernel = ring_attention_kernel.make_ring_attention(
        mask,
        is_mqa=False,
        ring_axis='seq',
        q_seq_shards=1,
        kv_seq_shards=1,
    )
    leaves, structure = jax.tree.flatten(kernel)
    restored = jax.tree.unflatten(structure, leaves)

    assert callable(restored)
    assert restored.ring_axis == 'seq'
    assert restored.expected_ring_size == 1
    assert restored.manual_sharding_spec().ring_axis == 'seq'


def test_ring_kernel_builder_rejects_dense_masks():
    with pytest.raises(NotImplementedError, match='dense NumpyMask'):
        ring_attention_kernel.make_ring_attention(
            np.ones((128, 128), dtype=np.bool_),
            is_mqa=False,
            ring_axis='seq',
        )


@pytest.mark.parametrize('kernel', ['flash', 'splash'])
def test_attention_entry_rejects_unsupported_additive_bias(kernel):
    query, key, value = _qkv()

    with pytest.raises(ValueError, match='does not support additive bias'):
        AttentionLegacy.apply(
            query,
            key,
            value,
            kernel=kernel,
            bias=jnp.zeros((1, 1, 4, 4)),
        )


def test_attention_entry_rejects_unknown_kernel():
    query, key, value = _qkv()

    with pytest.raises(ValueError, match='Unknown attention kernel'):
        AttentionLegacy.apply(query, key, value, kernel='unknown')


@pytest.mark.parametrize('kernel', [None, 1, object()])
def test_attention_entry_rejects_non_string_kernel(kernel):
    query, key, value = _qkv()

    with pytest.raises(TypeError, match='kernel must be a string'):
        AttentionLegacy.apply(query, key, value, kernel=kernel)


def _gmm_inputs(*, transpose_rhs=False):
    lhs = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 10
    rhs = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 10
    if transpose_rhs:
        rhs = rhs.swapaxes(1, 2)
    group_sizes = jnp.asarray([3, 5], dtype=jnp.int32)
    return lhs, rhs, group_sizes


def _reference_gmm(lhs, rhs, group_sizes, *, transpose_rhs=False):
    group_ids = jnp.repeat(
        jnp.arange(group_sizes.shape[0]),
        group_sizes,
        total_repeat_length=lhs.shape[0],
    )
    if transpose_rhs:
        rhs = rhs.swapaxes(1, 2)
    return jnp.einsum('mk,mkn->mn', lhs, rhs[group_ids])


@pytest.mark.parametrize('entry', [FusedGateMLP, MoEFFN])
@pytest.mark.parametrize('transpose_rhs', [False, True])
def test_gmm_entry_matches_grouped_matmul(entry, transpose_rhs):
    lhs, rhs, group_sizes = _gmm_inputs(
        transpose_rhs=transpose_rhs,
    )

    actual = entry.apply(
        lhs,
        rhs,
        group_sizes,
        kernel='GMM',
        transpose_rhs=transpose_rhs,
        interpret=True,
    )
    expected = _reference_gmm(
        lhs,
        rhs,
        group_sizes,
        transpose_rhs=transpose_rhs,
    )

    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_gmm_entry_forward_and_backward_match_reference():
    lhs, rhs, group_sizes = _gmm_inputs()
    tiling = (
        4, 4, 1,
        4, 1, 4,
        4, 4, 1,
    )

    def kernel_loss(left, right):
        out = MoEFFN.apply(
            left,
            right,
            group_sizes,
            interpret=True,
            tiling=tiling,
        )
        return jnp.sum(jnp.square(out))

    def reference_loss(left, right):
        out = _reference_gmm(left, right, group_sizes)
        return jnp.sum(jnp.square(out))

    actual = jax.grad(kernel_loss, argnums=(0, 1))(lhs, rhs)
    expected = jax.grad(reference_loss, argnums=(0, 1))(lhs, rhs)

    assert jnp.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_gmm_entry_is_jittable():
    lhs, rhs, group_sizes = _gmm_inputs()
    apply = jax.jit(
        lambda left, right, sizes: MoEFFN.apply(
            left,
            right,
            sizes,
            interpret=True,
        )
    )

    actual = apply(lhs, rhs, group_sizes)
    expected = _reference_gmm(lhs, rhs, group_sizes)

    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('entry', [FusedGateMLP, MoEFFN])
def test_route_and_unroute_preserve_token_multiplicity(entry):
    tokens = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    selected_experts = jnp.asarray(
        [
            [2, 0],
            [1, 2],
            [0, 1],
            [2, 1],
        ],
        dtype=jnp.int32,
    )
    order = jnp.argsort(selected_experts.reshape(-1)) // 2

    routed, group_sizes = entry.apply_route(
        tokens,
        selected_experts,
        num_groups=4,
    )
    restored = entry.apply_unroute(routed, selected_experts)

    assert jnp.array_equal(routed, tokens[order])
    assert jnp.array_equal(
        group_sizes,
        jnp.asarray([2, 3, 3, 0], dtype=jnp.int32),
    )
    assert jnp.array_equal(restored, tokens * 2)


def test_route_custom_gradient_matches_top_k_multiplicity():
    selected_experts = jnp.asarray(
        [[0, 1], [1, 0], [0, 1]],
        dtype=jnp.int32,
    )

    def loss(tokens):
        routed, _ = MoEFFN.apply_route(
            tokens,
            selected_experts,
            num_groups=2,
        )
        return jnp.sum(routed)

    gradient = jax.grad(loss)(
        jnp.ones((3, 4), dtype=jnp.float32)
    )

    assert jnp.array_equal(gradient, jnp.full((3, 4), 2.0))


def test_moe_entry_rejects_unknown_kernel():
    lhs, rhs, group_sizes = _gmm_inputs()

    with pytest.raises(ValueError, match='Unknown MoE kernel'):
        MoEFFN.apply(
            lhs,
            rhs,
            group_sizes,
            kernel='unknown',
        )


@pytest.mark.parametrize('entry', [FusedGateMLP, MoEFFN])
@pytest.mark.parametrize('kernel', [None, 1, object()])
def test_moe_entry_rejects_non_string_kernel(entry, kernel):
    lhs, rhs, group_sizes = _gmm_inputs()

    with pytest.raises(TypeError, match='kernel must be a string'):
        entry.apply(
            lhs,
            rhs,
            group_sizes,
            kernel=kernel,
        )


def test_ragged_gather_fallback_matches_weighted_jax_gather():
    values = jnp.arange(30, dtype=jnp.float32).reshape(10, 3)
    indices = jnp.asarray([7, 2, 2, 9], dtype=jnp.int32)
    weights = jnp.asarray([0.5, 1.0, -0.25, 2.0])

    gather = jax.jit(
        lambda x, i, w: ragged_gather(
            x,
            i,
            start=jnp.asarray(0, dtype=jnp.int32),
            end=jnp.asarray(i.shape[0], dtype=jnp.int32),
            weights=w,
            has_weights=True,
            enforce_fallback=True,
        )
    )
    actual = gather(values, indices, weights)
    expected = values[indices] * weights[:, None]

    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize(
    'ragged_gather_reduce',
    [ragged_gather_reduce_v1, ragged_gather_reduce_v2],
)
def test_ragged_gather_reduce_fallback_matches_reference(
    ragged_gather_reduce,
):
    values = jnp.arange(40, dtype=jnp.bfloat16).reshape(10, 4)
    indices = jnp.asarray([7, 2, 2, 9, 0, 4], dtype=jnp.int32)
    weights = jnp.asarray(
        [0.5, 1.0, -0.25, 2.0, 1.5, 0.75],
        dtype=jnp.float32,
    )
    valid = jnp.asarray([True, True, False, True, True, False])

    reduce = jax.jit(
        lambda x, i, w, m: ragged_gather_reduce(
            x,
            i,
            w,
            m,
            reduce_group_size=2,
            enforce_fallback=True,
        )
    )
    actual = reduce(values, indices, weights, valid)
    gathered = (
        values[indices].astype(jnp.float32)
        * weights[:, None]
    )
    gathered = jnp.where(valid[:, None], gathered, 0)
    expected = gathered.reshape(3, 2, 4).sum(axis=1).astype(
        jnp.bfloat16
    )

    assert actual.dtype == jnp.bfloat16
    assert jnp.array_equal(actual, expected)
