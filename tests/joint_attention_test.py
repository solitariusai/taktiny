import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.cosettes.layers import JointAttention
from taktiny.utils.typing import ShardMode


class AddPositions(nn.Module):
    def __call__(self, query, key, position_idx):
        offsets = jnp.asarray(position_idx)[None, :, None, None]
        return query + offsets, key - offsets


def _inputs():
    return (
        jax.random.normal(jax.random.key(1), (2, 3, 8)),
        jax.random.normal(jax.random.key(2), (2, 2, 6)),
    )


def _manual(module, x1, x2, **attention_options):
    q1, k1, v1 = (
        module.q_proj_1(x1),
        module.k_proj_1(x1),
        module.v_proj_1(x1),
    )
    q2, k2, v2 = (
        module.q_proj_2(x2),
        module.k_proj_2(x2),
        module.v_proj_2(x2),
    )
    if module.q_norm_1 is not None:
        q1, k1 = module.q_norm_1(q1), module.k_norm_1(k1)
        q2, k2 = module.q_norm_2(q2), module.k_norm_2(k2)

    attended = jax.nn.dot_product_attention(
        jnp.concatenate((q1, q2), axis=1),
        jnp.concatenate((k1, k2), axis=1),
        jnp.concatenate((v1, v2), axis=1),
        scale=module.scaling,
        **attention_options,
    )
    out1, out2 = jnp.split(attended, (x1.shape[1],), axis=1)
    return module.o_proj_1(out1), module.o_proj_2(out2)


def test_joint_attention_matches_direct_joint_attention():
    module = JointAttention(
        8,
        6,
        2,
        4,
        bias=True,
        use_qk_norm=True,
        qk_norm_eps=1e-6,
        scaling=0.25,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    x1, x2 = _inputs()
    mask = jnp.asarray(
        [
            [True, True, True, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    )
    bias = jnp.linspace(-0.2, 0.2, 25).reshape(1, 1, 5, 5)

    actual = jax.jit(
        lambda left, right: module(
            left,
            right,
            attention_mask=mask,
            attention_bias=bias,
        )
    )(x1, x2)
    expected = _manual(module, x1, x2, mask=mask, bias=bias)

    assert actual[0].shape == x1.shape
    assert actual[1].shape == x2.shape
    assert jnp.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_joint_attention_passes_combined_positions_to_positional_embedding():
    module = JointAttention(
        8,
        6,
        2,
        4,
        pos_emb=AddPositions(),
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    x1, x2 = _inputs()
    positions = jnp.asarray([0.0, 1.0, 2.0, 10.0, 11.0])

    actual = module(x1, x2, position_idx=positions)

    q = jnp.concatenate((module.q_proj_1(x1), module.q_proj_2(x2)), axis=1)
    k = jnp.concatenate((module.k_proj_1(x1), module.k_proj_2(x2)), axis=1)
    v = jnp.concatenate((module.v_proj_1(x1), module.v_proj_2(x2)), axis=1)
    q, k = module.pos_emb(q, k, positions)
    attended = jax.nn.dot_product_attention(q, k, v)
    expected1, expected2 = jnp.split(attended, (3,), axis=1)
    expected = module.o_proj_1(expected1), module.o_proj_2(expected2)

    assert jnp.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_joint_attention_infers_packed_segments_from_reset_positions():
    module = JointAttention(
        8,
        6,
        2,
        4,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    x1, x2 = _inputs()
    position_idx = jnp.asarray(
        [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 0, 1],
        ]
    )
    segments = jnp.cumsum(position_idx == 0, axis=-1)
    segment_mask = (
        segments[:, None, :, None]
        == segments[:, None, None, :]
    )

    actual = module(x1, x2, position_idx=position_idx)
    expected = _manual(module, x1, x2, mask=segment_mask)

    assert jnp.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_joint_attention_has_finite_jitted_input_gradients():
    module = JointAttention(
        8,
        6,
        2,
        4,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    x1, x2 = _inputs()

    def loss(left, right):
        output1, output2 = module(left, right)
        return jnp.mean(jnp.square(output1)) + jnp.mean(jnp.square(output2))

    grad1, grad2 = jax.jit(jax.grad(loss, argnums=(0, 1)))(x1, x2)

    assert grad1.shape == x1.shape
    assert grad2.shape == x2.shape
    assert jnp.all(jnp.isfinite(grad1))
    assert jnp.all(jnp.isfinite(grad2))


def test_joint_attention_applies_each_output_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    sharding = NamedSharding(mesh, P())
    module = JointAttention(
        8,
        6,
        2,
        4,
        shard_mode=ShardMode.EXPLICIT,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    x1, x2 = _inputs()

    output1, output2 = jax.jit(
        lambda left, right: module(
            left,
            right,
            out_shardings=(sharding, sharding),
        )
    )(x1, x2)

    assert output1.sharding.is_equivalent_to(sharding, output1.ndim)
    assert output2.sharding.is_equivalent_to(sharding, output2.ndim)


def test_joint_attention_records_axes_and_projection_biases():
    module = JointAttention(
        8,
        6,
        2,
        4,
        bias=True,
        q_bias=False,
        q_axis_names=('embed', 'heads', 'head_dim'),
        k_axis_names=('embed', 'kv_heads', 'head_dim'),
        v_axis_names=('embed', 'kv_heads', 'head_dim'),
        o_axis_names=('heads', 'head_dim', 'embed'),
        rngs=nn.Rngs(0),
    )

    assert module.q_proj_1.weight.axis_names == (
        'embed',
        'heads',
        'head_dim',
    )
    assert module.o_proj_2.weight.axis_names == (
        'heads',
        'head_dim',
        'embed',
    )
    assert not module.q_proj_1.has_bias
    assert not module.q_proj_2.has_bias
    assert module.k_proj_1.has_bias
    assert module.o_proj_2.has_bias


def test_joint_attention_validates_its_contract():
    with pytest.raises(ValueError, match='rngs'):
        JointAttention(8, 6, 2, 4)
    with pytest.raises(ValueError, match='hidden sizes'):
        JointAttention(0, 6, 2, 4, rngs=nn.Rngs(0))

    module = JointAttention(8, 6, 2, 4, rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='same batch size'):
        module(jnp.ones((2, 3, 8)), jnp.ones((1, 2, 6)))
    with pytest.raises(ValueError, match='hidden_size2'):
        module(jnp.ones((2, 3, 8)), jnp.ones((2, 2, 7)))
    with pytest.raises(ValueError, match='exactly two values'):
        module(
            jnp.ones((2, 3, 8)),
            jnp.ones((2, 2, 6)),
            out_shardings=(None,),
        )
