from dataclasses import dataclass

import dextiny as dx
import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.layers import Attention, JointAttention


@dataclass(frozen=True)
class AttentionCase:
    name: str
    num_heads: int
    num_kv_heads: int
    is_causal: bool
    masked: bool = False


CASES = (
    AttentionCase("mha", num_heads=4, num_kv_heads=4, is_causal=False),
    AttentionCase("causal_mha", num_heads=4, num_kv_heads=4, is_causal=True),
    AttentionCase("causal_gqa", num_heads=4, num_kv_heads=2, is_causal=True),
    AttentionCase(
        "masked_gqa",
        num_heads=4,
        num_kv_heads=2,
        is_causal=False,
        masked=True,
    ),
)


def linear(x, weight, input_dims=1):
    x_axes = tuple(range(x.ndim - input_dims, x.ndim))
    weight_axes = tuple(range(input_dims))
    return jax.lax.dot_general(
        x,
        weight,
        ((x_axes, weight_axes), ((), ())),
    )


def project_qkv(hidden_states, q_weight, k_weight, v_weight):
    return (
        linear(hidden_states, q_weight),
        linear(hidden_states, k_weight),
        linear(hidden_states, v_weight),
    )


def dot_product_attention(
    qkv_states,
    attention_mask=None,
    *,
    is_causal=False,
):
    query, key, value = qkv_states
    return jax.nn.dot_product_attention(
        query=query,
        key=key,
        value=value,
        mask=attention_mask,
        is_causal=is_causal,
    )


def output_projection(attended_states, o_weight):
    return linear(attended_states, o_weight, input_dims=2)


def joint_qkv_projection(
    stream1,
    stream2,
    q1_weight,
    k1_weight,
    v1_weight,
    q2_weight,
    k2_weight,
    v2_weight,
):
    return (
        linear(stream1, q1_weight),
        linear(stream1, k1_weight),
        linear(stream1, v1_weight),
        linear(stream2, q2_weight),
        linear(stream2, k2_weight),
        linear(stream2, v2_weight),
    )


def joint_attention(qkv_states, *, length1):
    query1, key1, value1, query2, key2, value2 = qkv_states
    query = jnp.concatenate((query1, query2), axis=1)
    key = jnp.concatenate((key1, key2), axis=1)
    value = jnp.concatenate((value1, value2), axis=1)
    output = jax.nn.dot_product_attention(query, key, value)
    return tuple(jnp.split(output, (length1,), axis=1))


def joint_output_projection(
    attended_states,
    o1_weight,
    o2_weight,
):
    stream1, stream2 = attended_states
    return (
        linear(stream1, o1_weight, input_dims=2),
        linear(stream2, o2_weight, input_dims=2),
    )


def make_reference(case):
    hidden_states = dx.AbstractArray("B S D", dtype="float32")
    qkv_projection = dx.AbstractModule(
        project_qkv,
        dx.AbstractArray("D H K", dtype="float32", name="q_weight"),
        dx.AbstractArray("D G K", dtype="float32", name="k_weight"),
        dx.AbstractArray("D G K", dtype="float32", name="v_weight"),
        name="qkv_projection",
    )
    attention_operands = (
        (dx.AbstractArray("B 1 S S", dtype="bool", name="attention_mask"),)
        if case.masked
        else ()
    )
    attend = dx.AbstractModule(
        dot_product_attention,
        *attention_operands,
        name="dot_product_attention",
        kwargs={"is_causal": case.is_causal},
    )
    project_output = dx.AbstractModule(
        output_projection,
        dx.AbstractArray("H K D", dtype="float32", name="o_weight"),
        name="output_projection",
    )
    return hidden_states >> qkv_projection >> attend >> project_output


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_attention_matches_reference_trace(case):
    batch_size = 2
    sequence_length = 8
    hidden_size = 16
    head_dim = 4

    compiled = make_reference(case).compile(
        B=batch_size,
        S=sequence_length,
        D=hidden_size,
        H=case.num_heads,
        G=case.num_kv_heads,
        K=head_dim,
    )
    module = Attention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        bias=False,
        dtype="float32",
        rngs=nn.Rngs(0),
    )

    def actual_attention(hidden_states, attention_mask=None):
        output, _ = module(
            hidden_states,
            attention_mask=attention_mask,
            is_causal=case.is_causal,
        )
        return output

    args = ()
    if case.masked:
        args = (
            jnp.ones(
                (batch_size, 1, sequence_length, sequence_length),
                dtype=jnp.bool_,
            ),
        )

    assert compiled.verify(
        actual_attention,
        *args,
    ), compiled.report(actual_attention, *args).render()


def test_joint_attention_matches_reference_trace():
    stream1 = dx.AbstractArray("B S1 D1", dtype="float32")
    stream2 = dx.AbstractArray("B S2 D2", dtype="float32", name="stream2")
    reference = (
        stream1
        >> dx.AbstractModule(
            joint_qkv_projection,
            stream2,
            dx.AbstractArray("D1 H K", dtype="float32", name="q1_weight"),
            dx.AbstractArray("D1 H K", dtype="float32", name="k1_weight"),
            dx.AbstractArray("D1 H K", dtype="float32", name="v1_weight"),
            dx.AbstractArray("D2 H K", dtype="float32", name="q2_weight"),
            dx.AbstractArray("D2 H K", dtype="float32", name="k2_weight"),
            dx.AbstractArray("D2 H K", dtype="float32", name="v2_weight"),
            name="joint_qkv_projection",
        )
        >> dx.AbstractModule(
            joint_attention,
            name="joint_attention",
            kwargs={"length1": 5},
        )
        >> dx.AbstractModule(
            joint_output_projection,
            dx.AbstractArray("H K D1", dtype="float32", name="o1_weight"),
            dx.AbstractArray("H K D2", dtype="float32", name="o2_weight"),
            name="joint_output_projection",
        )
    )
    compiled = reference.compile(
        B=2,
        S1=5,
        S2=3,
        D1=16,
        D2=12,
        H=4,
        K=4,
    )
    module = JointAttention(
        hidden_size1=16,
        hidden_size2=12,
        num_heads=4,
        head_dim=4,
        rngs=nn.Rngs(0),
    )
    actual_stream2 = jnp.ones((2, 3, 12), dtype=jnp.float32)

    assert compiled.verify(
        module,
        actual_stream2,
    ), compiled.report(module, actual_stream2).render()
