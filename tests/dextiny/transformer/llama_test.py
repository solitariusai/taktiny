import dextiny as dx
import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.cosettes.transformers.llama import LlamaDecoderLayer
from taktiny.maestro.config import ModelConfig


def linear(x, weight, input_dims=1):
    x_axes = tuple(range(x.ndim - input_dims, x.ndim))
    weight_axes = tuple(range(input_dims))
    return jax.lax.dot_general(
        x,
        weight,
        ((x_axes, weight_axes), ((), ())),
    )


def rms_norm(x, weight, *, eps):
    dtype = x.dtype
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normalized = x * jax.lax.rsqrt(variance + eps)
    return (normalized * weight).astype(dtype)


def input_norm(hidden_states, weight, *, eps):
    return hidden_states, rms_norm(hidden_states, weight, eps=eps)


def rotate_half(x):
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def apply_rotary(query, key, *, head_dim, rope_theta):
    sequence_length = query.shape[1]
    inv_freq = 1.0 / (
        rope_theta
        ** (
            jnp.arange(0, head_dim, 2, dtype=jnp.float32)
            / head_dim
        )
    )
    positions = jnp.arange(sequence_length, dtype=jnp.float32)
    frequencies = jnp.einsum("s,d->sd", positions, inv_freq)
    embedding = jnp.concatenate((frequencies, frequencies), axis=-1)
    cosine = jnp.cos(embedding)[None, :, None, :].astype(query.dtype)
    sine = jnp.sin(embedding)[None, :, None, :].astype(query.dtype)
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


def causal_self_attention(
    state,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    *,
    head_dim,
    rope_theta,
):
    residual, hidden_states = state
    query = linear(hidden_states, q_weight)
    key = linear(hidden_states, k_weight)
    value = linear(hidden_states, v_weight)
    query, key = apply_rotary(
        query,
        key,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )
    attended = jax.nn.dot_product_attention(
        query=query,
        key=key,
        value=value,
        is_causal=True,
    )
    return residual, linear(attended, o_weight, input_dims=2)


def post_attention_norm(state, weight, *, eps):
    residual, attention_output = state
    hidden_states = residual + attention_output
    return hidden_states, rms_norm(hidden_states, weight, eps=eps)


def gated_mlp_residual(
    state,
    gate_weight,
    up_weight,
    down_weight,
):
    residual, hidden_states = state
    gate = jax.nn.silu(linear(hidden_states, gate_weight))
    up = linear(hidden_states, up_weight)
    return residual + linear(gate * up, down_weight)


def llama_block_reference(*, eps, head_dim, rope_theta):
    hidden_states = dx.AbstractArray("B S D", dtype="float32")
    return (
        hidden_states
        >> dx.AbstractModule(
            input_norm,
            dx.AbstractArray("D", dtype="float32", name="input_norm_weight"),
            name="input_rms_norm",
            kwargs={"eps": eps},
        )
        >> dx.AbstractModule(
            causal_self_attention,
            dx.AbstractArray("D H K", dtype="float32", name="q_weight"),
            dx.AbstractArray("D G K", dtype="float32", name="k_weight"),
            dx.AbstractArray("D G K", dtype="float32", name="v_weight"),
            dx.AbstractArray("H K D", dtype="float32", name="o_weight"),
            name="causal_self_attention",
            kwargs={"head_dim": head_dim, "rope_theta": rope_theta},
        )
        >> dx.AbstractModule(
            post_attention_norm,
            dx.AbstractArray("D", dtype="float32", name="post_norm_weight"),
            name="post_attention_rms_norm",
            kwargs={"eps": eps},
        )
        >> dx.AbstractModule(
            gated_mlp_residual,
            dx.AbstractArray("D I", dtype="float32", name="gate_weight"),
            dx.AbstractArray("D I", dtype="float32", name="up_weight"),
            dx.AbstractArray("I D", dtype="float32", name="down_weight"),
            name="gated_mlp_residual",
        )
    )


def test_llama_decoder_layer_matches_reference_trace():
    hidden_size = 16
    intermediate_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 4
    eps = 1e-6
    rope_theta = 10000.0

    compiled = llama_block_reference(
        eps=eps,
        head_dim=head_dim,
        rope_theta=rope_theta,
    ).compile(
        B=2,
        S=8,
        D=hidden_size,
        I=intermediate_size,
        H=num_heads,
        G=num_kv_heads,
        K=head_dim,
    )
    config = ModelConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=64,
        rope_theta=rope_theta,
        rms_norm_eps=eps,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        dtype="float32",
    )
    layer = LlamaDecoderLayer(config, rngs=nn.Rngs(0), layer_idx=0)

    def actual_block(hidden_states):
        output, _ = layer(hidden_states, is_causal=True)
        return output

    report = compiled.report(actual_block)
    print(f"\n{report.render()}")
    assert report.valid, report.render()
