from dataclasses import dataclass

import dextiny as dx
import pytest

from taktiny import nn
from taktiny.cosettes.transformers.qwen import (
    Qwen2DecoderLayer,
    Qwen3DecoderLayer,
    QwenDecoderLayer,
)
from taktiny.maestro.config import ModelConfig

from ._reference import (
    attention_qkv_bias,
    attention_with_qk_norm,
    begin_norm,
    gate_mlp_residual,
    qwen_mlp_residual,
    qwen_rotary,
    render_and_assert,
    residual_then_norm,
    rms_norm,
)


@dataclass(frozen=True)
class QwenCase:
    name: str
    decoder: type
    qk_norm: bool = False
    legacy: bool = False


CASES = (
    QwenCase("qwen", QwenDecoderLayer, legacy=True),
    QwenCase("qwen2", Qwen2DecoderLayer),
    QwenCase("qwen3", Qwen3DecoderLayer, qk_norm=True),
)


def qkv_bias_operands():
    return (
        dx.AbstractArray("D H K", dtype="float32", name="q_weight"),
        dx.AbstractArray("H K", dtype="float32", name="q_bias"),
        dx.AbstractArray("D G K", dtype="float32", name="k_weight"),
        dx.AbstractArray("G K", dtype="float32", name="k_bias"),
        dx.AbstractArray("D G K", dtype="float32", name="v_weight"),
        dx.AbstractArray("G K", dtype="float32", name="v_bias"),
        dx.AbstractArray("H K D", dtype="float32", name="o_weight"),
    )


def qk_norm_operands():
    return (
        dx.AbstractArray("D H K", dtype="float32", name="q_weight"),
        dx.AbstractArray("D G K", dtype="float32", name="k_weight"),
        dx.AbstractArray("D G K", dtype="float32", name="v_weight"),
        dx.AbstractArray("H K D", dtype="float32", name="o_weight"),
        dx.AbstractArray("K", dtype="float32", name="q_norm_weight"),
        dx.AbstractArray("K", dtype="float32", name="k_norm_weight"),
    )


def qwen_reference(
    case,
    *,
    eps,
    head_dim,
    num_heads,
    num_kv_heads,
    rope_theta,
    max_position_embeddings,
    sequence_length,
):
    if case.qk_norm:
        attention = attention_with_qk_norm
        operands = qk_norm_operands()
    else:
        attention = attention_qkv_bias
        operands = qkv_bias_operands()

    attention_kwargs = {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "rope_theta": rope_theta,
        "max_position_embeddings": max_position_embeddings,
    }
    if case.qk_norm:
        attention_kwargs.update(qk_norm=rms_norm, norm_eps=eps)
    if case.legacy:
        attention_kwargs.update(
            rotary_fn=qwen_rotary,
            use_dynamic_ntk=True,
            use_logn_attn=True,
            sequence_length=sequence_length,
        )

    reference = (
        dx.AbstractArray("B S D", dtype="float32")
        >> dx.AbstractModule(
            begin_norm,
            dx.AbstractArray("D", dtype="float32", name="input_norm_weight"),
            name="input_rms_norm",
            kwargs={"eps": eps, "norm": rms_norm},
        )
        >> dx.AbstractModule(
            attention,
            *operands,
            name="causal_self_attention",
            kwargs=attention_kwargs,
        )
        >> dx.AbstractModule(
            residual_then_norm,
            dx.AbstractArray(
                "D",
                dtype="float32",
                name="post_attention_norm_weight",
            ),
            name="attention_residual_then_rms_norm",
            kwargs={"eps": eps, "norm": rms_norm},
        )
    )
    if case.legacy:
        return reference >> dx.AbstractModule(
            qwen_mlp_residual,
            dx.AbstractArray("D F", dtype="float32", name="w1_weight"),
            dx.AbstractArray("D F", dtype="float32", name="w2_weight"),
            dx.AbstractArray("F D", dtype="float32", name="c_proj_weight"),
            name="qwen_gated_mlp_residual",
        )
    return reference >> dx.AbstractModule(
        gate_mlp_residual,
        dx.AbstractArray("D I", dtype="float32", name="gate_weight"),
        dx.AbstractArray("D I", dtype="float32", name="up_weight"),
        dx.AbstractArray("I D", dtype="float32", name="down_weight"),
        name="gated_mlp_residual",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_qwen_decoder_layer_matches_reference_trace(case):
    hidden_size = 16
    feedforward_size = 32
    intermediate_size = feedforward_size * 2 if case.legacy else 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 4
    eps = 1e-6
    rope_theta = 10_000.0
    max_position_embeddings = 4 if case.legacy else 64
    sequence_length = 4 if case.legacy else max_position_embeddings

    dimensions = {
        "B": 2,
        "S": 8,
        "D": hidden_size,
        "H": num_heads,
        "G": num_kv_heads,
        "K": head_dim,
    }
    dimensions["F" if case.legacy else "I"] = feedforward_size
    compiled = qwen_reference(
        case,
        eps=eps,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        sequence_length=sequence_length,
    ).compile(**dimensions)
    config = ModelConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rms_norm_eps=eps,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        no_bias=True,
        seq_length=sequence_length,
        use_dynamic_ntk=True,
        use_logn_attn=True,
        dtype="float32",
    )
    layer = case.decoder(config, rngs=nn.Rngs(0), layer_idx=0)

    def actual_block(hidden_states):
        output, _ = layer(hidden_states, is_causal=True)
        return output

    render_and_assert(compiled, actual_block)
