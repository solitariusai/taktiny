from dataclasses import dataclass

import dextiny as dx
import pytest

from taktiny import nn
from taktiny.cosettes.transformers.gemma import (
    Gemma2DecoderLayer,
    Gemma3DecoderLayer,
    GemmaDecoderLayer,
)
from taktiny.maestro.config import ModelConfig

from ._reference import (
    attention_no_bias,
    attention_with_qk_norm,
    begin_norm,
    gate_mlp_branch,
    gate_mlp_residual,
    gemma3_norm,
    gemma_norm,
    norm_then_residual,
    render_and_assert,
    residual_then_norm,
)


@dataclass(frozen=True)
class GemmaCase:
    name: str
    decoder: type
    sandwich_norms: bool
    qk_norm: bool
    layer_type: str = "full_attention"


CASES = (
    GemmaCase("gemma", GemmaDecoderLayer, False, False),
    GemmaCase(
        "gemma2_local",
        Gemma2DecoderLayer,
        True,
        False,
        "sliding_attention",
    ),
    GemmaCase("gemma2_global", Gemma2DecoderLayer, True, False),
    GemmaCase(
        "gemma3_local",
        Gemma3DecoderLayer,
        True,
        True,
        "sliding_attention",
    ),
    GemmaCase("gemma3_global", Gemma3DecoderLayer, True, True),
)


def attention_operands(*, qk_norm):
    operands = [
        dx.AbstractArray("D H K", dtype="float32", name="q_weight"),
        dx.AbstractArray("D G K", dtype="float32", name="k_weight"),
        dx.AbstractArray("D G K", dtype="float32", name="v_weight"),
        dx.AbstractArray("H K D", dtype="float32", name="o_weight"),
    ]
    if qk_norm:
        operands.extend(
            (
                dx.AbstractArray("K", dtype="float32", name="q_norm_weight"),
                dx.AbstractArray("K", dtype="float32", name="k_norm_weight"),
            )
        )
    return operands


def gemma_reference(
    case,
    *,
    eps,
    head_dim,
    num_heads,
    num_kv_heads,
    rope_theta,
    max_position_embeddings,
    window_size,
    scaling,
    softcap,
):
    norm = gemma3_norm if case.qk_norm else gemma_norm
    attention = attention_with_qk_norm if case.qk_norm else attention_no_bias
    attention_kwargs = {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "rope_theta": rope_theta,
        "max_position_embeddings": max_position_embeddings,
        "window_size": window_size,
        "scaling": scaling,
        "softcap": softcap,
    }
    if case.qk_norm:
        attention_kwargs.update(qk_norm=gemma3_norm, norm_eps=eps)

    reference = (
        dx.AbstractArray("B S D", dtype="float32")
        >> dx.AbstractModule(
            begin_norm,
            dx.AbstractArray("D", dtype="float32", name="input_norm_weight"),
            name="input_rms_norm",
            kwargs={"eps": eps, "norm": norm},
        )
        >> dx.AbstractModule(
            attention,
            *attention_operands(qk_norm=case.qk_norm),
            name="causal_self_attention",
            kwargs=attention_kwargs,
        )
    )

    if not case.sandwich_norms:
        return (
            reference
            >> dx.AbstractModule(
                residual_then_norm,
                dx.AbstractArray(
                    "D",
                    dtype="float32",
                    name="post_attention_norm_weight",
                ),
                name="attention_residual_then_rms_norm",
                kwargs={"eps": eps, "norm": norm},
            )
            >> dx.AbstractModule(
                gate_mlp_residual,
                dx.AbstractArray("D I", dtype="float32", name="gate_weight"),
                dx.AbstractArray("D I", dtype="float32", name="up_weight"),
                dx.AbstractArray("I D", dtype="float32", name="down_weight"),
                name="gated_mlp_residual",
            )
        )

    return (
        reference
        >> dx.AbstractModule(
            norm_then_residual,
            dx.AbstractArray(
                "D",
                dtype="float32",
                name="post_attention_norm_weight",
            ),
            name="post_attention_norm_then_residual",
            kwargs={"eps": eps, "norm": norm},
        )
        >> dx.AbstractModule(
            begin_norm,
            dx.AbstractArray(
                "D",
                dtype="float32",
                name="pre_feedforward_norm_weight",
            ),
            name="pre_feedforward_rms_norm",
            kwargs={"eps": eps, "norm": norm},
        )
        >> dx.AbstractModule(
            gate_mlp_branch,
            dx.AbstractArray("D I", dtype="float32", name="gate_weight"),
            dx.AbstractArray("D I", dtype="float32", name="up_weight"),
            dx.AbstractArray("I D", dtype="float32", name="down_weight"),
            name="gated_mlp",
        )
        >> dx.AbstractModule(
            norm_then_residual,
            dx.AbstractArray(
                "D",
                dtype="float32",
                name="post_feedforward_norm_weight",
            ),
            name="post_feedforward_norm_then_residual",
            kwargs={"eps": eps, "norm": norm},
        )
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_gemma_decoder_layer_matches_reference_trace(case):
    hidden_size = 16
    intermediate_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 4
    eps = 1e-6
    max_position_embeddings = 64
    local_window = 4
    scaling = 16.0**-0.5 if case.sandwich_norms else None
    softcap = 5.0 if case.sandwich_norms else None
    rope_theta = 10_000.0
    if case.qk_norm and case.layer_type == "full_attention":
        rope_theta = 1_000_000.0
    if case.decoder is GemmaDecoderLayer:
        rope_theta = 10_000.0
        window_size = None
    elif case.layer_type == "sliding_attention":
        window_size = local_window
    else:
        window_size = max_position_embeddings

    compiled = gemma_reference(
        case,
        eps=eps,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        window_size=window_size,
        scaling=scaling,
        softcap=softcap,
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
        max_position_embeddings=max_position_embeddings,
        rope_theta=1_000_000.0 if case.qk_norm else 10_000.0,
        rope_local_base_freq=10_000.0,
        layer_types=[case.layer_type],
        sliding_window=local_window,
        query_pre_attn_scalar=16.0 if case.sandwich_norms else None,
        attn_logit_softcapping=softcap,
        rms_norm_eps=eps,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        dtype="float32",
    )
    layer = case.decoder(config, rngs=nn.Rngs(0), layer_idx=0)

    def actual_block(hidden_states):
        output, _ = layer(hidden_states, is_causal=True)
        return output

    render_and_assert(compiled, actual_block)
