from dataclasses import dataclass

import pytest

from taktiny import nn
from taktiny.cosettes.transformers._ordinario import (
    TransformerCausalLM,
    TransformerConditionalGeneration,
)
from taktiny.cosettes.transformers.gemma import (
    Gemma2DecoderLayer,
    Gemma3DecoderLayer,
    GemmaDecoderLayer,
)
from taktiny.cosettes.transformers.llama import LlamaDecoderLayer
from taktiny.cosettes.transformers.qwen import (
    Qwen2DecoderLayer,
    Qwen3DecoderLayer,
    QwenDecoderLayer,
)
from taktiny.maestro.config import ModelConfig
from taktiny.maestro.opus.deepseek import (
    Deepseek,
    DeepseekV2,
    DeepseekV3,
    DeepseekV3_2,
    DeepseekV4,
)
from taktiny.maestro.opus.gemma import (
    Gemma,
    Gemma2,
    Gemma3,
    Gemma3ConditionalGeneration,
    Gemma4,
    Gemma4Unified,
)
from taktiny.maestro.opus.gpt import GPTOSS
from taktiny.maestro.opus.llama import Llama, Llama4
from taktiny.maestro.opus.qwen import (
    Qwen,
    Qwen2,
    Qwen3,
    Qwen3_5MoE,
    Qwen3MoE,
    Qwen3Next,
)


@dataclass(frozen=True)
class OpusCase:
    opus: type
    base: type
    decoder: type


CASES = (
    OpusCase(Llama, TransformerCausalLM, LlamaDecoderLayer),
    OpusCase(Llama4, TransformerConditionalGeneration, LlamaDecoderLayer),
    OpusCase(Gemma, TransformerCausalLM, GemmaDecoderLayer),
    OpusCase(Gemma2, TransformerCausalLM, Gemma2DecoderLayer),
    OpusCase(Gemma3, TransformerCausalLM, Gemma3DecoderLayer),
    OpusCase(
        Gemma3ConditionalGeneration,
        TransformerConditionalGeneration,
        Gemma3DecoderLayer,
    ),
    OpusCase(Gemma4, TransformerConditionalGeneration, Gemma3DecoderLayer),
    OpusCase(
        Gemma4Unified,
        TransformerConditionalGeneration,
        Gemma3DecoderLayer,
    ),
    OpusCase(Qwen, TransformerCausalLM, QwenDecoderLayer),
    OpusCase(Qwen2, TransformerCausalLM, Qwen2DecoderLayer),
    OpusCase(Qwen3, TransformerCausalLM, Qwen3DecoderLayer),
    OpusCase(Qwen3MoE, TransformerCausalLM, Qwen2DecoderLayer),
    OpusCase(Qwen3Next, TransformerCausalLM, Qwen2DecoderLayer),
    OpusCase(
        Qwen3_5MoE,
        TransformerConditionalGeneration,
        Qwen2DecoderLayer,
    ),
    OpusCase(Deepseek, TransformerCausalLM, LlamaDecoderLayer),
    OpusCase(DeepseekV2, TransformerCausalLM, LlamaDecoderLayer),
    OpusCase(DeepseekV3, TransformerCausalLM, LlamaDecoderLayer),
    OpusCase(DeepseekV3_2, TransformerCausalLM, LlamaDecoderLayer),
    OpusCase(DeepseekV4, TransformerCausalLM, LlamaDecoderLayer),
    OpusCase(GPTOSS, TransformerCausalLM, LlamaDecoderLayer),
)


def opus_config():
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        rope_local_base_freq=10_000.0,
        layer_types=["full_attention"],
        sliding_window=4,
        query_pre_attn_scalar=16.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        no_bias=True,
        seq_length=64,
        use_dynamic_ntk=True,
        use_logn_attn=True,
        tie_word_embeddings=False,
        dtype="float32",
    )


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.opus.__name__,
)
def test_opus_routes_to_expected_decoder(case, monkeypatch):
    captured = {}

    def capture_init(self, config=None, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(case.base, "__init__", capture_init)
    case.opus(opus_config(), rngs=nn.Rngs(0))

    assert captured["decoder"] is case.decoder
