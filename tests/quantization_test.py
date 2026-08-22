from types import SimpleNamespace

import jax.numpy as jnp
import qwix

from taktiny.utils.quantization import (
    quantization_rules,
    quantize_linear_weight,
    resolve_quantization_rule,
)


def test_int4_shortcut_uses_grouped_linear_weight_quantization():
    rule, = quantization_rules('int4')

    assert rule.op_names == ('dot_general',)
    assert rule.weight_qtype == 'int4'
    assert rule.tile_size == 128
    assert resolve_quantization_rule(
        'int4',
        'model.layers.0.attention.q_proj',
        op_name='dot_general',
    ) == rule
    assert resolve_quantization_rule(
        'int4',
        'model.token_embedding',
        op_name='embedding',
    ) is None


def test_explicit_rule_can_quantize_embedding():
    rule = qwix.QuantizationRule(
        op_names=('embedding',),
        weight_qtype='int4',
        tile_size=64,
    )

    assert resolve_quantization_rule(
        rule,
        'model.token_embedding',
        op_name='embedding',
    ) == rule


def test_grouped_shortcut_falls_back_for_small_linear_axis():
    parameter = SimpleNamespace(
        dtype=jnp.bfloat16,
        input_axis_count=1,
        quantization_batch_axis_count=0,
    )
    rule, = quantization_rules('int4')

    quantized = quantize_linear_weight(
        jnp.arange(12, dtype=jnp.bfloat16).reshape(4, 3),
        parameter,
        rule,
    )

    assert isinstance(quantized, qwix.QArray)
    assert quantized.shape == (4, 3)
