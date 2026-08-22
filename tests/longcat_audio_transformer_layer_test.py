import jax
import jax.numpy as jnp
import numpy as np

from taktiny import nn
from taktiny.cosettes.transformers.ordinario import (
    ConditionalTransformerLayer,
)
from taktiny.cosettes.transformers.longcat import (
    LongCatAudioTransformerLayer,
)
from taktiny.maestro.config import ModelConfig


def _config(**overrides):
    values = {
        'dit_dim': 8,
        'dit_heads': 2,
        'ff_mult': 2.0,
        'dropout': 0.0,
        'bias': True,
        'qk_norm': True,
        'cross_attn': True,
        'cross_attn_norm': False,
        'adaln_type': 'global',
        'adaln_use_text_cond': True,
        'dtype': 'float32',
    }
    values.update(overrides)
    return ModelConfig(**values)


def _mask_output(value, mask):
    if mask is None:
        return value
    return value * mask[..., None].astype(value.dtype)


def test_longcat_audio_uses_conditional_transformer_principle():
    layer = LongCatAudioTransformerLayer(_config(), rngs=nn.Rngs(0))

    assert isinstance(layer, ConditionalTransformerLayer)
    assert layer.attn1.qk_norm_across_heads
    assert layer.attn2.qk_norm_across_heads
    assert layer.audio_position_embedding.base == 100_000.0
    assert layer.prompt_position_embedding.base == 100_000.0
    assert layer.scale_shift_table.shape == (6, 8)


def test_longcat_audio_global_path_matches_sequential_reference():
    layer = LongCatAudioTransformerLayer(_config(), rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 4, 8))
    enc_x = jax.random.normal(jax.random.key(2), (2, 3, 8))
    conditioning = jax.random.normal(jax.random.key(3), (2, 48))
    attention_mask = jnp.asarray(
        ((True, True, True, False), (True, True, True, True))
    )
    encoder_attention_mask = jnp.asarray(
        ((True, True, False), (True, True, True))
    )

    modulation = conditioning.reshape(2, 6, 8)
    modulation = modulation + layer.scale_shift_table.value[None]
    gate_sa, scale_sa, shift_sa, gate_ff, scale_ff, shift_ff = (
        modulation[:, index] for index in range(6)
    )

    normalized = layer.norm1(x)
    normalized = normalized * (1 + scale_sa[:, None]) + shift_sa[:, None]
    self_attention = layer.attn1(
        normalized,
        attention_mask=attention_mask[:, None, None, :],
    )[0]
    self_attention = _mask_output(self_attention, attention_mask)
    expected = x + gate_sa[:, None] * self_attention

    cross_attention = layer._cross_attention(
        expected,
        enc_x,
        attention_mask=encoder_attention_mask,
        position_idx=None,
        encoder_position_idx=None,
        out_sharding=None,
        kernel='dot_product',
    )
    cross_attention = _mask_output(cross_attention, attention_mask)
    expected = expected + cross_attention

    normalized = layer.norm2(expected)
    normalized = normalized * (1 + scale_ff[:, None]) + shift_ff[:, None]
    expected = expected + gate_ff[:, None] * layer.ff(normalized)

    output = layer(
        x,
        enc_x,
        conditioning,
        attention_mask=attention_mask,
        encoder_attention_mask=encoder_attention_mask,
    )
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_longcat_audio_local_adaln_projection_is_zero_initialized():
    layer = LongCatAudioTransformerLayer(
        _config(adaln_type='local'),
        rngs=nn.Rngs(0),
    )

    assert not hasattr(layer, 'scale_shift_table')
    np.testing.assert_array_equal(layer.adaln_mlp.linear.weight.value, 0)
    np.testing.assert_array_equal(layer.adaln_mlp.linear.bias.value, 0)

    x = jnp.ones((2, 4, 8), dtype=jnp.float32)
    enc_x = jnp.ones((2, 3, 8), dtype=jnp.float32)
    timestep = jnp.ones((2, 8), dtype=jnp.float32)
    modulation = layer._audio_modulation(
        timestep,
        enc_x,
        jnp.asarray(((True, True, False), (True, True, True))),
    )
    for value in modulation:
        np.testing.assert_array_equal(value, 0)


def test_longcat_audio_applies_distinct_audio_and_prompt_positions():
    layer = LongCatAudioTransformerLayer(_config(), rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(4), (1, 4, 8))
    enc_x = jax.random.normal(jax.random.key(5), (1, 3, 8))
    conditioning = jnp.zeros((1, 48), dtype=jnp.float32)

    default = layer(x, enc_x, conditioning)
    shifted = layer(
        x,
        enc_x,
        conditioning,
        position_idx=jnp.asarray(((3, 4, 5, 6),)),
        encoder_position_idx=jnp.asarray(((7, 8, 9),)),
    )
    assert not np.allclose(default, shifted)


def test_longcat_audio_is_jittable_in_global_and_local_modes():
    x = jnp.ones((1, 4, 8), dtype=jnp.float32)
    enc_x = jnp.ones((1, 3, 8), dtype=jnp.float32)
    mask = jnp.ones((1, 4), dtype=jnp.bool_)
    enc_mask = jnp.ones((1, 3), dtype=jnp.bool_)

    global_layer = LongCatAudioTransformerLayer(
        _config(),
        rngs=nn.Rngs(0),
    )
    global_output = jax.jit(global_layer)(
        x,
        enc_x,
        jnp.zeros((1, 48), dtype=jnp.float32),
        attention_mask=mask,
        encoder_attention_mask=enc_mask,
    )
    local_layer = LongCatAudioTransformerLayer(
        _config(adaln_type='local'),
        rngs=nn.Rngs(1),
    )
    local_output = jax.jit(local_layer)(
        x,
        enc_x,
        jnp.zeros((1, 8), dtype=jnp.float32),
        attention_mask=mask,
        encoder_attention_mask=enc_mask,
    )

    assert global_output.shape == local_output.shape == x.shape
