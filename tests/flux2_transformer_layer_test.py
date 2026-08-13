import jax
import jax.numpy as jnp
import numpy as np

from taktiny import layers as ly
from taktiny import nn
from taktiny.cosettes.transformers.flux import (
    Flux2Modulation,
    Flux2RotaryEmbedding,
    Flux2SingleTransformerLayer,
    Flux2TransformerLayer,
)
from taktiny.cosettes.transformers._ordinario import (
    GatedParallelTransformerLayer,
    JointTransformerLayer,
)
from taktiny.maestro.config import ModelConfig


def _config(**overrides):
    values = {
        'num_attention_heads': 2,
        'attention_head_dim': 4,
        'mlp_ratio': 2.0,
        'axes_dims_rope': (2, 2),
        'rope_theta': 2000.0,
        'eps': 1e-6,
        'dtype': 'float32',
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_flux2_modulation_shapes_and_split():
    module = Flux2Modulation(8, 2, rngs=nn.Rngs(0))
    output = module(jnp.ones((3, 8), dtype=jnp.float32))

    assert output.shape == (3, 48)
    groups = module.split(output)
    assert len(groups) == 2
    assert all(value.shape == (3, 1, 8) for group in groups for value in group)


def test_flux2_layers_use_general_transformer_principles():
    assert issubclass(Flux2TransformerLayer, JointTransformerLayer)
    assert issubclass(
        Flux2SingleTransformerLayer,
        GatedParallelTransformerLayer,
    )

    double = Flux2TransformerLayer(_config(), rngs=nn.Rngs(0))
    single = Flux2SingleTransformerLayer(_config(), rngs=nn.Rngs(0))
    assert double.norm1.linear is None
    assert double.norm1_context.linear is None
    assert single.norm.linear is None


def test_flux2_rotary_embedding_uses_multiple_position_axes():
    rotary = Flux2RotaryEmbedding(theta=2000.0, axes_dim=(2, 2))
    query = jnp.arange(1, 17, dtype=jnp.float32).reshape(1, 2, 2, 4)
    key = query + 1

    zero_ids = jnp.zeros((2, 2), dtype=jnp.int32)
    zero_query, zero_key = rotary(query, key, zero_ids)
    np.testing.assert_array_equal(zero_query, query)
    np.testing.assert_array_equal(zero_key, key)

    position_ids = jnp.asarray(((0, 0), (1, 2)), dtype=jnp.int32)
    rotated_query, rotated_key = rotary(query, key, position_ids)
    np.testing.assert_array_equal(rotated_query[:, :1], query[:, :1])
    np.testing.assert_array_equal(rotated_key[:, :1], key[:, :1])
    assert not np.array_equal(rotated_query[:, 1:], query[:, 1:])
    assert not np.array_equal(rotated_key[:, 1:], key[:, 1:])


def test_flux2_double_stream_zero_gates_are_identity():
    layer = Flux2TransformerLayer(_config(), rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 3, 8))
    enc_x = jax.random.normal(jax.random.key(2), (2, 2, 8))
    modulation = jnp.zeros((2, 48), dtype=jnp.float32)

    output_enc, output = layer(x, enc_x, modulation, modulation)

    np.testing.assert_array_equal(output, x)
    np.testing.assert_array_equal(output_enc, enc_x)


def test_flux2_double_stream_accepts_tokenwise_precomputed_modulation():
    layer = Flux2TransformerLayer(_config(), rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(21), (2, 3, 8))
    enc_x = jax.random.normal(jax.random.key(22), (2, 2, 8))
    image_modulation = jnp.zeros((2, 3, 48), dtype=jnp.float32)
    text_modulation = jnp.zeros((2, 2, 48), dtype=jnp.float32)

    output_enc, output = layer(
        x,
        enc_x,
        image_modulation,
        text_modulation,
    )

    np.testing.assert_array_equal(output, x)
    np.testing.assert_array_equal(output_enc, enc_x)


def test_flux2_double_stream_matches_explicit_residual_equations():
    layer = Flux2TransformerLayer(
        _config(axes_dims_rope=None),
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(3), (2, 3, 8))
    enc_x = jax.random.normal(jax.random.key(4), (2, 2, 8))
    image_modulation = jax.random.normal(jax.random.key(5), (2, 48))
    text_modulation = jax.random.normal(jax.random.key(6), (2, 48))

    image_groups = tuple(
        jnp.split(image_modulation[:, None, :], 6, axis=-1)
    )
    text_groups = tuple(
        jnp.split(text_modulation[:, None, :], 6, axis=-1)
    )
    image_shift, image_scale, image_attn_gate = image_groups[:3]
    image_ff_shift, image_ff_scale, image_ff_gate = image_groups[3:]
    text_shift, text_scale, text_attn_gate = text_groups[:3]
    text_ff_shift, text_ff_scale, text_ff_gate = text_groups[3:]

    image_base, _ = layer.norm1(x, image_modulation)
    normalized_x = (1 + image_scale) * image_base + image_shift
    text_base, _ = layer.norm1_context(enc_x, text_modulation)
    normalized_enc = (
        (1 + text_scale) * text_base + text_shift
    )
    x_attention, enc_attention = layer.attn(normalized_x, normalized_enc)
    expected_x = x + image_attn_gate * x_attention
    expected_enc = enc_x + text_attn_gate * enc_attention
    expected_x = expected_x + image_ff_gate * layer.ff(
        (1 + image_ff_scale) * layer.norm2(expected_x) + image_ff_shift
    )
    expected_enc = expected_enc + text_ff_gate * layer.ff_context(
        (1 + text_ff_scale) * layer.norm2_context(expected_enc) + text_ff_shift
    )

    output_enc, output = layer(
        x,
        enc_x,
        image_modulation,
        text_modulation,
    )
    np.testing.assert_allclose(output, expected_x, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        output_enc,
        expected_enc,
        rtol=1e-6,
        atol=1e-6,
    )


def test_flux2_joint_attention_uses_text_then_image_mask_order():
    layer = Flux2TransformerLayer(
        _config(axes_dims_rope=None),
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(11), (1, 2, 8))
    enc_x = jax.random.normal(jax.random.key(12), (1, 3, 8))
    attention_mask = jnp.zeros((1, 1, 5, 5), dtype=jnp.bool_)
    attention_mask = attention_mask.at[..., 0].set(True)

    image_q = layer.attn.q_norm_1(layer.attn.q_proj_1(x))
    image_k = layer.attn.k_norm_1(layer.attn.k_proj_1(x))
    image_v = layer.attn.v_proj_1(x)
    text_q = layer.attn.q_norm_2(layer.attn.q_proj_2(enc_x))
    text_k = layer.attn.k_norm_2(layer.attn.k_proj_2(enc_x))
    text_v = layer.attn.v_proj_2(enc_x)
    output = ly.Attention.apply(
        jnp.concatenate((text_q, image_q), axis=1),
        jnp.concatenate((text_k, image_k), axis=1),
        jnp.concatenate((text_v, image_v), axis=1),
        mask=attention_mask,
    )
    expected_text, expected_image = jnp.split(output, (3,), axis=1)
    expected_image = layer.attn.o_proj_1(expected_image)
    expected_text = layer.attn.o_proj_2(expected_text)

    image_output, text_output = layer.attn(
        x,
        enc_x,
        attention_mask=attention_mask,
    )
    np.testing.assert_allclose(
        image_output,
        expected_image,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        text_output,
        expected_text,
        rtol=1e-6,
        atol=1e-6,
    )


def test_flux2_single_stream_zero_gate_is_identity_and_can_split():
    layer = Flux2SingleTransformerLayer(_config(), rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(7), (2, 3, 8))
    enc_x = jax.random.normal(jax.random.key(8), (2, 2, 8))
    modulation = jnp.zeros((2, 24), dtype=jnp.float32)
    image_positions = jnp.asarray(((0, 0), (1, 1), (2, 1)))
    text_positions = jnp.asarray(((0, 0), (0, 1)))

    output_enc, output = layer(
        x,
        enc_x,
        modulation,
        position_idx=image_positions,
        encoder_position_idx=text_positions,
        split_hidden_states=True,
    )

    np.testing.assert_array_equal(output, x)
    np.testing.assert_array_equal(output_enc, enc_x)


def test_flux2_single_stream_accepts_tokenwise_precomputed_modulation():
    layer = Flux2SingleTransformerLayer(_config(), rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(23), (2, 4, 8))
    modulation = jnp.zeros((2, 4, 24), dtype=jnp.float32)

    output = layer(x, None, modulation)

    np.testing.assert_array_equal(output, x)


def test_flux2_single_stream_matches_parallel_residual_equation():
    layer = Flux2SingleTransformerLayer(
        _config(axes_dims_rope=None),
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(9), (2, 4, 8))
    modulation = jax.random.normal(jax.random.key(10), (2, 24))
    shift, scale, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
    normalized_base, _ = layer.norm(x, modulation)
    normalized = (1 + scale) * normalized_base + shift
    expected = x + gate * layer.attn(normalized)

    output = layer(x, None, modulation)

    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_flux2_single_stream_fuses_attention_and_mlp_projections():
    layer = Flux2SingleTransformerLayer(_config(), rngs=nn.Rngs(0))

    assert layer.attn.to_qkv_mlp_proj.weight.shape == (8, 56)
    assert layer.attn.to_out.weight.shape == (24, 8)
