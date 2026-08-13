import jax
import jax.numpy as jnp
import numpy as np

from taktiny import layers as ly
from taktiny import nn
from taktiny.cosettes._continuo import combine_joint_positions
from taktiny.cosettes.transformers._ordinario import (
    GatedParallelTransformerLayer,
    JointTransformerLayer,
)
from taktiny.cosettes.transformers.longcat import (
    LongCatImageSingleTransformerLayer,
    LongCatImageTransformerLayer,
)
from taktiny.maestro.config import ModelConfig


def _config(**overrides):
    values = {
        'num_attention_heads': 2,
        'attention_head_dim': 4,
        'mlp_ratio': 2.0,
        'axes_dims_rope': (2, 2),
        'rope_theta': 10_000.0,
        'dtype': 'float32',
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_longcat_layers_use_joint_and_gated_parallel_principles():
    assert issubclass(LongCatImageTransformerLayer, JointTransformerLayer)
    assert issubclass(
        LongCatImageSingleTransformerLayer,
        GatedParallelTransformerLayer,
    )

    double = LongCatImageTransformerLayer(_config(), rngs=nn.Rngs(0))
    single = LongCatImageSingleTransformerLayer(_config(), rngs=nn.Rngs(0))
    assert double.norm1.linear is not None
    assert double.norm1_context.linear is not None
    assert double.attn.context_first
    assert single.norm.linear is not None


def test_longcat_double_adapter_matches_general_joint_layer():
    layer = LongCatImageTransformerLayer(_config(), rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 3, 8))
    enc_x = jax.random.normal(jax.random.key(2), (2, 2, 8))
    conditioning = jax.random.normal(jax.random.key(3), (2, 8))
    image_ids = jnp.asarray(((0, 0), (1, 0), (2, 0)))
    text_ids = jnp.asarray(((0, 0), (0, 1)))
    joint_ids = combine_joint_positions(text_ids, image_ids, batch_size=2)

    expected_enc, expected = JointTransformerLayer.__call__(
        layer,
        x,
        enc_x,
        conditioning,
        context_conditioning=conditioning,
        position_idx=joint_ids,
    )
    output_enc, output = layer(
        x,
        enc_x,
        conditioning,
        position_idx=image_ids,
        encoder_position_idx=text_ids,
    )

    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output_enc, expected_enc, rtol=1e-6, atol=1e-6)


def test_longcat_single_matches_gated_parallel_residual_and_split():
    layer = LongCatImageSingleTransformerLayer(
        _config(axes_dims_rope=None),
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(4), (2, 3, 8))
    enc_x = jax.random.normal(jax.random.key(5), (2, 2, 8))
    conditioning = jax.random.normal(jax.random.key(6), (2, 8))
    combined = jnp.concatenate((enc_x, x), axis=1)

    normalized_base, modulation = layer.norm(combined, conditioning)
    shift, scale, gate = jnp.split(modulation, 3, axis=-1)
    normalized = normalized_base * (1 + scale[:, None]) + shift[:, None]
    expected = combined + gate[:, None] * layer.attn(normalized)
    expected_enc, expected_x = jnp.split(expected, (enc_x.shape[1],), axis=1)

    output_enc, output = layer(x, enc_x, conditioning)
    np.testing.assert_allclose(output, expected_x, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output_enc, expected_enc, rtol=1e-6, atol=1e-6)


def test_longcat_joint_attention_uses_text_then_image_token_order():
    layer = LongCatImageTransformerLayer(
        _config(axes_dims_rope=None),
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(11), (1, 2, 8))
    enc_x = jax.random.normal(jax.random.key(12), (1, 3, 8))
    mask = jnp.zeros((1, 1, 5, 5), dtype=jnp.bool_)
    mask = mask.at[..., 0].set(True)

    image_q = layer.attn.q_norm_1(layer.attn.q_proj_1(x))
    image_k = layer.attn.k_norm_1(layer.attn.k_proj_1(x))
    image_v = layer.attn.v_proj_1(x)
    text_q = layer.attn.q_norm_2(layer.attn.q_proj_2(enc_x))
    text_k = layer.attn.k_norm_2(layer.attn.k_proj_2(enc_x))
    text_v = layer.attn.v_proj_2(enc_x)
    combined = ly.Attention.apply(
        jnp.concatenate((text_q, image_q), axis=1),
        jnp.concatenate((text_k, image_k), axis=1),
        jnp.concatenate((text_v, image_v), axis=1),
        mask=mask,
    )
    expected_text, expected_image = jnp.split(combined, (3,), axis=1)

    image_output, text_output = layer.attn(x, enc_x, attention_mask=mask)
    np.testing.assert_allclose(
        image_output,
        layer.attn.o_proj_1(expected_image),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        text_output,
        layer.attn.o_proj_2(expected_text),
        rtol=1e-6,
        atol=1e-6,
    )


def test_longcat_layers_are_jittable_with_joint_positions():
    double = LongCatImageTransformerLayer(_config(), rngs=nn.Rngs(0))
    single = LongCatImageSingleTransformerLayer(_config(), rngs=nn.Rngs(1))
    x = jnp.ones((1, 3, 8), dtype=jnp.float32)
    enc_x = jnp.ones((1, 2, 8), dtype=jnp.float32)
    conditioning = jnp.ones((1, 8), dtype=jnp.float32)
    image_ids = jnp.asarray(((0, 0), (1, 0), (2, 0)))
    text_ids = jnp.asarray(((0, 0), (0, 1)))

    output_enc, output = jax.jit(double)(
        x,
        enc_x,
        conditioning,
        position_idx=image_ids,
        encoder_position_idx=text_ids,
    )
    single_enc, single_output = jax.jit(single)(
        x,
        enc_x,
        conditioning,
        position_idx=image_ids,
        encoder_position_idx=text_ids,
    )

    assert output.shape == single_output.shape == x.shape
    assert output_enc.shape == single_enc.shape == enc_x.shape
