import jax
import jax.numpy as jnp
import numpy as np

from taktiny import nn
from taktiny.cosettes.transformers.ordinario import (
    GatedParallelTransformerLayer,
    JointTransformerLayer,
)
from taktiny.cosettes.transformers.chroma import (
    ChromaSingleTransformerLayer,
    ChromaTransformerLayer,
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


def test_chroma_layers_use_joint_and_gated_parallel_principles():
    assert issubclass(ChromaTransformerLayer, JointTransformerLayer)
    assert issubclass(
        ChromaSingleTransformerLayer,
        GatedParallelTransformerLayer,
    )

    double = ChromaTransformerLayer(_config(), rngs=nn.Rngs(0))
    single = ChromaSingleTransformerLayer(_config(), rngs=nn.Rngs(0))
    assert double.norm1.linear is None
    assert double.norm1_context.linear is None
    assert double.attn.context_first
    assert single.norm.linear is None


def test_chroma_zero_modulation_makes_both_layer_types_identity():
    double = ChromaTransformerLayer(_config(), rngs=nn.Rngs(0))
    single = ChromaSingleTransformerLayer(_config(), rngs=nn.Rngs(1))
    x = jax.random.normal(jax.random.key(2), (2, 3, 8))
    enc_x = jax.random.normal(jax.random.key(3), (2, 2, 8))

    output_enc, output = double(
        x,
        enc_x,
        jnp.zeros((2, 12, 8), dtype=jnp.float32),
    )
    single_output = single(
        jnp.concatenate((enc_x, x), axis=1),
        jnp.zeros((2, 3, 8), dtype=jnp.float32),
    )

    np.testing.assert_array_equal(output, x)
    np.testing.assert_array_equal(output_enc, enc_x)
    np.testing.assert_array_equal(
        single_output,
        jnp.concatenate((enc_x, x), axis=1),
    )


def test_chroma_single_matches_parallel_residual_equation_and_mask():
    layer = ChromaSingleTransformerLayer(
        _config(axes_dims_rope=None),
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(4), (2, 4, 8))
    modulation = jax.random.normal(jax.random.key(5), (2, 3, 8))
    flat_modulation = modulation.reshape(2, 24)
    padding_mask = jnp.asarray(
        ((True, True, True, False), (True, True, True, True))
    )
    pairwise_mask = (
        padding_mask[:, None, None, :]
        & padding_mask[:, None, :, None]
    )

    normalized_base, _ = layer.norm(x, flat_modulation)
    shift, scale, gate = jnp.split(flat_modulation, 3, axis=-1)
    normalized = normalized_base * (1 + scale[:, None]) + shift[:, None]
    expected = x + gate[:, None] * layer.attn(
        normalized,
        attention_mask=pairwise_mask,
    )

    output = layer(x, modulation, attention_mask=padding_mask)
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_chroma_double_adapter_splits_modulation_and_builds_pairwise_mask():
    layer = ChromaTransformerLayer(
        _config(axes_dims_rope=None),
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(14), (2, 3, 8))
    enc_x = jax.random.normal(jax.random.key(15), (2, 2, 8))
    modulation = jax.random.normal(jax.random.key(16), (2, 12, 8))
    padding_mask = jnp.asarray(
        (
            (True, True, True, True, False),
            (True, True, True, True, True),
        )
    )
    pairwise_mask = (
        padding_mask[:, None, None, :]
        & padding_mask[:, None, :, None]
    )
    flat = modulation.reshape(2, 96)

    expected_enc, expected = JointTransformerLayer.__call__(
        layer,
        x,
        enc_x,
        flat[:, :48],
        context_conditioning=flat[:, 48:],
        attention_mask=pairwise_mask,
    )
    output_enc, output = layer(
        x,
        enc_x,
        modulation,
        attention_mask=padding_mask,
    )

    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output_enc, expected_enc, rtol=1e-6, atol=1e-6)


def test_chroma_layers_are_jittable_with_joint_positions():
    double = ChromaTransformerLayer(_config(), rngs=nn.Rngs(0))
    single = ChromaSingleTransformerLayer(_config(), rngs=nn.Rngs(1))
    x = jnp.ones((1, 3, 8), dtype=jnp.float32)
    enc_x = jnp.ones((1, 2, 8), dtype=jnp.float32)
    image_ids = jnp.asarray(((0, 0), (1, 0), (2, 0)))
    text_ids = jnp.asarray(((0, 0), (0, 1)))

    output_enc, output = jax.jit(double)(
        x,
        enc_x,
        jnp.zeros((1, 12, 8), dtype=jnp.float32),
        position_idx=image_ids,
        encoder_position_idx=text_ids,
    )
    single_output = jax.jit(single)(
        jnp.concatenate((enc_x, x), axis=1),
        jnp.zeros((1, 3, 8), dtype=jnp.float32),
        position_idx=jnp.concatenate((text_ids, image_ids), axis=0),
    )

    assert output.shape == x.shape
    assert output_enc.shape == enc_x.shape
    assert single_output.shape == (1, 5, 8)
