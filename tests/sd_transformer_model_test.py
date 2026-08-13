import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.cosettes.transformers.sd import (
    SD3PatchEmbedding,
)
from taktiny.cosettes.transformers._ordinario import DiffusionTransformerModel
from taktiny.maestro.opus.sd import SD3TransformerModel, _SD3_MODULE_MAP
from taktiny.maestro.config import ModelConfig
from taktiny.maestro import Maestro
from taktiny.utils.weights import map_state_dict


def _config(**overrides) -> ModelConfig:
    values = {
        'num_layers': 2,
        'num_attention_heads': 2,
        'attention_head_dim': 4,
        'in_channels': 3,
        'out_channels': 2,
        'patch_size': 2,
        'sample_size': 8,
        'pos_embed_max_size': 6,
        'pooled_projection_dim': 5,
        'joint_attention_dim': 6,
        'caption_projection_dim': 8,
        'intermediate_size': 16,
        'qk_norm': None,
        'dtype': 'float32',
    }
    values.update(overrides)
    return ModelConfig(**values)


def _inputs():
    return (
        jax.random.normal(jax.random.key(1), (2, 8, 6, 3)),
        jax.random.normal(jax.random.key(2), (2, 4, 6)),
        jax.random.normal(jax.random.key(3), (2, 5)),
        jnp.asarray([1.0, 2.0]),
    )


def test_sd3_patch_embedding_center_crops_fixed_position_table():
    embedding = SD3PatchEmbedding(
        sample_size=8,
        patch_size=2,
        in_channels=3,
        embedding_dim=8,
        pos_embed_max_size=6,
        dtype='float32',
        rngs=nn.Rngs(0),
    )
    embedding.projection.weight.value = jnp.zeros_like(
        embedding.projection.weight.value
    )
    embedding.projection.bias.value = jnp.zeros_like(
        embedding.projection.bias.value
    )

    output = embedding(jnp.zeros((1, 8, 6, 3)))
    table = embedding.pos_embed.value.reshape(1, 6, 6, 8)
    expected = table[:, 1:5, 1:4, :].reshape(1, 12, 8)

    assert output.shape == (1, 12, 8)
    assert jnp.array_equal(output, expected)


def test_sd3_patch_embedding_dynamic_positions_support_new_grid():
    embedding = SD3PatchEmbedding(
        sample_size=(8, 8),
        patch_size=(2, 2),
        in_channels=3,
        embedding_dim=8,
        pos_embed_max_size=None,
        dtype='float32',
        rngs=nn.Rngs(0),
    )

    assert embedding(jnp.ones((6, 10, 3))).shape == (15, 8)
    positions = embedding._positions((3, 5))[0].reshape(3, 5, 8)
    assert positions[0, 0, 0] == 0.0
    assert positions[0, 1, 0] != 0.0
    assert positions[1, 0, 0] == 0.0


def test_sd3_model_forward_is_nhwc_jittable_and_supports_tuple_return():
    model = SD3TransformerModel(_config(), rngs=nn.Rngs(0))
    inputs = _inputs()

    output = jax.jit(model)(*inputs)
    tuple_output = model(*inputs, return_dict=False)
    scalar_timestep_output = model(*inputs[:3], jnp.asarray(1.0))

    assert output.shape == (2, 8, 6, 2)
    assert jnp.isfinite(output).all()
    assert isinstance(tuple_output, tuple)
    assert len(tuple_output) == 1
    assert jnp.array_equal(tuple_output[0], model(*inputs))
    assert scalar_timestep_output.shape == output.shape
    assert isinstance(model, DiffusionTransformerModel)
    assert isinstance(model.patch_embedding, SD3PatchEmbedding)
    assert len(model.layers) == 2


def test_sd3_model_scans_compatible_layers_and_falls_back_for_mixed_layers():
    inputs = _inputs()
    list_model = SD3TransformerModel(
        _config(context_pre_only=False),
        rngs=nn.Rngs(0),
        use_list=True,
    )
    scanned_model = SD3TransformerModel(
        _config(context_pre_only=False),
        rngs=nn.Rngs(0),
        use_list=False,
    )

    assert list_model.use_list
    assert not scanned_model.use_list
    assert isinstance(scanned_model.layers, nn.SeqStack)
    assert jnp.allclose(list_model(*inputs), scanned_model(*inputs), atol=1e-6)

    mixed_model = SD3TransformerModel(
        _config(),
        rngs=nn.Rngs(0),
        use_list=False,
    )
    assert not mixed_model.requested_use_list
    assert not mixed_model.use_list
    assert isinstance(mixed_model.layers, nn.SeqStack)
    assert mixed_model.layers.group_sizes == (1, 1)
    assert jnp.allclose(
        SD3TransformerModel(_config(), rngs=nn.Rngs(0))(*inputs),
        mixed_model(*inputs),
        atol=1e-6,
    )


def test_sd3_model_controlnet_residual_contract_and_skip_layers():
    model = SD3TransformerModel(_config(), rngs=nn.Rngs(0))
    inputs = _inputs()
    token_shape = (2, 12, 8)

    output = model(
        *inputs,
        controlnet_x=[jnp.zeros(token_shape)],
        skip_layers=[0],
    )
    assert output.shape == (2, 8, 6, 2)

    with pytest.raises(ValueError, match='control residual'):
        model(*inputs, controlnet_x=[jnp.zeros((2, 11, 8))])
    with pytest.raises(ValueError, match='must not be empty'):
        model(*inputs, controlnet_x=[])
    with pytest.raises(ValueError, match='invalid layer index'):
        model(*inputs, skip_layers=[2])


def test_sd3_model_rematerialized_input_gradient_is_finite():
    model = SD3TransformerModel(_config(), rngs=nn.Rngs(0))
    model.enable_remat()
    x, enc_x, pooled, timestep = _inputs()

    gradient = jax.grad(
        lambda latent: model(latent, enc_x, pooled, timestep).sum()
    )(x)

    assert gradient.shape == x.shape
    assert jnp.isfinite(gradient).all()


def test_sd3_diffusers_module_map_covers_backbone_components():
    source = {
        'pos_embed.proj.weight': jnp.zeros((1,)),
        'time_text_embed.timestep_embedder.linear_1.weight': jnp.zeros((1,)),
        'transformer_blocks.0.attn.to_q.weight': jnp.zeros((1,)),
        'transformer_blocks.0.attn.add_k_proj.weight': jnp.zeros((1,)),
        'transformer_blocks.0.attn.norm_added_q.weight': jnp.zeros((1,)),
        'transformer_blocks.0.attn2.norm_q.weight': jnp.zeros((1,)),
        'transformer_blocks.0.ff_context.net.0.proj.weight': jnp.zeros((1,)),
    }

    mapped = map_state_dict(source, list(_SD3_MODULE_MAP))

    assert set(mapped) == {
        'patch_embedding.projection.weight',
        'condition_embedding.embeddings.timestep.1.projection.weight',
        'layers.0.attn.q_proj_1.weight',
        'layers.0.attn.k_proj_2.weight',
        'layers.0.attn.q_norm_2.weight',
        'layers.0.attn2.q_norm.weight',
        'layers.0.ff_context.input.weight',
    }


def test_maestro_resolves_sd3_diffusers_class_name():
    config = _config(_class_name='SD3Transformer2DModel')

    model = Maestro.eval_shape('unused', config=config)

    assert isinstance(model, SD3TransformerModel)
    assert isinstance(model, DiffusionTransformerModel)
    assert not model.use_list
    assert isinstance(model.layers, nn.SeqStack)
    assert model.layers.group_sizes == (1, 1)
