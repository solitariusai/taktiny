from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import save_file

from taktiny import nn
from taktiny.autoencoder import Autoencoder, AutoencoderDC
from taktiny.cosettes._continuo import (
    _pixel_shuffle,
    _pixel_unshuffle,
)
from taktiny.maestro.config import ModelConfig


class _Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.factor


def _dc_config(**overrides) -> ModelConfig:
    values = {
        'in_channels': 3,
        'latent_channels': 4,
        'attention_head_dim': 4,
        'encoder_block_types': 'ResBlock',
        'decoder_block_types': 'ResBlock',
        'encoder_block_out_channels': (8, 16),
        'decoder_block_out_channels': (8, 16),
        'encoder_layers_per_block': (1, 1),
        'decoder_layers_per_block': (1, 1),
        'encoder_qkv_multiscales': ((), ()),
        'decoder_qkv_multiscales': ((), ()),
        'upsample_block_type': 'pixel_shuffle',
        'downsample_block_type': 'pixel_unshuffle',
        'decoder_norm_types': 'rms_norm',
        'decoder_act_fns': 'silu',
        'scaling_factor': 0.5,
        'dtype': 'float32',
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_autoencoder_is_array_only_and_keeps_scaling_explicit():
    model = Autoencoder(
        _Scale(2.0),
        _Scale(0.5),
        scaling_factor=0.25,
        spatial_compression_ratio=8,
    )
    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    assert jnp.array_equal(model.encode(x), x * 2)
    assert jnp.array_equal(model(x), x)
    assert jnp.array_equal(
        model.unscale_latents(model.scale_latents(x)),
        x,
    )


def test_autoencoder_rejects_non_module_components():
    with pytest.raises(TypeError, match='encoder'):
        Autoencoder(lambda x: x, _Scale(1.0))


def test_pixel_shuffle_and_unshuffle_are_exact_inverses():
    x = jnp.arange(1 * 4 * 6 * 3).reshape(1, 4, 6, 3)
    packed = _pixel_unshuffle(x, 2)

    assert packed.shape == (1, 2, 3, 12)
    assert jnp.array_equal(_pixel_shuffle(packed, 2), x)


def test_autoencoder_dc_encode_decode_shape_and_jit():
    model = AutoencoderDC(_dc_config(), rngs=nn.Rngs(0))
    sample = jnp.ones((2, 8, 8, 3), dtype=jnp.float32)

    latent_shape = jax.eval_shape(model.encode, sample).shape
    output_shape = jax.eval_shape(model, sample).shape

    assert latent_shape == (2, 4, 4, 4)
    assert output_shape == sample.shape
    assert model.spatial_compression_ratio == 2
    assert jax.eval_shape(jax.jit(model), sample).shape == sample.shape


def test_autoencoder_dc_efficient_vit_path_is_finite():
    config = _dc_config(
        encoder_block_types='EfficientViTBlock',
        decoder_block_types='EfficientViTBlock',
        encoder_block_out_channels=(8,),
        decoder_block_out_channels=(8,),
        encoder_layers_per_block=(1,),
        decoder_layers_per_block=(1,),
        encoder_qkv_multiscales=((3,),),
        decoder_qkv_multiscales=((3,),),
    )
    model = AutoencoderDC(config, rngs=nn.Rngs(1))
    sample = jnp.ones((1, 4, 4, 3), dtype=jnp.float32)

    output = model(sample)

    assert output.shape == sample.shape
    assert jnp.all(jnp.isfinite(output))


def test_autoencoder_dc_parameters_are_differentiable():
    config = _dc_config(
        encoder_block_out_channels=(8,),
        decoder_block_out_channels=(8,),
        encoder_layers_per_block=(1,),
        decoder_layers_per_block=(1,),
        encoder_qkv_multiscales=((),),
        decoder_qkv_multiscales=((),),
    )
    model = AutoencoderDC(config, rngs=nn.Rngs(4))
    sample = jnp.ones((1, 4, 4, 3), dtype=jnp.float32)

    gradients = jax.eval_shape(
        jax.grad(
            lambda candidate: jnp.mean(jnp.square(candidate(sample)))
        ),
        model,
    )

    gradient_parameters = gradients.flat_parameter_dict()
    model_parameters = model.flat_parameter_dict()
    assert gradient_parameters.keys() == model_parameters.keys()
    assert {
        name: parameter.shape
        for name, parameter in gradient_parameters.items()
    } == {
        name: parameter.shape
        for name, parameter in model_parameters.items()
    }


def test_autoencoder_dc_input_layout_is_checked():
    model = AutoencoderDC(_dc_config(), rngs=nn.Rngs(2))

    with pytest.raises(ValueError, match='DCEncoder expects'):
        model.encode(jnp.ones((1, 3, 8, 8), dtype=jnp.float32))


def test_autoencoder_dc_loads_diffusers_filename_and_conv_layout(tmp_path):
    config = _dc_config()
    model = AutoencoderDC(config, rngs=nn.Rngs(3))
    checkpoint = {}
    for name, parameter in model.flat_parameter_dict().items():
        value = np.asarray(parameter.value)
        if name.endswith('.weight') and value.ndim == 2:
            value = value.T
        elif name.endswith('.weight') and value.ndim >= 3:
            value = value.transpose(
                value.ndim - 1,
                value.ndim - 2,
                *range(value.ndim - 2),
            )
        checkpoint[name] = np.ascontiguousarray(value)

    with (tmp_path / 'config.json').open('w') as config_file:
        json.dump(vars(config), config_file)
    save_file(
        checkpoint,
        tmp_path / 'diffusion_pytorch_model.safetensors',
    )

    loaded = AutoencoderDC.from_pretrained(tmp_path, local=True)

    expected = model.encoder.conv_in.weight.value
    actual = loaded.encoder.conv_in.weight.value
    assert jnp.array_equal(actual, expected)
