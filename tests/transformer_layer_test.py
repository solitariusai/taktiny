import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.layers import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from taktiny.utils.typing import ShardMode


def _transformer(**kwargs):
    options = {
        'hidden_size': 8,
        'num_heads': 2,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'intermediate_size': 16,
        'dropout': 0.0,
        'rngs': nn.Rngs(0),
    }
    options.update(kwargs)
    return Transformer(**options)


class _CustomEncoder(nn.Module):
    def __call__(
        self,
        source,
        mask=None,
        is_causal=False,
        *,
        out_sharding=None,
    ):
        del mask, is_causal, out_sharding
        return source + 1


class _CustomDecoder(nn.Module):
    def __call__(
        self,
        target,
        memory=None,
        target_mask=None,
        memory_mask=None,
        target_is_causal=False,
        memory_is_causal=False,
        *,
        out_sharding=None,
    ):
        del (
            target_mask,
            memory_mask,
            target_is_causal,
            memory_is_causal,
            out_sharding,
        )
        return target + (0 if memory is None else jnp.mean(memory, axis=1, keepdims=True))


@pytest.mark.parametrize('norm_first', [False, True])
def test_transformer_is_jittable_and_differentiable(norm_first):
    layer = _transformer(norm_first=norm_first)
    source = jax.random.normal(jax.random.key(1), (5, 3, 8))
    target = jax.random.normal(jax.random.key(2), (4, 3, 8))

    output = jax.jit(layer)(source, target, target_is_causal=True)
    gradient = jax.grad(
        lambda value: jnp.sum(layer(source, value, target_is_causal=True))
    )(target)

    assert output.shape == target.shape
    assert jnp.all(jnp.isfinite(output))
    assert jnp.all(jnp.isfinite(gradient))


def test_transformer_supports_batch_first_and_unbatched_inputs():
    batched = _transformer(batch_first=True)
    source = jnp.ones((3, 5, 8))
    target = jnp.ones((3, 4, 8))

    assert batched(source, target).shape == target.shape

    unbatched = _transformer(batch_first=False)
    source = jnp.ones((5, 8))
    target = jnp.ones((4, 8))

    assert unbatched(source, target).shape == target.shape


def test_public_transformer_encoder_and_decoder_containers():
    options = {
        'hidden_size': 8,
        'num_heads': 2,
        'intermediate_size': 16,
        'dropout': 0.0,
        'activation': 'relu',
        'norm_first': False,
        'norm_eps': 1e-5,
        'bias': True,
        'dtype': jnp.float32,
        'rngs': nn.Rngs(0),
        'shard_mode': ShardMode.AUTO,
        'quant': None,
        'dot_general': None,
    }
    encoder = TransformerEncoder(
        [TransformerEncoderLayer(**options)],
        nn.LayerNorm(8),
    )
    decoder = TransformerDecoder(
        [TransformerDecoderLayer(**options)],
        nn.LayerNorm(8),
    )
    source = jnp.ones((2, 3, 8))
    target = jnp.ones((2, 2, 8))

    memory = jax.jit(encoder)(source)
    output = jax.jit(decoder)(target, memory, target_is_causal=True)
    decoder_only = decoder(target, None, target_is_causal=True)

    assert len(encoder) == 1
    assert len(decoder) == 1
    assert memory.shape == source.shape
    assert output.shape == target.shape
    assert decoder_only.shape == target.shape


def test_transformer_accepts_custom_encoder_and_decoder():
    encoder = _CustomEncoder()
    decoder = _CustomDecoder()
    layer = _transformer(
        custom_encoder=encoder,
        custom_decoder=decoder,
        batch_first=True,
    )
    source = jnp.ones((2, 3, 8))
    target = jnp.ones((2, 2, 8)) * 3

    output = layer(source, target)

    assert layer.encoder is encoder
    assert layer.decoder is decoder
    assert jnp.array_equal(output, jnp.full_like(target, 5))


def test_transformer_supports_encoder_only_and_decoder_only_calls():
    layer = _transformer(batch_first=True)
    source = jax.random.normal(jax.random.key(1), (2, 3, 8))
    target = jax.random.normal(jax.random.key(2), (2, 4, 8))

    encoded = layer(source, None)
    decoded = layer(None, target, target_is_causal=True)

    assert encoded.shape == source.shape
    assert decoded.shape == target.shape


def test_transformer_causal_mask_prevents_future_target_influence():
    layer = _transformer(batch_first=True)
    source = jax.random.normal(jax.random.key(1), (1, 3, 8))
    target = jax.random.normal(jax.random.key(2), (1, 4, 8))
    changed = target.at[:, 1:, :].add(100)

    original = layer(source, target, target_is_causal=True)
    modified = layer(source, changed, target_is_causal=True)

    assert jnp.allclose(original[:, 0], modified[:, 0], atol=1e-5)
    assert not jnp.allclose(original[:, -1], modified[:, -1])


def test_transformer_padding_masks_hide_changed_source_tokens():
    layer = _transformer(batch_first=True)
    source = jax.random.normal(jax.random.key(1), (1, 4, 8))
    changed = source.at[:, -1, :].add(100)
    target = jax.random.normal(jax.random.key(2), (1, 3, 8))
    padding = jnp.asarray([[False, False, False, True]])

    original = layer(
        source,
        target,
        source_key_padding_mask=padding,
        memory_key_padding_mask=padding,
    )
    modified = layer(
        changed,
        target,
        source_key_padding_mask=padding,
        memory_key_padding_mask=padding,
    )

    assert jnp.allclose(original, modified, atol=1e-5)


def test_transformer_dropout_uses_recursive_module_mode():
    layer = _transformer(
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.5,
        batch_first=True,
    )
    source = jnp.ones((2, 3, 8))
    target = jnp.ones((2, 2, 8))

    first = layer(source, target)
    second = layer(source, target)
    layer.eval()
    evaluation = layer(source, target)

    assert not jnp.array_equal(first, second)
    assert jnp.all(jnp.isfinite(evaluation))
    for module in (
        layer,
        layer.encoder,
        layer.encoder.layers[0],
        layer.decoder,
        layer.decoder.layers[0],
    ):
        parameters = inspect.signature(module.__call__).parameters
        assert 'key' not in parameters
        assert 'training' not in parameters


def test_transformer_explicit_sharding_constrains_returned_layout():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    sharding = NamedSharding(mesh, P())
    layer = _transformer(
        num_encoder_layers=1,
        num_decoder_layers=1,
        batch_first=True,
        shard_mode=ShardMode.EXPLICIT,
    )
    source = jnp.ones((2, 3, 8))
    target = jnp.ones((2, 2, 8))

    apply = lambda source, target: layer(
        source,
        target,
        out_sharding=sharding,
    )
    jaxpr = jax.make_jaxpr(apply)(source, target).jaxpr
    output = jax.jit(apply)(source, target)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(sharding, output.ndim)


def test_transformer_validates_configuration_and_input_contracts():
    with pytest.raises(ValueError, match='divisible'):
        _transformer(hidden_size=7)
    with pytest.raises(ValueError, match='positive integer'):
        _transformer(num_encoder_layers=0)
    with pytest.raises(ValueError, match='trailing size'):
        _transformer()(jnp.ones((3, 7)), jnp.ones((2, 8)))
    with pytest.raises(TypeError, match='boolean array'):
        _transformer()(
            jnp.ones((3, 8)),
            jnp.ones((2, 8)),
            source_mask=jnp.ones((3, 3)),
        )
