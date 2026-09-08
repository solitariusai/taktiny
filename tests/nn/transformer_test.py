import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.utils.spmd import map_logical_axis_names


def _identity_attention(hidden_size: int = 4, num_heads: int = 2):
    attention = nn.Attention(
        hidden_size,
        num_heads,
        bias=False,
        dropout=0.0,
        rngs=nn.Rngs(0),
    )
    head_dim = hidden_size // num_heads
    split_identity = jnp.eye(hidden_size).reshape(
        hidden_size,
        num_heads,
        head_dim,
    )
    merged_identity = jnp.eye(hidden_size).reshape(
        num_heads,
        head_dim,
        hidden_size,
    )
    attention.load_state_dict({
        'q_proj': {'kernel': split_identity},
        'k_proj': {'kernel': split_identity},
        'v_proj': {'kernel': split_identity},
        'o_proj': {'kernel': merged_identity},
    })
    return attention


def test_attention_matches_scaled_dot_product_definition():
    attention = _identity_attention()
    x = jnp.asarray([
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
    ])

    output = attention(x)
    split = x.reshape(1, 2, 2, 2)
    scores = jnp.einsum('bqhd,bkhd->bhqk', split, split) / jnp.sqrt(2.0)
    weights = jax.nn.softmax(scores, axis=-1)
    expected = jnp.einsum('bhqk,bkhd->bqhd', weights, split).reshape(1, 2, 4)

    assert jnp.allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_attention_causal_mask_blocks_future_values():
    attention = _identity_attention()
    original = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4) / 10
    changed = original.at[:, 2].set(1000.0)

    original_output = attention(original, is_causal=True)
    changed_output = attention(changed, is_causal=True)

    assert jnp.allclose(original_output[:, :2], changed_output[:, :2])
    assert not jnp.allclose(original_output[:, 2], changed_output[:, 2])


def test_attention_supports_cross_attention_and_masks():
    attention = _identity_attention()
    query = jnp.ones((2, 4))
    context = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    mask = jnp.asarray([[True, True, False], [True, False, False]])

    output = jax.jit(
        lambda q, c, m: attention(q, context=c, attention_mask=m)
    )(query, context, mask)

    assert output.shape == (2, 4)
    assert jnp.all(jnp.isfinite(output))


def test_attention_fully_masked_query_returns_zero_before_output_bias():
    attention = _identity_attention()
    x = jnp.ones((1, 2, 4))
    mask = jnp.asarray([[False, False], [True, True]])

    output = attention(x, attention_mask=mask)

    assert jnp.array_equal(output[:, 0], jnp.zeros((1, 4)))


def test_feed_forward_and_layers_preserve_hidden_shape():
    x = jnp.ones((2, 5, 8))
    memory = jnp.ones((2, 7, 8))
    feed_forward = nn.FeedForward(
        8,
        32,
        activation='gelu',
        dropout=0.0,
        rngs=nn.Rngs(0),
    )
    encoder = nn.TransformerEncoderLayer(
        8,
        2,
        32,
        norm_first=True,
        dropout=0.0,
        rngs=nn.Rngs(1),
    )
    decoder = nn.TransformerDecoderLayer(
        8,
        2,
        32,
        dropout=0.0,
        rngs=nn.Rngs(2),
    )

    assert feed_forward(x).shape == x.shape
    assert jax.jit(encoder)(x).shape == x.shape
    assert jax.jit(decoder)(x, memory).shape == x.shape


def test_transformer_only_adds_final_norm_for_pre_norm_stacks():
    post_norm = nn.Transformer(
        8,
        2,
        1,
        1,
        16,
        0.0,
        norm_first=False,
        rngs=nn.Rngs(0),
    )
    pre_norm = nn.Transformer(
        8,
        2,
        1,
        1,
        16,
        0.0,
        norm_first=True,
        rngs=nn.Rngs(1),
    )

    assert post_norm.encoder.norm is None
    assert post_norm.decoder.norm is None
    assert pre_norm.encoder.norm is not None
    assert pre_norm.decoder.norm is not None


@pytest.mark.parametrize('batch_first', [False, True])
def test_transformer_encoder_decoder_shapes(batch_first):
    model = nn.Transformer(
        hidden_size=8,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        intermediate_size=16,
        dropout=0.0,
        batch_first=batch_first,
        rngs=nn.Rngs(0),
    )
    source_shape = (2, 5, 8) if batch_first else (5, 2, 8)
    target_shape = (2, 3, 8) if batch_first else (3, 2, 8)
    source = jnp.ones(source_shape)
    target = jnp.ones(target_shape)

    output = jax.jit(model)(source, target)

    assert output.shape == target_shape


def test_transformer_supports_unbatched_encoder_or_decoder_only_use():
    model = nn.Transformer(
        8,
        2,
        1,
        1,
        16,
        0.0,
        rngs=nn.Rngs(0),
    )

    encoded = model(jnp.ones((5, 8)))
    decoded = model(None, jnp.ones((3, 8)))

    assert encoded.shape == (5, 8)
    assert decoded.shape == (3, 8)


def test_transformer_accepts_attention_and_padding_masks():
    model = nn.Transformer(
        8,
        2,
        1,
        1,
        16,
        0.0,
        batch_first=True,
        rngs=nn.Rngs(0),
    )
    source = jnp.ones((2, 4, 8))
    target = jnp.ones((2, 3, 8))
    source_mask = jnp.tril(jnp.ones((4, 4), dtype=jnp.bool_))
    target_padding = jnp.asarray([
        [False, False, True],
        [False, True, True],
    ])
    memory_padding = jnp.asarray([
        [False, False, False, True],
        [False, False, True, True],
    ])

    output = model(
        source,
        target,
        source_mask=source_mask,
        target_key_padding_mask=target_padding,
        memory_key_padding_mask=memory_padding,
    )

    assert output.shape == target.shape
    assert jnp.all(jnp.isfinite(output))


def test_attention_projections_support_quantization():
    attention = nn.Attention(
        8,
        2,
        bias=False,
        dropout=0.0,
        quant='int8',
        rngs=nn.Rngs(0),
    )

    output = jax.jit(attention)(jnp.ones((2, 3, 8)))

    assert isinstance(attention.q_proj.kernel.value, qwix.QArray)
    assert isinstance(attention.o_proj.kernel.value, qwix.QArray)
    assert output.shape == (2, 3, 8)
    assert jnp.all(jnp.isfinite(output))


def test_attention_logical_axes_override_explicit_sharding():
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ('model',))
    num_heads = devices.size
    hidden_size = num_heads * 2
    explicit = {
        'q_proj': P('model', None, None),
        'o_proj': P(None, 'model', None),
    }
    logical = {
        'q_proj': (None, 'heads', None),
        'o_proj': ('heads', None, None),
    }

    with jax.set_mesh(mesh), map_logical_axis_names({'heads': 'model'}):
        attention = nn.Attention(
            hidden_size,
            num_heads,
            bias=False,
            dropout=0.0,
            axis_names=logical,
            partition_spec=explicit,
            rngs=nn.Rngs(0),
        )

    q_spec = P(None, 'model', None)
    o_spec = P('model', None, None)
    assert attention.q_proj.kernel.partition_spec == q_spec
    assert attention.o_proj.kernel.partition_spec == o_spec
    assert attention.q_proj.kernel.value.sharding.is_equivalent_to(
        NamedSharding(mesh, q_spec),
        3,
    )
    assert attention.o_proj.kernel.value.sharding.is_equivalent_to(
        NamedSharding(mesh, o_spec),
        3,
    )


def test_transformer_validates_shapes_and_masks():
    model = nn.Transformer(
        8,
        2,
        1,
        1,
        16,
        0.0,
        batch_first=True,
        rngs=nn.Rngs(0),
    )

    with pytest.raises(ValueError, match='trailing size'):
        model(jnp.ones((2, 4, 7)))
    with pytest.raises(TypeError, match='boolean'):
        model(
            jnp.ones((2, 4, 8)),
            source_mask=jnp.ones((4, 4)),
        )
    with pytest.raises(ValueError, match='batch sizes'):
        model(jnp.ones((2, 4, 8)), jnp.ones((3, 2, 8)))


def test_transformer_out_sharding_covers_final_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    model = nn.Transformer(
        4,
        2,
        1,
        1,
        8,
        0.0,
        batch_first=True,
        rngs=nn.Rngs(0),
    )
    source = jnp.ones((1, 2, 4))
    target = jnp.ones((1, 2, 4))
    apply = lambda first, second: model(
        first,
        second,
        out_sharding=out_sharding,
    )

    output = jax.jit(apply)(source, target)

    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)
