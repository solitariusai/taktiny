import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.utils.spmd import map_logical_axis_names


def test_embedding_supports_multi_axis_feature_shapes():
    layer = nn.Embedding(5, (2, 3), rngs=nn.Rngs(0))
    table = jnp.arange(30, dtype=jnp.float32).reshape(5, 2, 3)
    layer.load_state_dict({'embedding': table})
    indices = jnp.asarray([[0, 2], [4, 1]])

    output = jax.jit(layer)(indices)

    assert layer.num_embeddings == (5,)
    assert layer.embed_features == (2, 3)
    assert output.shape == (2, 2, 2, 3)
    assert jnp.array_equal(output, table[indices])


def test_embedding_uses_trailing_coordinates_for_nd_vocabulary():
    layer = nn.Embedding((2, 3), (2, 2), rngs=nn.Rngs(0))
    table = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 2, 2)
    layer.load_state_dict({'embedding': table})
    coordinates = jnp.asarray(
        [
            [[0, 0], [1, 2]],
            [[0, 1], [1, 0]],
        ]
    )

    output = jax.jit(layer)(coordinates)
    expected = table[coordinates[..., 0], coordinates[..., 1]]

    assert layer.num_embeddings == (2, 3)
    assert layer.embed_features == (2, 2)
    assert output.shape == (2, 2, 2, 2)
    assert jnp.array_equal(output, expected)


def test_embedding_records_parameter_configuration():
    layer = nn.Embedding(
        (2, 3),
        (4, 5),
        axis_names=('vocab_row', 'vocab_column', 'height', 'width'),
        metadata={'kind': 'token_table'},
        precision=jax.lax.Precision.HIGH,
        preferred_element_type=jnp.float32,
        rngs=nn.Rngs(0),
    )

    assert layer.embedding.shape == (2, 3, 4, 5)
    assert layer.embedding.axis_names == (
        'vocab_row',
        'vocab_column',
        'height',
        'width',
    )
    assert layer.embedding.metadata == {'kind': 'token_table'}
    assert layer.precision == jax.lax.Precision.HIGH
    assert layer.preferred_element_type == jnp.float32
    assert layer.extra_repr() == '2×3 ➤ 4×5'


def test_embedding_validates_shapes_and_nd_coordinates():
    with pytest.raises(ValueError, match='num_embeddings.*positive integer'):
        nn.Embedding((2, 0), 3, rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='embed_features.*at least one'):
        nn.Embedding(2, (), rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='axis_names length'):
        nn.Embedding(
            (2, 3),
            4,
            axis_names=('vocabulary', 'embedding'),
            rngs=nn.Rngs(0),
        )

    layer = nn.Embedding((2, 3), 4, rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='vocabulary rank 2'):
        layer(jnp.asarray([0, 1, 0]))


def test_embedding_supports_quantized_nd_tables():
    rule = qwix.QuantizationRule(
        op_names=('embedding',),
        weight_qtype='int8',
    )
    layer = nn.Embedding(
        (2, 3),
        (2, 2),
        quant=rule,
        rngs=nn.Rngs(0),
    )
    coordinates = jnp.asarray([[0, 1], [1, 2]])

    output = jax.jit(layer)(coordinates)
    table = qwix.dequantize(layer.embedding.value)
    expected = table[coordinates[..., 0], coordinates[..., 1]]

    assert isinstance(layer.embedding.value, qwix.QArray)
    assert output.shape == (2, 2, 2)
    assert jnp.allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_embedding_applies_explicit_parameter_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('model',))
    table_spec = P(None, 'model', None, None)

    with jax.set_mesh(mesh):
        layer = nn.Embedding(
            (2, 3),
            (4, 5),
            partition_spec=table_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.embedding.partition_spec == table_spec
    assert layer.embedding.value.sharding.is_equivalent_to(
        NamedSharding(mesh, table_spec),
        layer.embedding.ndim,
    )


def test_embedding_logical_axes_override_explicit_parameter_sharding():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ('data', 'model'))
    explicit_spec = P('data', None, None)
    logical_spec = P(None, None, 'model')

    with jax.set_mesh(mesh), map_logical_axis_names({'feature': 'model'}):
        layer = nn.Embedding(
            4,
            (2, 3),
            axis_names=('vocabulary', 'group', 'feature'),
            partition_spec=explicit_spec,
            rngs=nn.Rngs(0),
        )

    assert layer.embedding.partition_spec == logical_spec
    assert layer.embedding.value.sharding.spec == logical_spec


def test_embedding_out_sharding_constrains_lookup_result():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = nn.Embedding((2, 3), 4, rngs=nn.Rngs(0))
    coordinates = jnp.asarray([[0, 1], [1, 2]])
    apply = lambda value: layer(value, out_sharding=out_sharding)

    jaxpr = jax.make_jaxpr(apply)(coordinates).jaxpr
    output = jax.jit(apply)(coordinates)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)
