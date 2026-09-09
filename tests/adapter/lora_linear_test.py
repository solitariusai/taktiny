import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.utils.spmd import map_logical_axis_names


@pytest.mark.parametrize('bias', [False, True])
def test_lora_starts_as_base_and_learns_through_zero_b(bias):
    base = nn.Linear((2, 3), (2, 2), rngs=nn.Rngs(0))
    layer = nn.LoRALinear(base, 2, 4.0, bias=bias, rngs=nn.Rngs(1))
    x = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3) / 10
    assert jnp.allclose(jax.jit(layer)(x), base(x))
    assert layer.lora_A.bias is None
    assert (layer.lora_B.bias is not None) == bias

    gradients = jax.grad(lambda model: jnp.sum(model(x)))(layer)
    assert jnp.all(jnp.isfinite(gradients.lora_B.kernel.value))
    assert jnp.any(gradients.lora_B.kernel.value != 0)


def test_lora_nd_update_matches_explicit_contraction_and_bias():
    calls = []

    def dot(lhs, rhs, dimension_numbers, precision,
            preferred_element_type, *, out_sharding=None):
        calls.append((precision, preferred_element_type))
        return jax.lax.dot_general(
            lhs, rhs, dimension_numbers, precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )

    base = nn.Linear((2, 3), (2, 2), rngs=nn.Rngs(0))
    layer = nn.LoRALinear(
        base, 2, 6.0, rngs=nn.Rngs(1),
        kernel_initializer=jax.nn.initializers.ones,
        bias_initializer=jax.nn.initializers.ones,
        kernel_metadata={'role': 'adapter'},
        bias_metadata={'role': 'offset'},
        dot_general=dot,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2) / 10
    layer.lora_B.load_state_dict({'kernel': b})
    x = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3) / 10
    expected = base(x) + 3 * (
        jnp.einsum('nij,ijr,rkl->nkl', x, layer.lora_A.kernel.value, b) + 1
    )
    assert jnp.allclose(layer(x), expected)
    assert len(calls) == 2
    assert all(item == (jax.lax.Precision.HIGHEST, jnp.float32)
               for item in calls)
    assert layer.lora_A.kernel.metadata == {'role': 'adapter'}
    assert layer.lora_B.kernel.metadata == {'role': 'adapter'}
    assert layer.lora_B.bias.metadata == {'role': 'offset'}
    assert layer.extra_repr() == '2×3 ➤ 2×2, rank=2, alpha=6'


@pytest.mark.parametrize('logical', [False, True])
def test_lora_initializes_sharded_nd_factors_and_constrains_output(logical):
    mesh = Mesh(
        np.asarray(jax.devices()), ('model',), axis_types=(AxisType.Explicit,),
    )
    width = 2 * mesh.size
    base = nn.Linear((2, 2), (width, 2), rngs=nn.Rngs(0))
    names = (None, None, 'output', None) if logical else None
    explicit = P(None, None, 'model', None)
    # A deliberately different spec must be overridden by the logical axes.
    if logical:
        explicit = P()
    with jax.set_mesh(mesh), map_logical_axis_names({'output': 'model'}):
        layer = nn.LoRALinear(
            base, 2, 4.0, rngs=nn.Rngs(1),
            axis_names=names, partition_spec=explicit,
        )
        sharding = NamedSharding(mesh, P(None, 'model', None))
        output = jax.jit(
            lambda x: layer(x, out_sharding=sharding)
        )(jnp.ones((3, 2, 2)))

    assert layer.lora_A.kernel.partition_spec == P(None, None, None)
    assert layer.lora_B.kernel.partition_spec == P(None, 'model', None)
    assert layer.lora_B.bias.partition_spec == P('model', None)
    assert layer.lora_B.kernel.value.sharding.is_equivalent_to(
        sharding, 3,
    )
    assert output.sharding.is_equivalent_to(sharding, 3)
    if logical:
        assert layer.lora_A.kernel.axis_names == (None, None, None)
        assert layer.lora_B.kernel.axis_names == (None, 'output', None)


def test_lora_quantizes_adapter_kernels_without_changing_base():
    base = nn.Linear(4, 3, rngs=nn.Rngs(0))
    layer = nn.LoRALinear(base, 2, 4.0, quant='int8', rngs=nn.Rngs(1))
    x = jnp.ones((2, 4))
    assert not isinstance(base.kernel.value, qwix.QArray)
    assert isinstance(layer.lora_A.kernel.value, qwix.QArray)
    assert isinstance(layer.lora_B.kernel.value, qwix.QArray)
    assert jnp.allclose(jax.jit(layer)(x), base(x))


@pytest.mark.parametrize('rank', [0, -1, True, 1.5])
def test_lora_rejects_invalid_rank(rank):
    base = nn.Linear(2, 3, rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='rank'):
        nn.LoRALinear(base, rank, 4.0, rngs=nn.Rngs(1))


@pytest.mark.parametrize('bias', [False, True])
def test_lora_inherits_base_dtype_bias_metadata_and_dot_settings(bias):
    def dot(lhs, rhs, dimension_numbers, precision,
            preferred_element_type, *, out_sharding=None):
        return jax.lax.dot_general(
            lhs, rhs, dimension_numbers, precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )

    base = nn.Linear(
        4, 3, bias=bias, dtype=jnp.bfloat16, rngs=nn.Rngs(0),
        dot_general=dot, precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
        kernel_metadata={'kind': 'weight'}, bias_metadata={'kind': 'bias'},
    )
    layer = nn.LoRALinear(base, 2, 4.0, rngs=nn.Rngs(1))
    for factor in (layer.lora_A, layer.lora_B):
        assert factor.kernel.dtype == jnp.bfloat16
        assert factor.dot_general is dot
        assert factor.precision == jax.lax.Precision.HIGHEST
        assert factor.preferred_element_type == jnp.float32
        assert factor.kernel.metadata == base.kernel.metadata
        assert factor.kernel.metadata is not base.kernel.metadata
    assert (layer.lora_B.bias is not None) == bias
    if bias:
        assert layer.lora_B.bias.metadata == base.bias.metadata
    x = jnp.ones((2, 4), dtype=jnp.bfloat16)
    assert jnp.allclose(jax.jit(layer)(x), base(x))


def test_lora_explicit_settings_override_inherited_properties():
    base = nn.Linear(
        4, 3, dtype=jnp.bfloat16, rngs=nn.Rngs(0),
        kernel_metadata={'kind': 'base'}, precision=jax.lax.Precision.HIGHEST,
    )
    layer = nn.LoRALinear(
        base, 2, 4.0, bias=False, dtype=jnp.float32, rngs=nn.Rngs(1),
        kernel_metadata={}, precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32, dot_general=jax.lax.dot_general,
    )
    assert layer.lora_B.bias is None
    for factor in (layer.lora_A, layer.lora_B):
        assert factor.kernel.dtype == jnp.float32
        assert factor.kernel.metadata == {}
        assert factor.precision == jax.lax.Precision.DEFAULT
        assert factor.dot_general is jax.lax.dot_general


@pytest.mark.parametrize('logical', [False, True])
def test_lora_resolves_inherited_names_in_current_mapping_context(logical):
    mesh = Mesh(np.asarray(jax.devices()), ('model',))
    width = 2 * mesh.size
    with jax.set_mesh(mesh):
        with map_logical_axis_names({'output': 'model'}):
            base = nn.Linear(
                (2, 2), (width, 2), rngs=nn.Rngs(0),
                axis_names=(None, None, 'output', None) if logical else None,
                partition_spec=P(None, None, 'model', None),
            )
            inside = nn.LoRALinear(base, 2, 4.0, rngs=nn.Rngs(3))
        layer = nn.LoRALinear(base, 2, 4.0, rngs=nn.Rngs(1))
        replicated = nn.LoRALinear(
            base, 2, 4.0, partition_spec=P(), rngs=nn.Rngs(2),
        )
        x = jnp.ones((3, 2, 2))
        assert jnp.allclose(jax.jit(layer)(x), base(x))

    assert inside.lora_B.kernel.partition_spec == P(None, 'model', None)
    assert inside.lora_B.kernel.value.sharding.is_equivalent_to(
        NamedSharding(mesh, P(None, 'model', None)), 3,
    )
    spec = P(None, None, None) if logical else P(None, 'model', None)
    assert layer.lora_B.kernel.partition_spec == spec
    assert layer.lora_B.kernel.value.sharding.is_equivalent_to(
        NamedSharding(mesh, spec), 3,
    )
    assert layer.lora_B.bias.partition_spec == (
        P(None, None) if logical else P('model', None)
    )
    assert replicated.lora_B.kernel.value.sharding.is_fully_replicated
    assert replicated.lora_B.kernel.axis_names is None
    if logical:
        assert layer.lora_B.kernel.axis_names == (None, 'output', None)
        assert layer.lora_B.bias.axis_names == ('output', None)


def test_lora_uses_floating_dtype_for_quantized_base():
    base = nn.Linear(4, 3, quant='int8', dtype=jnp.bfloat16, rngs=nn.Rngs(0))
    layer = nn.LoRALinear(base, 2, 4.0, rngs=nn.Rngs(1))
    assert not isinstance(layer.lora_A.kernel.value, qwix.QArray)
    assert layer.lora_A.kernel.dtype == jnp.bfloat16
    x = jnp.ones((2, 4), dtype=jnp.bfloat16)
    assert jnp.allclose(jax.jit(layer)(x), base(x))
