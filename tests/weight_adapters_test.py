import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.utils.spmd import map_logical_axis_names


CLASSES = (nn.DoRALinear, nn.AdaLoRALinear, nn.LoHaLinear,
           nn.LoKrLinear, nn.VeRALinear)


def make_adapter(cls, base, rank=2, **kwargs):
    if cls is nn.VeRALinear:
        a = jax.random.normal(jax.random.key(2),
                              (math.prod(base.in_features) + 2, rank + 1))
        b = jax.random.normal(jax.random.key(3),
                              (rank + 1, math.prod(base.out_features) + 2))
        return cls(base, rank, vera_A=nn.Parameter(a, trainable=False),
                   vera_B=nn.Parameter(b, trainable=False), **kwargs)
    return cls(base, rank, 4.0, rngs=nn.Rngs(1), **kwargs)


def activate(layer):
    if isinstance(layer, nn.DoRALinear):
        parameter = layer.lora_B.kernel
    elif isinstance(layer, nn.AdaLoRALinear):
        parameter = layer.lora_E
    elif isinstance(layer, nn.LoHaLinear):
        parameter = layer.loha_B2
    elif isinstance(layer, nn.LoKrLinear):
        parameter = layer.lokr_w1_A if layer.decompose_w1 else layer.lokr_w1
    else:
        parameter = layer.vera_lambda_b
    parameter._value = jnp.arange(parameter.value.size, dtype=parameter.dtype).reshape(
        parameter.shape,
    ) / 30 + 0.1
    return parameter


def delta_kernel(layer):
    shape = layer.base.in_features + layer.base.out_features
    rank = layer.rank
    if isinstance(layer, (nn.DoRALinear, nn.AdaLoRALinear)):
        a = layer.lora_A.kernel.value.reshape((-1, rank))
        b = layer.lora_B.kernel.value.reshape((rank, -1))
        if isinstance(layer, nn.AdaLoRALinear):
            a = a * layer.lora_E.value
        return (layer.scaling * (a @ b)).reshape(shape)
    if isinstance(layer, nn.LoHaLinear):
        a1 = layer.loha_A1.value.reshape((-1, rank))
        b1 = layer.loha_B1.value.reshape((rank, -1))
        a2 = layer.loha_A2.value.reshape((-1, rank))
        b2 = layer.loha_B2.value.reshape((rank, -1))
        return (layer.scaling * (a1 @ b1) * (a2 @ b2)).reshape(shape)
    if isinstance(layer, nn.LoKrLinear):
        w1 = (layer.lokr_w1_A.value @ layer.lokr_w1_B.value
              if layer.decompose_w1 else layer.lokr_w1.value)
        w2 = (layer.lokr_w2_A.value @ layer.lokr_w2_B.value
              if layer.decompose_w2 else layer.lokr_w2.value)
        return (layer.scaling * jnp.kron(w1, w2)).reshape(shape)
    i, o = math.prod(layer.base.in_features), math.prod(layer.base.out_features)
    a = layer.vera_A.value[:i, :rank]
    b = layer.vera_B.value[:rank, :o]
    return (((a * layer.vera_lambda_d.value) @ b).reshape(shape)
            * layer.vera_lambda_b.value)


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('quant', [None, 'int8'])
def test_adapters_start_as_base_with_finite_nonzero_adapter_gradient(cls, quant):
    base = nn.Linear((2, 3), (2, 2), quant=quant, rngs=nn.Rngs(0))
    layer = make_adapter(cls, base)
    x = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3) / 10
    assert jnp.allclose(jax.jit(layer)(x), base(x), atol=1e-5)
    # Differentiate only floating adapter parameters, including with a Qwix base.
    parameter = activate(layer)
    original = parameter.value
    parameter._value = jnp.zeros_like(original)

    def objective(value):
        clone = jax.tree.map(lambda array: array, layer)
        if isinstance(clone, nn.DoRALinear):
            clone.lora_B.kernel._value = value
        elif isinstance(clone, nn.AdaLoRALinear):
            clone.lora_E._value = value
        elif isinstance(clone, nn.LoHaLinear):
            clone.loha_B2._value = value
        elif isinstance(clone, nn.LoKrLinear):
            clone.lokr_w1._value = value
        else:
            clone.vera_lambda_b._value = value
        return jnp.sum(clone(x))

    gradient = jax.grad(objective)(parameter.value)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0)


@pytest.mark.parametrize('cls', CLASSES)
def test_adapter_nonzero_update_matches_explicit_nd_weight(cls):
    base = nn.Linear((2, 3), (2, 2), rngs=nn.Rngs(0))
    base.bias._value = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])
    layer = make_adapter(cls, base)
    activate(layer)
    weight = base.kernel.value + delta_kernel(layer)
    if cls is nn.DoRALinear:
        norms = jnp.sqrt(jnp.sum(weight**2, axis=(0, 1)))
        weight = weight * layer.magnitude.value / norms
    x = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3) / 10
    expected = jnp.tensordot(x, weight, axes=((1, 2), (0, 1))) + base.bias.value
    assert jnp.allclose(jax.jit(layer)(x), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('both,rank,factor', [(False, 1, 2), (True, 1, -1),
                                            (False, 8, 2)])
def test_lokr_factorization_branches_match_kronecker(both, rank, factor):
    base = nn.Linear((6, 6), (6, 6), rngs=nn.Rngs(0))
    layer = make_adapter(nn.LoKrLinear, base, rank, decompose_both=both,
                         decompose_factor=factor)
    activate(layer)
    x = jnp.arange(72, dtype=jnp.float32).reshape(2, 6, 6) / 100
    expected = base(x) + jnp.tensordot(x, delta_kernel(layer), axes=2)
    assert jnp.allclose(jax.jit(layer)(x), expected, atol=1e-5, rtol=1e-5)


def test_dora_zero_columns_stay_finite_and_bias_is_not_scaled():
    base = nn.Linear(3, 2, kernel_initializer=jax.nn.initializers.zeros,
                     bias_initializer=jax.nn.initializers.ones, rngs=nn.Rngs(0))
    layer = make_adapter(nn.DoRALinear, base)
    x = jnp.ones((4, 3))
    assert jnp.array_equal(layer(x), base(x))
    gradient = jax.grad(lambda model: jnp.sum(model(x)))(layer)
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient))


def test_adalora_mask_and_orthogonal_loss():
    base = nn.Linear((2, 3), 4, rngs=nn.Rngs(0))
    layer = make_adapter(nn.AdaLoRALinear, base)
    layer.lora_E._value = jnp.ones((2,))
    layer.mask_rank(jnp.asarray([True, False]))
    assert jnp.array_equal(layer.lora_E.value, jnp.asarray([1., 0.]))
    a = layer.lora_A.kernel.value.reshape(6, 2)
    b = layer.lora_B.kernel.value.reshape(2, 4)
    expected = (jnp.linalg.norm(a.T @ a - jnp.eye(2))
                + jnp.linalg.norm(b @ b.T - jnp.eye(2))) / 2
    assert jnp.allclose(layer.orthogonal_loss(), expected)
    with pytest.raises(TypeError, match='boolean'):
        layer.mask_rank(jnp.ones((2,)))


def test_vera_preserves_shared_projections_and_stops_their_gradients():
    base = nn.Linear(3, 4, rngs=nn.Rngs(0))
    layer = make_adapter(nn.VeRALinear, base)
    a, b = layer.vera_A, layer.vera_B
    another = nn.VeRALinear(base, 2, vera_A=a, vera_B=b)
    assert another.vera_A is a and another.vera_B is b
    activate(layer)
    gradient = jax.grad(lambda model: jnp.sum(model(jnp.ones((2, 3)))))(layer)
    assert jnp.all(gradient.vera_A.value == 0)
    assert jnp.all(gradient.vera_B.value == 0)
    assert jnp.any(gradient.vera_lambda_b.value != 0)


@pytest.mark.parametrize('cls', CLASSES)
def test_adapter_inherits_metadata_dtype_and_current_sharding(cls):
    mesh = Mesh(np.asarray(jax.devices()), ('tp',))
    width = 2 * mesh.size
    with jax.set_mesh(mesh):
        with map_logical_axis_names({'output': 'tp'}):
            base = nn.Linear(
                (2, 3), (width, 2), rngs=nn.Rngs(0),
                dtype=jnp.float32, kernel_metadata={'role': 'adapter'},
                precision=jax.lax.Precision.HIGHEST,
                axis_names=(None, None, 'output', None),
            )
            layer = make_adapter(cls, base)
            x = jnp.ones((3, 2, 3))
            sharding = NamedSharding(mesh, P(None, 'tp', None))
            output = jax.jit(lambda value: layer(value, out_sharding=sharding))(x)
            assert output.sharding.is_equivalent_to(sharding, 3)
            assert jnp.allclose(output, base(x), atol=1e-5)
            if cls in (nn.DoRALinear, nn.AdaLoRALinear):
                parameter = layer.lora_B.kernel
            elif cls is nn.LoHaLinear:
                parameter = layer.loha_B2
            elif cls is nn.VeRALinear:
                parameter = layer.vera_lambda_b
            else:
                parameter = layer.lokr_w1
            assert parameter.dtype == base.kernel.dtype
            assert parameter.metadata == {'role': 'adapter'}
            assert layer.precision == base.precision
            if cls is not nn.LoKrLinear:
                expected = P('tp', None) if cls is nn.VeRALinear else P(None, 'tp', None)
                assert parameter.value.sharding.is_equivalent_to(
                    NamedSharding(mesh, expected), parameter.ndim,
                )
        outside = make_adapter(cls, base)
        if cls in (nn.DoRALinear, nn.AdaLoRALinear):
            assert outside.lora_B.kernel.value.sharding.is_fully_replicated
        elif cls is nn.LoHaLinear:
            assert outside.loha_B2.value.sharding.is_fully_replicated
        elif cls is nn.VeRALinear:
            assert outside.vera_lambda_b.value.sharding.is_fully_replicated


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('layout', ['logical', 'physical', 'replicated'])
def test_adapter_explicit_mesh_and_sharding_overrides(cls, layout):
    mesh = Mesh(np.asarray(jax.devices()), ('tp',),
                axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh), map_logical_axis_names({'output': 'tp'}):
        base = nn.Linear((2, 3), (2 * mesh.size, 2), rngs=nn.Rngs(0),
                         axis_names=(None, None, 'output', None))
        kwargs = {'partition_spec': P()}
        if layout == 'logical':
            kwargs['axis_names'] = (None, None, 'output', None)
        elif layout == 'physical':
            kwargs['partition_spec'] = P(None, None, 'tp', None)
        layer = make_adapter(cls, base, **kwargs)
        if cls in (nn.DoRALinear, nn.AdaLoRALinear):
            parameter = layer.lora_B.kernel
        elif cls is nn.LoHaLinear:
            parameter = layer.loha_B2
        elif cls is nn.VeRALinear:
            parameter = layer.vera_lambda_b
        else:
            parameter = layer.lokr_w1
        if cls is nn.LoKrLinear or layout == 'replicated':
            assert parameter.value.sharding.is_fully_replicated
        else:
            spec = P('tp', None) if cls is nn.VeRALinear else P(None, 'tp', None)
            assert parameter.value.sharding.is_equivalent_to(
                NamedSharding(mesh, spec), parameter.ndim,
            )
        x = jnp.ones((3, 2, 3))
        target = NamedSharding(mesh, P(None, 'tp', None))
        output = jax.jit(lambda value: layer(value, out_sharding=target))(x)
        assert output.sharding.is_equivalent_to(target, output.ndim)
        assert jnp.allclose(output, base(x), atol=1e-5)
        replicated = NamedSharding(mesh, P())
        output = jax.jit(lambda value: layer(value, out_sharding=replicated))(x)
        assert output.sharding.is_fully_replicated
        assert jnp.allclose(output, base(x), atol=1e-5)
