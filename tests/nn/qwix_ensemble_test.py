import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from taktiny import nn
from taktiny.ensemble import quantize_model
from taktiny.ensemble.qwix import _current_module


def provider(**kwargs):
    return qwix.QtProvider([qwix.QtRule(
        weight_qtype='int8', act_qtype='int8', bwd_qtype='int8', **kwargs,
    )])


def test_linear_clone_jit_and_int8_backward():
    model = nn.Linear(8, 4, rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (3, 8))
    quantized = quantize_model(model, provider(), x)
    assert type(model) is nn.Linear
    assert isinstance(quantized.kernel, nn.Parameter)
    assert quantized.kernel is not model.kernel
    np.testing.assert_array_equal(quantized.kernel.value, model.kernel.value)
    forward = jax.jit(lambda m, x: m(x))
    np.testing.assert_allclose(forward(quantized, x), model(x), atol=0.08, rtol=0.05)
    gradient = jax.jit(jax.grad(lambda m, x: jnp.sum(m(x) ** 2), argnums=(0, 1)))
    dm, dx = gradient(quantized, x)
    assert jnp.all(jnp.isfinite(dm.kernel.value))
    assert jnp.all(jnp.isfinite(dx))
    # Check lowering, not just numerical similarity: forward and both backward
    # dot products must use integer operands.
    ir = str(gradient.lower(quantized, x).compiler_ir())
    integer_dots = [line for line in ir.splitlines() if 'stablehlo.dot_general' in line and 'xi8>' in line]
    assert len(integer_dots) >= 3
    updated = jax.tree.map(lambda p, g: p - 0.01 * g, quantized, dm)
    assert not jnp.array_equal(updated.kernel.value, quantized.kernel.value)
    assert jnp.isfinite(jnp.sum(forward(updated, x)))


class Network(nn.Module):
    def __init__(self):
        self.first = nn.Linear(8, 8, rngs=nn.Rngs(0))
        self.second = nn.Linear(8, 4, rngs=nn.Rngs(1))

    def __call__(self, x):
        return self.second(self.first(x))

    def encode(self, x):
        return self.first(x)


def test_paths_exclusions_and_selected_methods():
    model = Network()
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 7
    rules = qwix.QtProvider([
        qwix.QtRule(module_path='second', weight_qtype=None),
        qwix.QtRule(weight_qtype='int8', act_qtype='int8', bwd_qtype='int8'),
    ])
    quantized = quantize_model(model, rules, x, methods=('encode',))
    np.testing.assert_array_equal(quantized(x), model(x))
    assert not jnp.array_equal(quantized.encode(x), model.encode(x))
    quantized = quantize_model(model, rules, x)
    first = quantize_model(model.first, provider(), x)
    np.testing.assert_array_equal(quantized(x), model.second(first(x)))


@pytest.mark.parametrize('padding', ['VALID', 'SAME', 'SAME_LOWER'])
def test_convolution_forward_and_backward(padding):
    model = nn.Conv(2, 3, kernel_size=3, padding=padding, rngs=nn.Rngs(0))
    x = jax.random.normal(jax.random.key(2), (2, 6, 2))
    quantized = quantize_model(model, provider(), x)
    np.testing.assert_allclose(quantized(x), model(x), atol=0.08, rtol=0.05)
    dm, dx = jax.jit(jax.grad(lambda m, x: jnp.sum(m(x)), argnums=(0, 1)))(quantized, x)
    assert jnp.all(jnp.isfinite(dm.kernel.value))
    assert jnp.all(jnp.isfinite(dx))


def test_einsum_and_shared_parameters():
    class Einsum(nn.Module):
        def __init__(self):
            self.weight = nn.Linear(8, 4, rngs=nn.Rngs(0)).kernel
            self.shared = self.weight

        def __call__(self, x):
            return jnp.einsum('bi,io->bo', x, self.weight.value)

    model = Einsum()
    x = jnp.ones((2, 8))
    quantized = quantize_model(model, provider(op_names=('einsum',)), x)
    assert quantized.weight is quantized.shared
    np.testing.assert_allclose(quantized(x), model(x), atol=0.06)
    assert jnp.all(jnp.isfinite(jax.grad(lambda m: m(x).sum())(quantized).weight.value))


def test_validation_and_context_cleanup():
    model = nn.Linear(8, 4, rngs=nn.Rngs(0))
    x = jnp.ones((2, 8))
    with pytest.raises(NotImplementedError, match='QtProvider'):
        quantize_model(model, qwix.PtqProvider([]), x)
    with pytest.raises(NotImplementedError, match='Static'):
        quantize_model(model, provider(act_static_scale=True), x)
    with pytest.raises(NotImplementedError, match='Stochastic'):
        quantize_model(model, provider(bwd_stochastic_rounding='uniform'), x)
    with pytest.raises(ValueError, match='methods'):
        quantize_model(model, provider(), x, methods='__call__')
    with pytest.raises(ValueError, match='Unknown'):
        quantize_model(model, provider(), x, methods=('missing',))
    with pytest.raises(TypeError, match='contracting dimensions'):
        quantize_model(model, provider(), jnp.ones((2, 7)))
    assert _current_module.get() is None
    np.testing.assert_array_equal(model(x), model(x))


def test_warmup_does_not_change_state():
    class Stateful(nn.Module):
        def __init__(self):
            self.count = 0
            self.linear = nn.Linear(8, 4, rngs=nn.Rngs(0))

        def __call__(self, x):
            self.count += 1
            return self.linear(x)

    model = Stateful()
    quantized = quantize_model(model, provider(), jnp.ones((2, 8)))
    assert model.count == quantized.count == 0
    quantized(jnp.ones((2, 8)))
    assert quantized.count == 1
    assert model.count == 0


def test_structured_features_and_metadata():
    model = nn.Linear(
        (2, 4), (2, 3), rngs=nn.Rngs(0),
        axis_names=('a', 'b', 'c', 'd'), kernel_metadata={'tag': 'weight'},
    )
    x = jnp.ones((3, 2, 4))
    quantized = quantize_model(model, provider(), x)
    assert quantized.kernel.axis_names == model.kernel.axis_names
    assert quantized.kernel.metadata == model.kernel.metadata
    assert quantized(x).shape == (3, 2, 3)
    assert jnp.all(jnp.isfinite(jax.grad(lambda m: m(x).sum())(quantized).kernel.value))


def test_prequantized_and_rewrapped_models_are_rejected():
    x = jnp.ones((2, 8))
    model = nn.Linear(8, 4, rngs=nn.Rngs(0), quant='int8')
    with pytest.raises(ValueError, match='QArrays'):
        quantize_model(model, provider(), x)
    model = nn.Linear(8, 4, rngs=nn.Rngs(0))
    quantized = quantize_model(model, provider(), x)
    with pytest.raises(ValueError, match='already wrapped'):
        quantize_model(quantized, provider(), x)


def test_empty_rules_are_identity():
    model = Network()
    x = jnp.ones((2, 8))
    quantized = quantize_model(model, qwix.QtProvider([]), x)
    np.testing.assert_array_equal(quantized(x), model(x))


def test_sharding_is_preserved():
    mesh = Mesh(np.asarray(jax.devices()), ('model',))
    output_size = 4 * jax.device_count()
    with jax.set_mesh(mesh):
        model = nn.Linear(
            8, output_size, partition_spec=P(None, 'model'), rngs=nn.Rngs(0),
        )
        x = jax.device_put(jnp.ones((2, 8)), NamedSharding(mesh, P()))
        quantized = quantize_model(model, provider(), x)
        assert quantized.kernel.value.sharding == model.kernel.value.sharding
        assert quantized.kernel.partition_spec == model.kernel.partition_spec
        output_sharding = NamedSharding(mesh, P(None, 'model'))
        output = jax.jit(lambda m, x: m(x, out_sharding=output_sharding))(quantized, x)
        assert output.sharding == output_sharding
        assert jnp.all(jnp.isfinite(output))
