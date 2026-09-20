import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix

from taktiny import nn
from taktiny.utils.ops import einsum, linear


@pytest.mark.parametrize('qtype', ['int4', jnp.int4, 'nf4'])
@pytest.mark.parametrize('operation', ['linear', 'einsum'])
@pytest.mark.parametrize('backward', [False, True])
def test_low_bit_training_eager_and_jit(qtype, operation, backward):
    rule = qwix.QtRule(weight_qtype=qtype, act_qtype=qtype, bwd_qtype=qtype if backward else None)
    w = jax.random.uniform(jax.random.key(0), (3, 6))
    x = w.T

    def loss(w, x):
        output = linear(x, w, quant=rule) if operation == 'linear' else einsum('bi,io->bo', x, w, quant=rule)
        return output.mean()

    gradient = jax.grad(loss, argnums=(0, 1))
    eager = gradient(w, x)
    compiled = jax.jit(gradient)(w, x)
    for actual, expected in zip(compiled, eager):
        assert jnp.all(jnp.isfinite(actual)) and jnp.any(actual != 0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert loss(w - 0.5 * eager[0], x) < loss(w, x)


@pytest.mark.parametrize('qtype', ['int8', 'int4', 'nf4'])
@pytest.mark.parametrize('structured', [False, True])
def test_linear_module_training_rules(qtype, structured):
    rule = qwix.QtRule(weight_qtype=qtype, act_qtype=qtype, bwd_qtype=qtype)
    inputs, outputs = ((2, 3), (2, 2)) if structured else (3, 6)
    layer = nn.Linear(inputs, outputs, quant=qwix.QtProvider([rule]), rngs=nn.Rngs(0))
    assert not isinstance(layer.kernel.value, qwix.QArray)
    x = jnp.ones((4, *layer.in_features))
    gradient = jax.jit(jax.grad(lambda m: jnp.mean(m(x) ** 2)))(layer)
    assert gradient.kernel.shape == layer.kernel.shape
    assert jnp.all(jnp.isfinite(gradient.kernel.value))
    assert jnp.any(gradient.kernel.value != 0)
    updated = jax.tree.map(lambda p, g: p - 0.01 * g, layer, gradient)
    assert not jnp.array_equal(updated.kernel.value, layer.kernel.value)


def test_linear_qt_custom_operation_conflict():
    with pytest.raises(ValueError, match='custom dot_general'):
        nn.Linear(3, 6, quant=qwix.QtRule(weight_qtype='int8'),
                  dot_general=jax.lax.dot_general, rngs=nn.Rngs(0))


@pytest.mark.parametrize('qtype', ['int4', 'nf4'])
def test_low_bit_backward_matches_quantized_matrix_products(qtype):
    x = jax.random.uniform(jax.random.key(0), (6, 3))
    w = jax.random.uniform(jax.random.key(1), (3, 6))
    rule = qwix.QtRule(weight_qtype=qtype, act_qtype=qtype, bwd_qtype=qtype)
    dx, dw = jax.grad(lambda x, w: linear(x, w, quant=rule).mean(), argnums=(0, 1))(x, w)
    g = jnp.full((6, 6), 1 / 36)

    def rounded(value, channel_axis):
        return qwix.dequantize(qwix.quantize(value, qtype, channelwise_axes=(channel_axis,)))

    np.testing.assert_allclose(dx, rounded(g, 0) @ rounded(w, 0).T, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dw, rounded(x, 1).T @ rounded(g, 1), rtol=1e-5, atol=1e-6)
