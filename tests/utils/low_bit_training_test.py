import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix

from taktiny import nn
from taktiny.utils.ops import einsum, linear


@pytest.mark.parametrize('qtype', ['int8'])
@pytest.mark.parametrize('operation', ['linear', 'einsum'])
@pytest.mark.parametrize('backward', [False, True])
def test_qwix_training_eager_and_jit(qtype, operation, backward):
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


@pytest.mark.parametrize('qtype', ['int8'])
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


@pytest.mark.parametrize('qtype', ['int4', jnp.int4, 'nf4'])
@pytest.mark.parametrize('operation', ['linear', 'einsum', 'module'])
def test_low_bit_training_delegates_errors_to_qwix(monkeypatch, qtype, operation):
    from taktiny.utils import ops

    class BackendError(RuntimeError):
        pass

    def unavailable(lhs, rhs, dimensions, config):
        assert config.lhs_qtype == qtype
        assert config.rhs_qtype == qtype
        assert config.dlhs_grad_qtype == qtype
        raise BackendError('unsupported backend')

    monkeypatch.setattr(ops.dot_general_qt, 'dot_general_qt', unavailable)
    rule = qwix.QtRule(weight_qtype=qtype, act_qtype=qtype, bwd_qtype=qtype)
    x, w = jnp.ones((2, 3)), jnp.ones((3, 4))
    with pytest.raises(BackendError, match='unsupported backend'):
        if operation == 'module':
            nn.Linear(3, 4, quant=rule, rngs=nn.Rngs(0))(x)
        elif operation == 'linear':
            linear(x, w, quant=rule)
        else:
            einsum('bi,io->bo', x, w, quant=rule)
