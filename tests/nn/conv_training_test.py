import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix

from taktiny import nn


@pytest.mark.parametrize('cls', [nn.Conv, nn.ConvTranspose])
@pytest.mark.parametrize('qtype', ['int8'])
@pytest.mark.parametrize('groups', [1])
@pytest.mark.parametrize('backward', [False, True])
def test_convolution_training(cls, qtype, groups, backward):
    rule = qwix.QtRule(weight_qtype=qtype, act_qtype=qtype, bwd_qtype=qtype if backward else None)
    layer = cls((2, 4), (2, 4), 3, groups=groups, stride=2, padding='SAME',
                quant=qwix.QtProvider([rule]), rngs=nn.Rngs(0))
    assert not isinstance(layer.kernel.value, qwix.QArray)
    x = jax.random.normal(jax.random.key(1), (2, 6, 2, 4))
    derivative = jax.value_and_grad(lambda m, x: jnp.mean(m(x) ** 2), argnums=(0, 1))
    loss, (dm, dx) = derivative(layer, x)
    jit_loss, (jit_dm, jit_dx) = jax.jit(derivative)(layer, x)
    np.testing.assert_allclose(jit_loss, loss, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(jit_dm.kernel.value, dm.kernel.value, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(jit_dx, dx, rtol=1e-4, atol=1e-5)
    assert dm.kernel.shape == layer.kernel.shape
    assert dx.shape == x.shape
    assert jnp.all(jnp.isfinite(dm.kernel.value)) and jnp.any(dm.kernel.value != 0)
    assert jnp.all(jnp.isfinite(dx)) and jnp.any(dx != 0)
    updated = jax.tree.map(lambda p, g: p - 0.1 * g, layer, dm)
    assert jnp.mean(updated(x) ** 2) < loss


@pytest.mark.parametrize('cls', [nn.Conv, nn.ConvTranspose])
@pytest.mark.parametrize('qtype', ['int8', 'int4', 'nf4'])
@pytest.mark.parametrize('groups', [1, (2, 2)])
def test_convolution_delegates_errors_to_qwix(monkeypatch, cls, qtype, groups):
    from taktiny.utils import ops

    class BackendError(RuntimeError):
        pass

    def unavailable(lhs, rhs, config, *args):
        assert config.lhs_qtype == qtype
        assert config.rhs_qtype == qtype
        assert config.dlhs_grad_qtype == qtype
        raise BackendError('unsupported backend')

    monkeypatch.setattr(ops.conv_general_qt, 'conv_general_qt', unavailable)
    rule = qwix.QtRule(weight_qtype=qtype, act_qtype=qtype, bwd_qtype=qtype)
    layer = cls((2, 2), (2, 2), (2, 3), groups=groups, padding='SAME',
                quant=rule, rngs=nn.Rngs(0))
    x = jnp.ones((4, 5, 2, 2))  # Unbatched.
    with pytest.raises(BackendError, match='unsupported backend'):
        layer(x)


@pytest.mark.parametrize('cls', [nn.Conv, nn.ConvTranspose])
def test_convolution_qt_validation(cls):
    with pytest.raises(NotImplementedError, match='tiled'):
        cls(2, 4, 3, quant=qwix.QtRule(weight_qtype='int8', tile_size=2), rngs=nn.Rngs(0))
    with pytest.raises(ValueError, match='custom dot_general'):
        cls(2, 4, 3, quant=qwix.QtRule(weight_qtype='int8'),
            dot_general=jax.lax.conv_general_dilated, rngs=nn.Rngs(0))
