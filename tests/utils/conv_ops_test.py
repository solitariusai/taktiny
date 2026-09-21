import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix

from taktiny.utils import ops

LAYOUT = ('NWC', 'WIO', 'NWC')


@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('padding', ['VALID', 'SAME', 'SAME_LOWER', ((1, 2),)])
def test_dense_convolution(groups, padding):
    x = np.arange(32, dtype=np.float32).reshape(1, 8, 4)
    w = np.ones((3, 4 // groups, 6), dtype=np.float32)
    kwargs = dict(dimension_numbers=LAYOUT, feature_group_count=groups, rhs_dilation=(2,))
    expected = jax.lax.conv_general_dilated(x, w, (2,), padding, **kwargs)
    def operation(x, w):
        return ops.conv_general_dilated(x, w, (2,), padding, **kwargs)
    np.testing.assert_allclose(operation(x, w), expected)
    np.testing.assert_allclose(jax.jit(operation)(x, w), expected)


@pytest.mark.parametrize('kind', ['string', 'rule', 'provider', 'qarray'])
def test_quantized_convolution(kind):
    x = jnp.arange(16, dtype=jnp.float32).reshape(1, 8, 2) / 16
    w = jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4) / 24
    qw = qwix.quantize(w, 'int8', channelwise_axes=(2,))
    rule = qwix.QuantizationRule(weight_qtype='int8')
    quant = {'string': 'int8', 'rule': rule, 'provider': qwix.PtqProvider([rule]), 'qarray': None}[kind]
    actual = ops.conv_general_dilated(x, qw if kind == 'qarray' else w, (1,), 'SAME',
                                     dimension_numbers=LAYOUT, quant=quant)
    expected = qwix.conv_general_dilated(x, qw, (1,), 'SAME', dimension_numbers=LAYOUT)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_qt_gradients():
    rule = qwix.QtRule(weight_qtype='int8', act_qtype='int8', bwd_qtype='int8')
    x, w = jnp.ones((1, 8, 2)), jnp.ones((3, 2, 4))

    def loss(x, w):
        return ops.conv_general_dilated(x, w, (1,), 'SAME', dimension_numbers=LAYOUT,
                                       quant=qwix.QtProvider([rule])).mean()

    grad = jax.grad(loss, argnums=(0, 1))
    eager, compiled = grad(x, w), jax.jit(grad)(x, w)
    for a, b in zip(eager, compiled):
        np.testing.assert_allclose(a, b, atol=1e-6)
        assert jnp.all(jnp.isfinite(a)) and jnp.any(a != 0)


def test_dtype_and_rule_filtering():
    x, w = jnp.ones((1, 8, 2)), jnp.ones((3, 2, 4))
    excluded = qwix.QtRule(module_path='other', weight_qtype='nf4')
    actual = ops.conv_general_dilated(x, w, (1,), 'SAME', dimension_numbers=LAYOUT,
                                     quant=excluded, preferred_element_type=jnp.bfloat16)
    expected = jax.lax.conv_general_dilated(x, w, (1,), 'SAME', dimension_numbers=LAYOUT)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_allclose(actual.astype(jnp.float32), expected)


@pytest.mark.parametrize('qtype', ['int4', 'nf4'])
def test_qt_errors_propagate(monkeypatch, qtype):
    def unavailable(*args, **kwargs):
        raise RuntimeError('backend unsupported')

    monkeypatch.setattr(ops.conv_general_qt, 'conv_general_qt', unavailable)
    with pytest.raises(RuntimeError, match='backend unsupported'):
        ops.conv_general_dilated(jnp.ones((1, 8, 2)), jnp.ones((3, 2, 4)),
                                 (1,), 'SAME', dimension_numbers=LAYOUT,
                                 quant=qwix.QtRule(weight_qtype=qtype))


def test_output_sharding():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ('data',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    result = ops.conv_general_dilated(jnp.ones((1, 8, 2)), jnp.ones((3, 2, 4)),
                                      (1,), 'SAME', dimension_numbers=LAYOUT,
                                      out_sharding=sharding)
    assert result.sharding.is_equivalent_to(sharding, result.ndim)
