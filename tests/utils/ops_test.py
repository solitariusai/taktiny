import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix

from taktiny.utils.ops import einsum, linear


def quantize(x):
    return qwix.quantize(x, 'int8')


def dense(x):
    return qwix.dequantize(x) if isinstance(x, qwix.QArray) else x


@pytest.mark.parametrize('qx,qw,qb', [(False, False, False), (True, False, False),
                                    (False, True, False), (True, True, True)])
@pytest.mark.parametrize('shape', [(3,), (2, 3), (2, 4, 3)])
def test_linear_dense_and_quantized(shape, qx, qw, qb):
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 7
    w = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 5
    b = jnp.arange(4, dtype=jnp.float32) / 3
    x, w, b = (quantize(v) if q else v for v, q in ((x, qx), (w, qw), (b, qb)))
    np.testing.assert_allclose(jax.jit(linear)(x, w, b), dense(x) @ dense(w) + dense(b), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('equation,shapes', [
    ('...i,io->...o', [(2, 3), (3, 4)]),
    ('ij,jk,kl->il', [(2, 3), (3, 4), (4, 2)]),
    ('ij->ji', [(2, 3)]),
    ('ij->', [(2, 3)]),
])
def test_einsum(quantized, equation, shapes):
    operands = [jnp.arange(np.prod(s), dtype=jnp.float32).reshape(s) / 7 for s in shapes]
    if quantized:
        operands[0] = quantize(operands[0])
    result = jax.jit(lambda *xs: einsum(equation, *xs))(*operands)
    np.testing.assert_allclose(result, jnp.einsum(equation, *(dense(x) for x in operands)), rtol=1e-5, atol=1e-5)


def test_arraylikes_and_dense_gradients():
    np.testing.assert_array_equal(linear([1., 2.], [[1.], [3.]]), [7.])
    np.testing.assert_array_equal(einsum('i,i->', [1., 2.], np.array([3., 4.])), 11.)
    x, w = jnp.ones((2, 3)), jnp.ones((3, 4))
    np.testing.assert_array_equal(jax.grad(lambda w: linear(x, w).sum())(w), jnp.full_like(w, 2))
    np.testing.assert_array_equal(jax.grad(lambda w: einsum('bi,io->bo', x, w).sum())(w), jnp.full_like(w, 2))


def test_validation_and_preferred_dtype():
    with pytest.raises(ValueError, match='rank'):
        linear(jnp.ones(3), jnp.ones(3))
    with pytest.raises(ValueError, match='dimension'):
        linear(jnp.ones(2), jnp.ones((3, 4)))
    with pytest.raises(ValueError, match='bias'):
        linear(jnp.ones(3), jnp.ones((3, 4)), jnp.ones((2, 4)))
    with pytest.raises(TypeError, match='string'):
        einsum(None, jnp.ones(3))
    with pytest.raises(NotImplementedError, match='optimize'):
        einsum('i->i', quantize(jnp.ones(3)), optimize=False)
    assert einsum('i->i', quantize(jnp.ones(3)), preferred_element_type=jnp.bfloat16).dtype == jnp.bfloat16


def test_quantized_einsum_operands_and_output_sharding():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ('model',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    x = quantize(jnp.ones((2, 3)))
    w = quantize(jnp.ones((3, 4)))
    for operation in (lambda x, w: linear(x, w, out_sharding=sharding),
                      lambda x, w: einsum('bi,io->bo', x, w, out_sharding=sharding)):
        output = jax.jit(operation)(x, w)
        np.testing.assert_allclose(output, dense(x) @ dense(w), rtol=1e-5)
        assert output.sharding.is_equivalent_to(sharding, output.ndim)


@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('equation,shapes', [
    ('ij,jk->ki', [(4, 8), (8, 4)]),
    ('ij->ji', [(4, 8)]),
    ('ij,jk,kl->il', [(4, 8), (8, 4), (4, 8)]),
])
def test_einsum_explicit_output_layout(quantized, equation, shapes, monkeypatch):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('model',),
                             axis_types=(jax.sharding.AxisType.Explicit,))
    output_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('model', None))

    def forbidden(*args, **kwargs):
        raise AssertionError('post-operation sharding constraint must not be used')

    monkeypatch.setattr(jax.lax, 'with_sharding_constraint', forbidden)
    with jax.set_mesh(mesh):
        arrays = tuple(jnp.arange(np.prod(s), dtype=jnp.float32).reshape(s) / 8 for s in shapes)
        if quantized:
            arrays = tuple(quantize(a) for a in arrays)
        result = jax.jit(lambda *xs: einsum(equation, *xs, out_sharding=output_sharding))(*arrays)
        expected = jnp.einsum(equation, *(dense(a) for a in arrays), out_sharding=output_sharding)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
        # JAX currently ignores out_sharding for unary transpose-only einsums.
        # Preserve native behavior instead of inserting a post-hoc constraint.
        assert result.sharding == expected.sharding
        if len(arrays) > 1:
            assert result.sharding == output_sharding


@pytest.mark.parametrize('quantized', [False, True])
def test_linear_explicit_output_layout(quantized, monkeypatch):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('model',),
                             axis_types=(jax.sharding.AxisType.Explicit,))
    output_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, 'model'))

    def forbidden(*args, **kwargs):
        raise AssertionError('post-operation sharding constraint must not be used')

    monkeypatch.setattr(jax.lax, 'with_sharding_constraint', forbidden)
    with jax.set_mesh(mesh):
        x, w = jnp.ones((4, 8)), jnp.ones((8, 4))
        x = jax.device_put(x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('model', None)))
        w = jax.device_put(w, output_sharding)
        if quantized:
            x, w = quantize(x), quantize(w)
        bias = jnp.ones(4)
        result = jax.jit(lambda x, w, b: linear(x, w, b, out_sharding=output_sharding))(x, w, bias)
        np.testing.assert_allclose(result, np.asarray(dense(x)) @ np.asarray(dense(w)) + np.asarray(bias), rtol=1e-5)
        assert result.sharding == output_sharding


@pytest.mark.parametrize('qtype', ['int8', 'int4', 'nf4', 'fp8', jnp.float8_e4m3fn])
@pytest.mark.parametrize('prequantized', [False, True])
def test_requested_quantization(qtype, prequantized):
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 13 - 0.5
    w = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 11 - 0.7
    if prequantized:
        x, w = quantize(x), quantize(w)
    qdtype = jnp.float8_e4m3fn if qtype == 'fp8' else qtype
    qx = qwix.quantize(dense(x), qdtype, channelwise_axes=(0, 1))
    qw = qwix.quantize(dense(w), qdtype, channelwise_axes=(1,))
    expected = jnp.einsum('...i,io->...o', dense(qx), dense(qw))
    for operation in (lambda x, w: linear(x, w, quant=qtype),
                      lambda x, w: einsum('...i,io->...o', x, w, quant=qtype)):
        result = jax.jit(operation)(x, w)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)
        assert result.dtype == jnp.float32


@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize('qtype', [None, 'int8'])
def test_result_dtype_after_bias(dtype, qtype):
    x = jnp.ones((2, 4), dtype=jnp.float32)
    w = jnp.ones((4, 3), dtype=jnp.bfloat16)
    b = jnp.ones(3, dtype=jnp.float32)
    result = linear(x, w, b, quant=qtype, preferred_element_type=dtype)
    assert result.dtype == dtype
    assert einsum('bi,io->bo', x, w, quant=qtype, preferred_element_type=dtype).dtype == dtype
    assert linear(x, w, b, quant=qtype).dtype == jnp.float32


def test_quantized_integer_arraylikes_and_implicit_scale_dtype():
    assert linear([[1, 2]], [[1], [2]], quant='int8').dtype == jnp.float32
    assert einsum('bi,io->bo', [[1, 2]], [[1], [2]], quant='int8').dtype == jnp.float32
    x = qwix.quantize(jnp.ones((2, 4), dtype=jnp.bfloat16), 'int8')
    w = qwix.quantize(jnp.ones((4, 3), dtype=jnp.bfloat16), 'int8')
    assert linear(x, w, quant='int8').dtype == jnp.bfloat16
    assert einsum('bi,io->bo', x, w, quant='int8').dtype == jnp.bfloat16


def test_quantized_einsum_multiple_contractions_and_sharding():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('model',),
                             axis_types=(jax.sharding.AxisType.Explicit,))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('model', None))
    with jax.set_mesh(mesh):
        x, w, v = jnp.ones((4, 8)), jnp.ones((8, 4)), jnp.ones((4, 8))
        operation = jax.jit(lambda x, w, v: einsum(
            'ij,jk,kl->li', x, w, v, quant='int8', optimize='optimal', out_sharding=sharding,
        ))
        result = operation(x, w, v)
        assert result.sharding == sharding
        np.testing.assert_allclose(result, jnp.full((8, 4), 32.), atol=0.8)
        ir = str(operation.lower(x, w, v).compiler_ir())
        integer_dots = [line for line in ir.splitlines() if 'stablehlo.dot_general' in line and 'xi8>' in line]
        assert len(integer_dots) == 2


def test_quantization_rejects_complex_contractions():
    with pytest.raises(TypeError, match='real-valued'):
        linear(jnp.ones(4, dtype=jnp.complex64), jnp.ones((4, 2)), quant='int8')


@pytest.mark.parametrize('qtype', [None, 'int8'])
def test_mixed_qarray_scale_dtypes(qtype):
    x = qwix.quantize(jnp.ones((2, 4), dtype=jnp.bfloat16), 'int8')
    w = jnp.ones((4, 3), dtype=jnp.float32)
    for operation in (lambda x, w: linear(x, w, quant=qtype),
                      lambda x, w: einsum('bi,io->bo', x, w, quant=qtype)):
        output = jax.jit(operation)(x, w)
        assert output.dtype == jnp.float32
        np.testing.assert_allclose(output, np.asarray(dense(x), dtype=np.float32) @ np.asarray(w), atol=0.08)


@pytest.mark.parametrize('qtype', ['int8', 'fp8'])
def test_requested_linear_quant_with_explicit_sharding(qtype):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('model',),
                             axis_types=(jax.sharding.AxisType.Explicit,))
    output_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, 'model'))
    with jax.set_mesh(mesh):
        x = jax.device_put(jnp.ones((4, 8)),
                           jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('model', None)))
        w = jax.device_put(jnp.ones((8, 4)), output_sharding)
        output = jax.jit(lambda x, w: linear(x, w, quant=qtype, out_sharding=output_sharding))(x, w)
        assert output.sharding == output_sharding
        np.testing.assert_allclose(output, jnp.full((4, 4), 8.), atol=0.2)


def test_unary_quant_einsum_preserves_jax_reduction():
    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    output = einsum('ij->j', x, quant='int8', preferred_element_type=jnp.bfloat16)
    assert output.dtype == jnp.bfloat16
    np.testing.assert_array_equal(output, jnp.sum(x, axis=0))
