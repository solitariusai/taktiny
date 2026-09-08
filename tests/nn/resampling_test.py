import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn


def test_downsample_uses_scale_factor_as_divisor():
    layer = nn.Downsample(scale_factor=2, method='linear')
    x = jnp.arange(8, dtype=jnp.float32)[:, None]

    output = jax.jit(layer)(x)
    expected = jax.image.resize(
        x,
        (4, 1),
        method='linear',
        antialias=True,
    )

    assert output.shape == (4, 1)
    assert jnp.allclose(output, expected)


def test_downsample_supports_anisotropic_batched_inputs():
    layer = nn.Downsample(
        scale_factor=(2, 3),
        method='nearest',
    )
    x = jnp.arange(2 * 8 * 9 * 3, dtype=jnp.float32).reshape(
        2,
        8,
        9,
        3,
    )

    output = jax.jit(layer)(x)

    assert output.shape == (2, 4, 3, 3)


def test_downsample_supports_exact_and_preserved_sizes():
    layer = nn.Downsample(size=(3, None), method='cubic')
    x = jnp.ones((2, 7, 5, 4), dtype=jnp.float32)

    output = layer(x)

    assert output.shape == (2, 3, 5, 4)


def test_downsample_is_differentiable_and_jittable():
    layer = nn.Downsample(scale_factor=2)
    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4, 1)

    gradient = jax.jit(jax.grad(lambda value: jnp.sum(layer(value))))(x)

    assert gradient.shape == x.shape
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize('cls', [nn.Upsample, nn.Downsample])
@pytest.mark.parametrize('axis_type', [AxisType.Auto, AxisType.Explicit])
def test_resampling_applies_explicit_output_sharding(cls, axis_type):
    mesh = Mesh(np.asarray(jax.devices()), ('data',), axis_types=(axis_type,))
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P('data', None, None))
        layer = cls(scale_factor=2)
        x = jax.device_put(jnp.ones((2 * mesh.size, 8, 2)), sharding)
        for target in (sharding, NamedSharding(mesh, P())):
            output = jax.jit(
                lambda value: layer(value, out_sharding=target)
            )(x)
            assert output.sharding.is_equivalent_to(target, output.ndim)
            assert jnp.allclose(output, 1)


def test_downsample_validates_configuration_and_target_size():
    with pytest.raises(ValueError, match='mutually exclusive'):
        nn.Downsample(size=4, scale_factor=2)
    with pytest.raises(ValueError, match='greater than or equal to 1'):
        nn.Downsample(scale_factor=0.5)
    with pytest.raises(ValueError, match='cannot exceed'):
        nn.Downsample(size=9)(jnp.ones((8, 2)))
    with pytest.raises(ValueError, match='finite and positive'):
        nn.Downsample(scale_factor=float('inf'))


def test_upsample_retains_existing_scale_semantics():
    layer = nn.Upsample(scale_factor=2, method='nearest')
    x = jnp.asarray([[1.0], [2.0]])

    output = jax.jit(layer)(x)

    assert jnp.array_equal(
        output[:, 0],
        jnp.asarray([1.0, 1.0, 2.0, 2.0]),
    )


@pytest.mark.parametrize('cls', [nn.Upsample, nn.Downsample])
@pytest.mark.parametrize('size', [True, (2, False), '23', 2.5])
def test_resampling_rejects_invalid_size_types(cls, size):
    with pytest.raises(TypeError, match='size'):
        cls(size=size)


@pytest.mark.parametrize('cls', [nn.Upsample, nn.Downsample])
@pytest.mark.parametrize('size', [(), 0, (-1, 2)])
def test_resampling_rejects_invalid_sizes(cls, size):
    with pytest.raises(ValueError, match='size'):
        cls(size=size)


@pytest.mark.parametrize('cls', [nn.Upsample, nn.Downsample])
@pytest.mark.parametrize('scale', [True, '2', (), (0,), (-1,), float('nan')])
def test_resampling_rejects_invalid_scales(cls, scale):
    with pytest.raises((TypeError, ValueError), match='scale_factor'):
        cls(scale_factor=scale)


@pytest.mark.parametrize('cls', [nn.Upsample, nn.Downsample])
def test_resampling_validates_method_and_antialias(cls):
    with pytest.raises(ValueError):
        cls(method='not-a-method')
    with pytest.raises(TypeError, match='method'):
        cls(method=42)
    with pytest.raises(TypeError, match='antialias'):
        cls(antialias='false')


@pytest.mark.parametrize('method', ['BILINEAR', 'bicubic', 'triangle',
                                  jax.image.ResizeMethod.LANCZOS3])
def test_resize_methods_and_precision_match_jax(method):
    layer = nn.Downsample(size=(3, 4), method=method, antialias=False,
                          precision=jax.lax.Precision.DEFAULT)
    x = jnp.arange(8 * 9 * 2, dtype=jnp.float32).reshape(8, 9, 2)
    expected = jax.image.resize(x, (3, 4, 2), layer.method, antialias=False,
                                precision=jax.lax.Precision.DEFAULT)
    assert jnp.allclose(jax.jit(layer)(x), expected)
    assert layer.precision == jax.lax.Precision.DEFAULT


def test_resampling_scalar_rank_rounding_minimum_and_preserved_axes():
    assert nn.Upsample()(jnp.ones((3, 1))).shape == (6, 1)
    assert nn.Downsample()(jnp.ones((3, 1))).shape == (1, 1)
    assert nn.Upsample(scale_factor=1.5)(jnp.ones((3, 1))).shape == (4, 1)
    assert nn.Upsample(scale_factor=0.01)(jnp.ones((3, 1))).shape == (1, 1)
    assert nn.Downsample(scale_factor=100)(jnp.ones((3, 1))).shape == (1, 1)
    assert nn.Upsample(size=[None, 7])(jnp.ones((3, 4, 2))).shape == (3, 7, 2)
    # A scalar describes 1-D input; the leading dimension here is a batch.
    assert nn.Upsample(scale_factor=2)(jnp.ones((3, 4, 2))).shape == (3, 8, 2)
    with pytest.raises(ValueError, match='rank'):
        nn.Upsample(scale_factor=2)(jnp.ones((2, 3, 4, 1)))


@pytest.mark.parametrize('cls', [nn.Upsample, nn.Downsample])
def test_resampling_rejects_empty_spatial_input(cls):
    with pytest.raises(ValueError, match='nonempty'):
        cls()(jnp.ones((0, 2)))


def test_resampling_repr_and_nearest_dtype():
    layer = nn.Upsample(size=(None, 6), antialias=False)
    assert 'size=*×6' in layer.extra_repr()
    assert 'antialias=False' in layer.extra_repr()
    assert layer(jnp.ones((3, 4, 2), dtype=jnp.int32)).dtype == jnp.int32
