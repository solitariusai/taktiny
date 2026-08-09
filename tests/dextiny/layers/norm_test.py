import dextiny as dx
import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.layers import AdaXNorm#, SpatialNorm


def linear(x, weight, bias):
    output = jax.lax.dot_general(
        x,
        weight,
        (((x.ndim - 1,), (0,)), ((), ())),
    )
    return output + bias


def adaptive_norm(
    x,
    vec,
    weight,
    bias,
    *,
    norm_type,
    eps,
):
    if norm_type == "layer_norm":
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) * jax.lax.rsqrt(variance + eps)
    else:
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normalized = x * jax.lax.rsqrt(variance + eps)
    modulation = linear(jax.nn.silu(vec), weight, bias)
    return normalized, modulation


def adaptive_norm_chunks(
    x,
    vec,
    weight,
    bias,
    *,
    norm_type,
    eps,
    num_chunks,
):
    normalized, modulation = adaptive_norm(
        x,
        vec,
        weight,
        bias,
        norm_type=norm_type,
        eps=eps,
    )
    return normalized, tuple(jnp.split(modulation, num_chunks, axis=-1))


def group_norm(x, weight, bias, *, num_groups, eps):
    batch, height, width, channels = x.shape
    grouped = x.reshape(
        batch,
        height,
        width,
        num_groups,
        channels // num_groups,
    )
    mean = jnp.mean(grouped, axis=(1, 2, 4), keepdims=True)
    variance = jnp.var(grouped, axis=(1, 2, 4), keepdims=True)
    normalized = (grouped - mean) * jax.lax.rsqrt(variance + eps)
    normalized = normalized.reshape(batch, height, width, channels)
    return normalized * weight + bias


def conv1x1(x, weight, bias):
    output = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=weight,
        window_strides=(1, 1),
        padding=((0, 0), (0, 0)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=1,
    )
    return output + bias


def spatial_norm(
    features,
    conditioning,
    norm_weight,
    norm_bias,
    y_weight,
    y_bias,
    b_weight,
    b_bias,
    *,
    num_groups,
    eps,
):
    conditioning = jax.image.resize(
        conditioning,
        shape=features.shape,
        method="nearest",
    )
    normalized = group_norm(
        features,
        norm_weight,
        norm_bias,
        num_groups=num_groups,
        eps=eps,
    )
    return (
        normalized * conv1x1(conditioning, y_weight, y_bias)
        + conv1x1(conditioning, b_weight, b_bias)
    )


@pytest.mark.parametrize("norm_type", ("layer_norm", "rms_norm"))
def test_adaptive_layer_norm_matches_reference_trace(norm_type):
    hidden = dx.AbstractArray("B S D", dtype="float32")
    reference = hidden >> dx.AbstractModule(
        adaptive_norm,
        dx.AbstractArray("B E", dtype="float32", name="conditioning"),
        dx.AbstractArray("E O", dtype="float32", name="weight"),
        dx.AbstractArray("O", dtype="float32", name="bias"),
        name=f"adaptive_{norm_type}",
        kwargs={"norm_type": norm_type, "eps": 1e-6},
    )
    compiled = reference.compile(B=2, S=4, D=8, E=6, O=12)
    module = AdaXNorm(
        embedding_dim=6,
        out_dim=12,
        norm=norm_type,
        eps=1e-6,
        rngs=nn.Rngs(0),
    )
    conditioning = jnp.ones((2, 6), dtype=jnp.float32)

    assert compiled.verify(
        module,
        conditioning,
    ), compiled.report(module, conditioning).render()


def test_adaptive_layer_norm_chunks_matches_reference_trace():
    hidden = dx.AbstractArray("B S D", dtype="float32")
    reference = hidden >> dx.AbstractModule(
        adaptive_norm_chunks,
        dx.AbstractArray("B E", dtype="float32", name="conditioning"),
        dx.AbstractArray("E C*O", dtype="float32", name="weight"),
        dx.AbstractArray("C*O", dtype="float32", name="bias"),
        name="adaptive_layer_norm_chunks",
        kwargs={
            "norm_type": "layer_norm",
            "eps": 1e-6,
            "num_chunks": 3,
        },
    )
    compiled = reference.compile(B=2, S=4, D=8, E=6, C=3, O=4)
    module = AdaXNorm(
        embedding_dim=6,
        out_dim=12,
        norm='layernorm',
        eps=1e-6,
        rngs=nn.Rngs(0),
    )
    conditioning = jnp.ones((2, 6), dtype=jnp.float32)

    def actual(x, vec):
        normalized, modulation = module(x, vec)
        return normalized, tuple(jnp.split(modulation, 3, axis=-1))

    assert compiled.verify(
        actual,
        conditioning,
    ), compiled.report(actual, conditioning).render()


# def test_spatial_norm_matches_reference_trace():
#     features = dx.AbstractArray("B H W C", dtype="float32")
#     reference = features >> dx.AbstractModule(
#         spatial_norm,
#         dx.AbstractArray("B H W C", dtype="float32", name="conditioning"),
#         dx.AbstractArray("C", dtype="float32", name="norm_weight"),
#         dx.AbstractArray("C", dtype="float32", name="norm_bias"),
#         dx.AbstractArray("1 1 C C", dtype="float32", name="y_weight"),
#         dx.AbstractArray("C", dtype="float32", name="y_bias"),
#         dx.AbstractArray("1 1 C C", dtype="float32", name="b_weight"),
#         dx.AbstractArray("C", dtype="float32", name="b_bias"),
#         name="spatial_norm",
#         kwargs={"num_groups": 32, "eps": 1e-6},
#     )
#     compiled = reference.compile(B=2, H=4, W=4, C=32)
#     module = SpatialNorm(
#         f_channels=32,
#         zq_channels=32,
#         rngs=nn.Rngs(0),
#     )
#     conditioning = jnp.ones((2, 4, 4, 32), dtype=jnp.float32)

#     assert compiled.verify(
#         module.forward,
#         conditioning,
#     ), compiled.report(module.forward, conditioning).render()
