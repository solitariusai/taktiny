# Copyright 2026 Shinapri.
# Copyright 2025 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import qwix
from jax.lax import PrecisionLike
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from jax.typing import DTypeLike

from taktiny.utils.quantization import normalize_qtype
from taktiny.utils.typing import Array, ArrayLike


def _as_operand(value: ArrayLike | qwix.QArray) -> Array | qwix.QArray:
    return value if isinstance(value, qwix.QArray) else jnp.asarray(value)


def _dense(value: Array | qwix.QArray) -> Array:
    return qwix.dequantize(value) if isinstance(value, qwix.QArray) else value


def _promote(value: Array | qwix.QArray, dtype: DTypeLike) -> Array | qwix.QArray:
    if isinstance(value, qwix.QArray):
        return replace(value, scale=value.scale.astype(dtype))
    return value.astype(dtype)


def _dtype(*values: Array | qwix.QArray, quantized: bool = False) -> jnp.dtype:
    dtypes = [value.scale.dtype if isinstance(value, qwix.QArray) else value.dtype for value in values]
    # JAX has no implicit promotion between FP8 and ordinary floating types.
    dtypes = [jnp.bfloat16 if 'float8' in str(dtype) else dtype for dtype in dtypes]
    result = jnp.result_type(*dtypes)
    if quantized and not jnp.issubdtype(result, jnp.inexact):
        result = jnp.dtype(jnp.float32)
    return result


def _quantized_dot(
    lhs: Array,
    rhs: Array,
    dimension_numbers: Any,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    quant: DTypeLike,
    out_sharding: NamedSharding | None = None,
) -> Array:
    """Quantize contraction operands with scales along non-contracting axes."""
    if jnp.iscomplexobj(lhs) or jnp.iscomplexobj(rhs):
        raise TypeError('quantized operations require real-valued operands')
    dtype = _dtype(lhs, rhs, quantized=True)
    lhs_contract, rhs_contract = dimension_numbers[0]
    qlhs = qwix.quantize(
        lhs.astype(dtype), quant,
        channelwise_axes=tuple(i for i in range(lhs.ndim) if i not in lhs_contract),
    )
    qrhs = qwix.quantize(
        rhs.astype(dtype), quant,
        channelwise_axes=tuple(i for i in range(rhs.ndim) if i not in rhs_contract),
    )
    if (
        out_sharding is not None
        and AxisType.Explicit in out_sharding.mesh.axis_types
        and quant != 'nf4'
    ):
        # Scale tensors must broadcast into the requested output layout, which
        # may differ from the operand layouts used during calibration.
        replicated = NamedSharding(out_sharding.mesh, PartitionSpec())
        qlhs = replace(qlhs, scale=jax.reshard(qlhs.scale, replicated))
        qrhs = replace(qrhs, scale=jax.reshard(qrhs.scale, replicated))
    return qwix.dot_general(
        qlhs, qrhs, dimension_numbers, precision=precision,
        preferred_element_type=preferred_element_type, out_sharding=out_sharding,
    )


def linear(
    x: ArrayLike | qwix.QArray,
    w: ArrayLike | qwix.QArray,
    b: ArrayLike | qwix.QArray | None = None,
    *,
    quant: DTypeLike | None = None,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: NamedSharding | None = None,
) -> Array:
    """Compute ``x @ w + b`` with optional quantized multiplication.

    ``x`` has shape ``(..., input_features)`` and ``w`` has shape
    ``(input_features, output_features)``, matching Taktiny Linear's kernel
    layout. Leading input dimensions are preserved. Bias must broadcast to
    the result; a quantized bias is dequantized before addition.

    ``quant=None`` uses JAX for dense inputs and Qwix for QArray inputs.
    A quantization dtype such as ``'int8'``, ``'int4'``, ``'nf4'``, ``'fp8'``,
    or ``jnp.float8_e4m3fn`` quantizes both multiplication operands with dynamic
    per-channel scales. Existing QArrays are dequantized and requantized to
    that format. Bias is added in floating point for quantized computation.

    Returns a dense JAX array. ``preferred_element_type`` selects its dtype,
    including after bias addition; otherwise the dtype is promoted from the
    inputs and bias, using QArray scale dtypes. Raw FP8 inputs promote to BF16;
    integer-only inputs with ``quant`` set promote to FP32. Accumulation may
    use a higher precision than the returned dtype. ``out_sharding`` is passed
    to the dot operation. Quantization does not install training-specific
    gradient rules; use the quantized-training integration for that purpose.

    Example:
        >>> linear(jnp.ones((2, 3)), jnp.ones((3, 4))).shape
        (2, 4)
        >>> linear(jnp.ones((2, 3)), jnp.ones((3, 4)), quant='int8',
        ...        preferred_element_type=jnp.bfloat16).dtype
        dtype(bfloat16)
    """
    x, w = _as_operand(x), _as_operand(w)
    if x.ndim < 1 or w.ndim != 2:
        raise ValueError('linear requires x with rank >= 1 and a rank-2 weight')
    if x.shape[-1] != w.shape[0]:
        raise ValueError('the last input dimension must match the first weight dimension')
    dimension_numbers = (((x.ndim - 1,), (0,)), ((), ()))
    bias = None if b is None else _dense(_as_operand(b))
    implicit_dtype = _dtype(x, w, *((bias,) if bias is not None else ()), quantized=quant is not None)
    result_dtype = implicit_dtype if preferred_element_type is None else jnp.dtype(preferred_element_type)
    compute_dtype = jnp.result_type(implicit_dtype, result_dtype)
    if quant is not None:
        output = _quantized_dot(
            _dense(x), _dense(w), dimension_numbers, quant=normalize_qtype(quant),
            precision=precision, preferred_element_type=compute_dtype,
            out_sharding=out_sharding,
        )
    elif isinstance(x, qwix.QArray) or isinstance(w, qwix.QArray):
        output = qwix.dot_general(
            _promote(x, compute_dtype), _promote(w, compute_dtype), dimension_numbers,
            precision=precision, preferred_element_type=compute_dtype,
            out_sharding=out_sharding,
        )
    else:
        output = jax.lax.dot_general(
            x.astype(compute_dtype), w.astype(compute_dtype), dimension_numbers,
            precision=precision, preferred_element_type=compute_dtype,
            out_sharding=out_sharding,
        )
    if bias is not None:
        if jnp.broadcast_shapes(output.shape, bias.shape) != output.shape:
            raise ValueError('bias must broadcast to the linear output without adding dimensions')
        output = output + bias
    return output.astype(result_dtype)


def einsum(
    subscripts: str,
    *operands: ArrayLike | qwix.QArray,
    quant: DTypeLike | None = None,
    optimize: Any = 'auto',
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: NamedSharding | None = None,
) -> Array:
    """Evaluate an Einstein summation with optional quantized contractions.

    Uses the string-equation form of ``jax.numpy.einsum``. Dense array-like
    inputs are converted to JAX arrays, while QArrays retain their quantized
    representation when ``quant`` and ``out_sharding`` are unset. In that case,
    QArray inputs select ``qwix.einsum`` and its dequantization fallbacks.
    ``quant`` accepts a Qwix dtype or an alias such as ``'int8'`` or ``'fp8'``.
    When set, each dot-product contraction quantizes both operands, including
    intermediate results, using dynamic scales along non-contracting axes.
    Existing QArrays are dequantized before this process. Unary reductions
    and rearrangements use ordinary JAX operations. The result is always a
    dense JAX array. No training-specific gradient rule is installed.

    ``precision`` is forwarded to the selected backend. ``optimize`` is
    supported for dense inputs and when ``quant`` is set. With ``quant=None``,
    unsharded QArray inputs require ``'auto'`` because
    Qwix manages its own contraction path. ``preferred_element_type``
    selects the returned dtype; otherwise input dtypes (QArray scale dtypes)
    are promoted. Raw FP8 promotes to BF16, and integer-only inputs promote
    to FP32 when ``quant`` is set. Accumulation may use higher precision.
    ``out_sharding`` is passed to JAX's einsum. If ``quant`` is unset, this
    sharded path dequantizes QArrays and uses dense computation.

    Example:
        >>> einsum('...i,io->...o', jnp.ones((2, 3)), jnp.ones((3, 4))).shape
        (2, 4)
    """
    if not isinstance(subscripts, str):
        raise TypeError('subscripts must be an einsum equation string')
    arrays = tuple(_as_operand(operand) for operand in operands)
    if not arrays:
        raise ValueError('einsum requires at least one operand')
    implicit_dtype = _dtype(*arrays, quantized=quant is not None)
    result_dtype = implicit_dtype if preferred_element_type is None else jnp.dtype(preferred_element_type)
    compute_dtype = jnp.result_type(implicit_dtype, result_dtype)
    if quant is not None or out_sharding is not None:
        dot = jax.lax.dot_general if quant is None else partial(_quantized_dot, quant=normalize_qtype(quant))
        output = jnp.einsum(
            subscripts,
            *(_dense(operand).astype(compute_dtype) for operand in arrays),
            optimize=optimize, precision=precision,
            preferred_element_type=compute_dtype, out_sharding=out_sharding,
            _dot_general=dot,
        )
    elif any(isinstance(operand, qwix.QArray) for operand in arrays):
        if not isinstance(optimize, str) or optimize != 'auto':
            raise NotImplementedError("QArray einsum currently requires optimize='auto'")
        output = qwix.einsum(
            subscripts, *(_promote(operand, compute_dtype) for operand in arrays), precision=precision,
            preferred_element_type=compute_dtype,
        )
    else:
        output = jnp.einsum(
            subscripts, *(_dense(operand).astype(compute_dtype) for operand in arrays), optimize=optimize, precision=precision,
            preferred_element_type=compute_dtype,
        )
    return output.astype(result_dtype)


__all__ = ['linear', 'einsum']
