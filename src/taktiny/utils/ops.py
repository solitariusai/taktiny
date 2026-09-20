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

import re
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import qwix
from jax.lax import PrecisionLike
from jax.sharding import NamedSharding
from jax.typing import DTypeLike
from qwix._src.core import conv_general_qt, dot_general_qt

from taktiny.utils.quantization import normalize_qtype, quantization_rules
from taktiny.utils.typing import Array, ArrayLike, QuantConfig


def _rule(quant: QuantConfig, module_path: str, op: str) -> qwix.QuantizationRule | None:
    for rule in quantization_rules(quant):
        if rule.op_names and op not in rule.op_names and not (op == 'einsum' and 'dot_general' in rule.op_names):
            continue
        if not (re.fullmatch(rule.module_path, module_path) or re.fullmatch(rule.module_path, module_path.replace('.', '/'))):
            continue
        if rule.weight_qtype is None and rule.act_qtype is None and (
            not isinstance(rule, qwix.QtRule) or rule.bwd_qtype is None
        ):
            return None
        if rule.act_static_scale:
            raise NotImplementedError('Functional ops do not maintain static activation calibration state')
        if isinstance(rule, qwix.QtRule) and rule.bwd_stochastic_rounding is not None:
            raise NotImplementedError('Functional QT stochastic rounding requires an RNG interface')
        return replace(rule, weight_qtype=normalize_qtype(rule.weight_qtype), act_qtype=normalize_qtype(rule.act_qtype))
    return None


def _rule_dot(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, *,
              rule, lhs_weight=False, rhs_weight=True, out_sharding=None):
    if jnp.iscomplexobj(lhs) or jnp.iscomplexobj(rhs):
        raise TypeError('quantized operations require real-valued operands')
    if isinstance(rule, qwix.QtRule):
        if out_sharding is not None:
            raise NotImplementedError('Qwix QT kernels do not yet accept out_sharding')
        config = dot_general_qt.DotGeneralQtConfig(
            lhs_qtype=rule.weight_qtype if lhs_weight else rule.act_qtype,
            rhs_qtype=rule.weight_qtype if rhs_weight else rule.act_qtype,
            lhs_calibration_method=rule.weight_calibration_method if lhs_weight else rule.act_calibration_method or 'absmax',
            rhs_calibration_method=rule.weight_calibration_method if rhs_weight else rule.act_calibration_method or 'absmax',
            tile_size=rule.tile_size,
            lhs_disable_channelwise_axes=rule.disable_channelwise_axes,
            rhs_disable_channelwise_axes=rule.disable_channelwise_axes,
            dlhs_grad_qtype=normalize_qtype(rule.bwd_qtype),
            drhs_grad_qtype=normalize_qtype(rule.bwd_qtype),
            dlhs_grad_calibration_method=rule.bwd_calibration_method,
            drhs_grad_calibration_method=rule.bwd_calibration_method,
            dlhs_grad_disable_channelwise_axes=rule.disable_channelwise_axes,
            drhs_grad_disable_channelwise_axes=rule.disable_channelwise_axes,
            dlhs_tile_size=rule.bwd_weight_grad_tile_size if lhs_weight else None,
            drhs_tile_size=rule.bwd_weight_grad_tile_size if rhs_weight else None,
            use_original_residuals=rule.bwd_qtype is None,
        )
        if rule.additional_qt_config:
            config = replace(config, **rule.additional_qt_config)
        output = dot_general_qt.dot_general_qt(lhs, rhs, dimension_numbers, config)
        return output if preferred_element_type is None else output.astype(preferred_element_type)

    def prepare(value, contracting, weight):
        qtype = rule.weight_qtype if weight else rule.act_qtype
        if qtype is None:
            return value
        tiled = None
        if rule.tile_size is not None and contracting:
            size = rule.tile_size
            axis = contracting[-1]
            if not isinstance(size, int) or value.shape[axis] % size == 0:
                tiled = {axis: size}
        return qwix.quantize(
            value, qtype, channelwise_axes=tuple(i for i in range(value.ndim) if i not in contracting),
            tiled_axes=tiled,
            calibration_method=rule.weight_calibration_method if weight else rule.act_calibration_method or 'absmax',
        )

    left = prepare(lhs, dimension_numbers[0][0], lhs_weight)
    right = prepare(rhs, dimension_numbers[0][1], rhs_weight)
    return qwix.dot_general(left, right, dimension_numbers, precision=precision,
                            preferred_element_type=preferred_element_type, out_sharding=out_sharding)


def _as_operand(value: ArrayLike | qwix.QArray) -> Array | qwix.QArray:
    return value if isinstance(value, qwix.QArray) else jnp.asarray(value)


def _validate_conv_training_rule(rule: qwix.QtRule) -> None:
    if rule.tile_size is not None or rule.bwd_weight_grad_tile_size is not None:
        raise NotImplementedError('Convolution QT does not support tiled quantization')
    if rule.additional_qt_config:
        raise NotImplementedError('Convolution QT does not support additional_qt_config')


def _rule_conv(
    lhs, rhs, window_strides, padding, lhs_dilation=None, rhs_dilation=None,
    dimension_numbers=None, feature_group_count=1, batch_group_count=1,
    precision=None, preferred_element_type=None, out_sharding=None, *, rule,
):
    """Apply a training rule to a convolution on floating-point parameters."""
    _validate_conv_training_rule(rule)
    dims = jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    bwd_type = normalize_qtype(rule.bwd_qtype)
    config = conv_general_qt.ConvGeneralQtConfig(
        lhs_qtype=rule.act_qtype, rhs_qtype=rule.weight_qtype,
        lhs_calibration_method=rule.act_calibration_method or 'absmax',
        rhs_calibration_method=rule.weight_calibration_method,
        lhs_disable_channelwise_axes=rule.disable_channelwise_axes,
        rhs_disable_channelwise_axes=rule.disable_channelwise_axes,
        dlhs_grad_qtype=bwd_type, drhs_grad_qtype=bwd_type,
        dlhs_grad_calibration_method=rule.bwd_calibration_method,
        drhs_grad_calibration_method=rule.bwd_calibration_method,
        dlhs_grad_disable_channelwise_axes=rule.disable_channelwise_axes,
        drhs_grad_disable_channelwise_axes=rule.disable_channelwise_axes,
    )
    output = conv_general_qt.conv_general_qt(
        lhs, rhs, config, window_strides, padding, lhs_dilation, rhs_dilation,
        dims, feature_group_count, batch_group_count, out_sharding,
    )
    return output if preferred_element_type is None else output.astype(preferred_element_type)


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


def linear(
    x: ArrayLike | qwix.QArray,
    w: ArrayLike | qwix.QArray,
    b: ArrayLike | qwix.QArray | None = None,
    *,
    quant: QuantConfig = None,
    module_path: str = '',
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
    ``quant`` accepts a dtype string, a Qwix rule, a sequence of rules, or a
    PtqProvider/QtProvider. Strings use Taktiny's weight-only shorthand.
    Rules match ``module_path`` (empty by default) and ``dot_general`` in
    first-match order. ``weight_qtype`` applies to ``w`` and ``act_qtype`` to
    ``x``. Existing QArrays are dequantized before applying a selected rule.
    A rule with unset qtypes disables quantization for its matched operation.

    ``QtRule`` uses Qwix's custom-gradient training kernels; ``bwd_qtype``
    controls backward quantization. Ordinary rules and string shortcuts use
    PTQ, not trainable weight quantization. For training, pass floating-point
    parameters and a QtRule. Static activation scales and stochastic rounding
    are not supported by these stateless functions. QT currently does not
    support an explicit ``out_sharding`` argument.
    Training uses Qwix QT kernels; supported formats and backward operations
    depend on Qwix and the execution backend.

    Returns a dense JAX array. ``preferred_element_type`` selects its dtype,
    including after bias addition; otherwise the dtype is promoted from the
    inputs and bias, using QArray scale dtypes. Raw FP8 inputs promote to BF16;
    integer-only inputs with ``quant`` set promote to FP32. Accumulation may
    use a higher precision than the returned dtype. ``out_sharding`` is passed
    to the dot operation where supported.

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
    rule = _rule(quant, module_path, 'dot_general')
    bias = None if b is None else _dense(_as_operand(b))
    implicit_dtype = _dtype(x, w, *((bias,) if bias is not None else ()), quantized=rule is not None)
    result_dtype = implicit_dtype if preferred_element_type is None else jnp.dtype(preferred_element_type)
    compute_dtype = jnp.result_type(implicit_dtype, result_dtype)
    if rule is not None:
        output = _rule_dot(
            _dense(x).astype(compute_dtype), _dense(w).astype(compute_dtype), dimension_numbers, rule=rule,
            precision=precision, preferred_element_type=compute_dtype,
            out_sharding=out_sharding,
        )
    elif quant is not None:
        output = jax.lax.dot_general(
            _dense(x).astype(compute_dtype), _dense(w).astype(compute_dtype), dimension_numbers,
            precision=precision, preferred_element_type=compute_dtype, out_sharding=out_sharding,
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
    quant: QuantConfig = None,
    module_path: str = '',
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
    ``quant`` accepts the same rule configurations as :func:`linear`.
    Strings are weight-only shortcuts. In two-operand expressions the first
    operand is the activation and the second is the weight, independent of
    contraction ordering. Rules match ``module_path`` and either ``einsum``
    or ``dot_general``. QtRule enables training and its ``bwd_qtype`` controls
    backward quantization. Rule-based expressions support at most two operands;
    transformations that obscure operand roles raise an error. Unary operations
    remain ordinary JAX operations. Returns a dense JAX array.

    Static scales, stochastic rounding, and explicit output sharding for QT
    have the same limitations as :func:`linear`.

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
    rule = _rule(quant, module_path, 'einsum')
    implicit_dtype = _dtype(*arrays, quantized=rule is not None)
    result_dtype = implicit_dtype if preferred_element_type is None else jnp.dtype(preferred_element_type)
    compute_dtype = jnp.result_type(implicit_dtype, result_dtype)
    if quant is not None or out_sharding is not None:
        if rule is not None and len(arrays) > 2:
            raise NotImplementedError('Rule-based einsum supports at most two operands; split multi-operand contractions')
        inputs = tuple(_dense(operand).astype(compute_dtype) for operand in arrays)

        def dot(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, **kwargs):
            if rule is None:
                return jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=precision,
                                           preferred_element_type=preferred_element_type, **kwargs)
            lhs_weight = len(inputs) == 2 and lhs is inputs[1]
            rhs_weight = len(inputs) == 2 and rhs is inputs[1]
            if len(inputs) == 2 and not (lhs_weight or rhs_weight) and rule.weight_qtype != rule.act_qtype:
                raise NotImplementedError('Cannot identify the weight after einsum operand transformations')
            return _rule_dot(lhs, rhs, dimension_numbers, precision, preferred_element_type,
                             rule=rule, lhs_weight=lhs_weight, rhs_weight=rhs_weight, **kwargs)

        with jax.disable_jit():
            output = jnp.einsum(
                subscripts, *inputs, optimize=optimize, precision=precision,
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
