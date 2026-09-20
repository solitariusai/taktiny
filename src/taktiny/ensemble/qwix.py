# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""experiment"""

import copy
import dataclasses
import functools
import re
from collections.abc import Collection
from contextvars import ContextVar
from typing import Any

import jax
import jax.numpy as jnp
import qwix
from qwix import QuantizationProvider

# Qwix currently exposes its QT kernels and interception only internally.
from qwix._src import interception
from qwix._src.core import conv_general_qt, dot_general_qt

from taktiny.nn.base import Module, Parameter, iter_children

_current_module: ContextVar[tuple[str, Module] | None] = ContextVar(
    'taktiny_qwix_module', default=None,
)


def _clone(model: Module) -> Module:
    # Parameter forwards unknown attributes to its array, including deepcopy.
    # Seed the memo with module shells to retain wrappers and shared parameters.
    memo = {}
    modules = []

    def collect(module):
        if id(module) in memo:
            return
        memo[id(module)] = object.__new__(type(module))
        modules.append(module)
        for _, child in iter_children(module):
            collect(child)

    collect(model)
    for module in modules:
        memo[id(module)].__dict__.update(copy.deepcopy(module.__dict__, memo))
    return memo[id(model)]


class _TrainingAdapter:
    """Translate Qwix rules without depending on Flax's module discovery."""

    def __init__(self, provider: qwix.QtProvider) -> None:
        self.rules = tuple(provider._rules)
        for rule in self.rules:
            if rule.act_static_scale:
                raise NotImplementedError('Static activation scales are not supported yet.')
            if rule.bwd_stochastic_rounding is not None:
                raise NotImplementedError('Stochastic rounding is not supported yet.')
            unsupported = set(rule.op_names) - {'dot_general', 'einsum', 'conv_general_dilated'}
            if unsupported:
                raise NotImplementedError(f'Unsupported quantized operations: {unsupported}')

    def rule(self, op: str) -> qwix.QtRule | None:
        context = _current_module.get()
        if context is None:
            return None
        path, _ = context
        for rule in self.rules:
            if rule.op_names and op not in rule.op_names:
                continue
            if re.fullmatch(rule.module_path, path) or re.fullmatch(
                rule.module_path, path.replace('.', '/'),
            ):
                return rule
        return None

    def is_weight(self, value: Any) -> bool:
        context = _current_module.get()
        return context is not None and any(
            parameter.value is value
            for parameter in context[1].flat_parameter_dict().values()
        )

    def dot(self, lhs, rhs, dimension_numbers, precision=None,
            preferred_element_type=None, *, out_sharding=None, _rule=None):
        rule = _rule if _rule is not None else self.rule('dot_general')
        if rule is None or rule.weight_qtype is None:
            return jax.lax.dot_general(
                lhs, rhs, dimension_numbers, precision=precision,
                preferred_element_type=preferred_element_type, out_sharding=out_sharding,
            )
        lhs_weight, rhs_weight = self.is_weight(lhs), self.is_weight(rhs)
        config = dot_general_qt.DotGeneralQtConfig(
            lhs_qtype=rule.weight_qtype if lhs_weight else rule.act_qtype,
            rhs_qtype=rule.weight_qtype if rhs_weight else rule.act_qtype,
            lhs_calibration_method=rule.weight_calibration_method if lhs_weight else rule.act_calibration_method,
            rhs_calibration_method=rule.weight_calibration_method if rhs_weight else rule.act_calibration_method,
            tile_size=rule.tile_size,
            lhs_disable_channelwise_axes=rule.disable_channelwise_axes,
            rhs_disable_channelwise_axes=rule.disable_channelwise_axes,
            dlhs_grad_qtype=rule.bwd_qtype,
            drhs_grad_qtype=rule.bwd_qtype,
            dlhs_grad_calibration_method=rule.bwd_calibration_method,
            drhs_grad_calibration_method=rule.bwd_calibration_method,
            dlhs_grad_disable_channelwise_axes=rule.disable_channelwise_axes,
            drhs_grad_disable_channelwise_axes=rule.disable_channelwise_axes,
            dlhs_tile_size=rule.bwd_weight_grad_tile_size if lhs_weight else None,
            drhs_tile_size=rule.bwd_weight_grad_tile_size if rhs_weight else None,
            use_original_residuals=rule.bwd_qtype is None,
        )
        if rule.additional_qt_config:
            config = dataclasses.replace(config, **rule.additional_qt_config)
        result = dot_general_qt.dot_general_qt(lhs, rhs, dimension_numbers, config)
        if preferred_element_type is not None:
            result = result.astype(preferred_element_type)
        if out_sharding is not None:
            result = jax.lax.with_sharding_constraint(result, out_sharding)
        return result

    def einsum(self, equation, *operands, **kwargs):
        rule = self.rule('einsum')
        if rule is None or rule.weight_qtype is None:
            return jnp.einsum(equation, *operands, **kwargs)
        if len(operands) != 2:
            raise NotImplementedError('Quantized einsum requires two operands.')
        kwargs['_dot_general'] = functools.partial(self.dot, _rule=rule)
        # Einsum internally jits its contraction; interception must see its trace.
        with jax.disable_jit():
            return jnp.einsum(equation, *operands, **kwargs)

    def conv(self, lhs, rhs, window_strides, padding, lhs_dilation=None,
             rhs_dilation=None, dimension_numbers=None, feature_group_count=1,
             batch_group_count=1, precision=None, preferred_element_type=None,
             out_sharding=None):
        rule = self.rule('conv_general_dilated')
        if rule is None or rule.weight_qtype is None:
            return jax.lax.conv_general_dilated(
                lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
                dimension_numbers, feature_group_count, batch_group_count,
                precision, preferred_element_type, out_sharding=out_sharding,
            )
        if rule.tile_size or rule.bwd_weight_grad_tile_size or rule.additional_qt_config:
            raise NotImplementedError('Convolution QT does not support tiling or additional_qt_config.')
        config = conv_general_qt.ConvGeneralQtConfig(
            lhs_qtype=rule.act_qtype, rhs_qtype=rule.weight_qtype,
            lhs_calibration_method=rule.act_calibration_method,
            rhs_calibration_method=rule.weight_calibration_method,
            lhs_disable_channelwise_axes=rule.disable_channelwise_axes,
            rhs_disable_channelwise_axes=rule.disable_channelwise_axes,
            dlhs_grad_qtype=rule.bwd_qtype, drhs_grad_qtype=rule.bwd_qtype,
            dlhs_grad_calibration_method=rule.bwd_calibration_method,
            drhs_grad_calibration_method=rule.bwd_calibration_method,
            dlhs_grad_disable_channelwise_axes=rule.disable_channelwise_axes,
            drhs_grad_disable_channelwise_axes=rule.disable_channelwise_axes,
        )
        dims = jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
        if isinstance(padding, str):
            spatial = dims.lhs_spec[2:]
            kernel_spatial = dims.rhs_spec[2:]
            dilations = rhs_dilation or (1,) * len(spatial)
            kernel_shape = tuple((rhs.shape[d] - 1) * r + 1 for d, r in zip(kernel_spatial, dilations))
            padding = jax.lax.padtype_to_pads(
                tuple(lhs.shape[d] for d in spatial), kernel_shape, window_strides, padding,
            )
        result = conv_general_qt.conv_general_qt(
            lhs, rhs, config, window_strides, padding, lhs_dilation,
            rhs_dilation, dims, feature_group_count, batch_group_count, out_sharding,
        )
        return result if preferred_element_type is None else result.astype(preferred_element_type)

    def interceptors(self):
        return interception.Interceptor(mapping={
            'jax.lax.dot_general': self.dot,
            'jax.numpy.einsum': self.einsum,
            'jax.lax.conv_general_dilated': self.conv,
        }, id=id(self))


def _scoped_method(method, path):
    @functools.wraps(method)
    def scoped(self, *args, **kwargs):
        token = _current_module.set((path, self))
        try:
            return method(self, *args, **kwargs)
        finally:
            _current_module.reset(token)
    return scoped


def quantize_model(
    model: Module,
    provider: QuantizationProvider,
    *model_inputs: Any,
    methods: Collection[str] = ("__call__",),
    **model_inputs_kwargs: Any,
) -> Module:
    """Return an independent model using Qwix quantized training operations.

    Supports the standard :class:`qwix.QtProvider` with dynamic activation
    scales for ``lax.dot_general``, two-operand ``jnp.einsum``, and
    ``lax.conv_general_dilated``. Parameters remain floating point; Qwix
    quantizes operation operands and, when ``bwd_qtype`` is set, backward
    matrix products. This is not conversion to permanently integer weights.

    Rules use first-match precedence and dotted or slash-separated module
    paths (the root path is an empty string). Selected entry methods are
    intercepted, including operations in their child modules. Calls through
    other root methods are unchanged. Existing custom operation hooks must
    ultimately call one of the supported JAX operations to be intercepted.
    Shared child modules use their first traversal path for rule matching.
    Weight detection for dot products uses parameter-array identity; if a
    custom module transforms a weight before the dot, it is treated as an
    activation. Convolutions always treat the right operand as a weight.

    Each selected method is run once on a disposable copy with the supplied
    inputs to validate execution without changing the returned model's state.
    The same inputs must be valid for every selected method. Python branches
    not exercised by these inputs are only checked when subsequently called.

    Args:
        model: A Taktiny module with floating-point parameters.
        provider: A standard Qwix QtProvider. Other providers, static scales,
            and stochastic rounding are not supported yet and raise errors.
        *model_inputs: Positional example inputs for the selected methods.
        methods: Root entry methods to intercept. Defaults to ``('__call__',)``.
        **model_inputs_kwargs: Keyword example inputs, forwarded unchanged.

    Returns:
        A quantized model instance.

    Example:
        >>> import jax.numpy as jnp
        >>> import qwix
        >>> from taktiny import nn
        >>> model = nn.Linear(8, 4, rngs=nn.Rngs(0))
        >>> provider = qwix.QtProvider([qwix.QtRule(
        ...     weight_qtype='int8', act_qtype='int8', bwd_qtype='int8')])
        >>> model = quantize_model(model, provider, jnp.ones((2, 8)))
        >>> model(jnp.ones((2, 8))).shape
        (2, 4)

    Note:
        Uses private Qwix kernel/interception APIs; Qwix upgrades should be
        checked against the integration tests. Precision hints are handled by
        the QT kernels, not by ordinary floating-point dot precision settings.
    """
    if not isinstance(model, Module):
        raise TypeError('model must be a Taktiny Module.')
    if type(provider) is not qwix.QtProvider:
        raise NotImplementedError('Only the standard qwix.QtProvider is supported currently.')
    if isinstance(methods, str) or not methods:
        raise ValueError('methods must be a non-empty collection of method names.')
    methods = tuple(dict.fromkeys(methods))
    for name in methods:
        if not isinstance(name, str) or not callable(getattr(model, name, None)):
            raise ValueError(f'Unknown model method: {name!r}')
    adapter = _TrainingAdapter(provider)
    result = _clone(model)
    seen = set()

    def prepare(module, path):
        if id(module) in seen:
            return
        seen.add(id(module))
        if hasattr(type(module), '_taktiny_qwix_original'):
            raise ValueError('Model is already wrapped by quantize_model; use the original model.')
        for name, child in iter_children(module):
            if isinstance(child, Parameter):
                if isinstance(child.value, qwix.QArray):
                    raise ValueError('QT requires floating-point parameters, not prequantized QArrays.')
            elif isinstance(child, Module):
                prepare(child, f'{path}.{name}' if path else name)
        original = type(module)
        fields = {'_taktiny_qwix_original': original}
        names = methods if module is result else tuple(set(methods) | {'__call__'})
        for name in names:
            method = getattr(original, name, None)
            if not callable(method):
                continue
            method = _scoped_method(method, path)
            if module is result:
                method = interception.wrap_func_intercepted(
                    method, adapter.interceptors, disable_jit=provider.disable_jit,
                )
            fields[name] = method
        module.__class__ = type(original.__name__, (original,), fields)

    prepare(result, '')
    for name in methods:
        getattr(_clone(result), name)(*model_inputs, **model_inputs_kwargs)
    return result
