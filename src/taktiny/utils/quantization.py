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
"""Quantization helpers."""

from __future__ import annotations
from typing import Any


from collections.abc import Sequence
import dataclasses
import re

import jax.numpy as jnp
import qwix


_QTYPE_ALIASES = {
    'fp8': jnp.float8_e4m3fn,
    'float8': jnp.float8_e4m3fn,
}


def normalize_qtype(qtype: Any) -> Any:
    if isinstance(qtype, str):
        qtype = qtype.lower()
        if qtype == 'fp4':
            raise ValueError(
                'Qwix does not provide the legacy Taktiny fp4 format; '
                'use int4, nf4, or fp8'
            )
        return _QTYPE_ALIASES.get(qtype, qtype)
    return qtype


def quantization_rules(quantization: Any) -> tuple[qwix.QuantizationRule, ...]:
    if quantization is None:
        return ()
    if isinstance(quantization, str):
        qtype = normalize_qtype(quantization)
        return (
            qwix.QuantizationRule(
                # String shortcuts are intended for weight-only matrix
                # multiplications. Embedding tables, including tied output
                # heads, are too sensitive for an unconditional 4-bit
                # fallback and remain dense unless explicitly selected by a
                # user-provided rule.
                op_names=('dot_general',),
                weight_qtype=qtype,
                tile_size=(
                    128
                    if qtype in {'int4', 'nf4'}
                    else None
                ),
            ),
        )
    if isinstance(quantization, qwix.QuantizationRule):
        return (quantization,)
    if isinstance(quantization, qwix.PtqProvider):
        return tuple(quantization._rules)  # Qwix exposes no public rule accessor.
    if isinstance(quantization, Sequence):
        rules = tuple(quantization)
        if all(isinstance(rule, qwix.QuantizationRule) for rule in rules):
            return rules
    raise TypeError(
        'quant must be a Qwix QuantizationRule, PtqProvider, '
        'a sequence of rules, or a qtype string'
    )


def merge_quantization(quantization: Any, fallback_qtype: Any) -> Any:
    """Append a uniform fallback after any explicit quantization rules."""
    return (
        quantization_rules(quantization)
        + quantization_rules(fallback_qtype)
    )


def resolve_quantization_rule(
    quantization: Any,
    module_path: str,
    *,
    op_name: str='dot_general',
) -> Any:
    dotted_path = module_path
    slash_path = module_path.replace('.', '/')
    for rule in quantization_rules(quantization):
        if rule.op_names and op_name not in rule.op_names:
            continue
        if not (
            re.fullmatch(rule.module_path, dotted_path)
            or re.fullmatch(rule.module_path, slash_path)
        ):
            continue
        if rule.weight_qtype is None:
            return None
        if rule.act_qtype is not None:
            raise NotImplementedError(
                'Taktiny Qwix PTQ currently supports weight-only rules; '
                'activation quantization requires an operation interceptor'
            )
        return dataclasses.replace(
            rule,
            weight_qtype=normalize_qtype(rule.weight_qtype),
        )
    return None


def quantize_linear_weight(array: Any, parameter: Any, rule: Any) -> Any:
    batch_axis_count = getattr(
        parameter,
        'quantization_batch_axis_count',
        0,
    )
    input_axis_count = getattr(parameter, 'input_axis_count', None)
    if input_axis_count is None:
        raise ValueError(
            'Qwix quantization metadata is missing from Linear weight'
        )

    output_start = batch_axis_count + input_axis_count
    channelwise_axes = tuple(range(batch_axis_count)) + tuple(
        range(output_start, array.ndim)
    )
    tiled_axes = None
    if rule.tile_size is not None:
        tiled_axis = output_start - 1
        tile_size = rule.tile_size
        if (
            not isinstance(tile_size, int)
            or array.shape[tiled_axis] % tile_size == 0
        ):
            tiled_axes = {tiled_axis: tile_size}

    return qwix.quantize(
        jnp.asarray(array),
        rule.weight_qtype,
        channelwise_axes=channelwise_axes,
        tiled_axes=tiled_axes,
        calibration_method=rule.weight_calibration_method,
        scale_dtype=parameter.dtype,
    )


def quantize_embedding_weight(array: Any, parameter: Any, rule: Any) -> Any:
    tiled_axes = None
    if rule.tile_size is not None:
        tiled_axes = {1: rule.tile_size}

    return qwix.quantize(
        jnp.asarray(array),
        rule.weight_qtype,
        channelwise_axes=(0,),
        tiled_axes=tiled_axes,
        calibration_method=rule.weight_calibration_method,
        scale_dtype=parameter.dtype,
    )


__all__ = [
    'normalize_qtype',
    'quantization_rules',
    'merge_quantization',
    'resolve_quantization_rule',
    'quantize_linear_weight',
    'quantize_embedding_weight',
]
