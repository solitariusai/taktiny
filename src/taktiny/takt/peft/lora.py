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
"""LoRA model transformation."""

from __future__ import annotations
from typing import Any


import re

import jax
import jax.numpy as jnp
import numpy as np
import qwix

from taktiny.nn.lora import LoRALinear
from taktiny.nn.module import Module, iter_children
from taktiny.nn.rng import Rngs
from taktiny.takt._prelude import Takt, _replace_child
from taktiny.takt.peft.config import LoraConfig
from taktiny.utils.quantization import (
    quantize_linear_weight,
    resolve_quantization_rule,
)


def _lora_modules(model: Any) -> Any:
    modules = {}

    def collect(module: Any, prefix: str='') -> None:
        for name, child in iter_children(module):
            full_name = f'{prefix}.{name}' if prefix else name
            if isinstance(child, LoRALinear):
                modules[full_name] = child
            elif isinstance(child, Module):
                collect(child, full_name)

    collect(model)
    return modules


def _lora_parameters(model: Any) -> Any:
    parameters = {}
    for name, module in _lora_modules(model).items():
        parameters[f'{name}.lora_A'] = module.lora_A
        parameters[f'{name}.lora_B'] = module.lora_B
    return parameters


@Takt.register_peft(LoraConfig)
def _apply_lora(model: Module, config: LoraConfig) -> Any:
    if not isinstance(model, Module):
        raise TypeError('LoRA currently requires a Taktiny nn.Module model')

    rngs = config.rngs or Rngs(0)
    matched = []
    adapters = []
    trainable_state = {
        name: parameter.trainable
        for name, parameter in model.flat_parameter_dict().items()
    }

    def transform(module: Any, prefix: str='') -> None:
        for name, child in list(iter_children(module)):
            full_name = f'{prefix}.{name}' if prefix else name
            is_target = any(
                re.search(pattern, full_name)
                for pattern in config.target_modules
            )

            if is_target:
                if isinstance(child, LoRALinear):
                    raise ValueError(
                        f'LoRA is already applied to {full_name}'
                    )
                if not (
                    hasattr(child, 'in_features')
                    and hasattr(child, 'out_features')
                ):
                    raise TypeError(
                        f'PEFT target {full_name} is not a linear module'
                    )

                replacement = LoRALinear(
                    base_layer=child,
                    rank=config.rank,
                    alpha=config.alpha,
                    rngs=rngs,
                )
                _replace_child(module, name, replacement)
                matched.append(full_name)
                adapters.extend(
                    (replacement.lora_A, replacement.lora_B)
                )
            elif isinstance(child, Module):
                transform(child, full_name)

    transform(model)
    if not matched:
        patterns = ', '.join(config.target_modules)
        raise ValueError(
            f'No modules matched the PEFT target patterns: {patterns}'
        )

    for parameter in model.flat_parameter_dict().values():
        parameter.trainable = False
    for parameter in adapters:
        parameter.trainable = True

    peft_config = {
        'peft_type': 'LORA',
        'target_modules': list(config.target_modules),
        'rank': int(config.rank),
        'alpha': float(config.alpha),
    }
    base_model = getattr(model, 'base_model_name_or_path', None)
    if base_model is not None:
        peft_config['base_model_name_or_path'] = str(base_model)
    model.peft_config = peft_config
    model._peft_trainable_state = trainable_state

    return model


@Takt.register_peft_loader('LORA')
def _load_lora(model: Any, config: Any, state: Any, *, rngs: Any) -> Any:
    rank = config.get('rank')
    alpha = config.get('alpha')
    target_modules = config.get('target_modules')
    if rank is None or alpha is None or target_modules is None:
        raise ValueError(
            'LoRA adapter_config.json must contain rank, alpha, and '
            'target_modules'
        )

    lora_config = LoraConfig(
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        rngs=rngs,
    )
    existing_modules = _lora_modules(model)
    if not existing_modules:
        model = Takt.apply_peft(
            model,
            lora_config,
        )
    else:
        for name, module in existing_modules.items():
            module_alpha = module.scaling * module.rank
            if (
                module.rank != lora_config.rank
                or not np.isclose(module_alpha, lora_config.alpha)
            ):
                raise ValueError(
                    f'Existing LoRA module {name!r} uses rank '
                    f'{module.rank} and alpha {module_alpha}, but the '
                    f'adapter requires rank {lora_config.rank} and alpha '
                    f'{lora_config.alpha}'
                )
    parameters = _lora_parameters(model)

    loaded = {}
    stacked_layers = {}
    unexpected = []

    for name, value in state.items():
        if name in parameters:
            parameter = parameters[name]
            if value.shape != parameter.shape:
                raise ValueError(
                    f'Adapter tensor {name!r} has shape {value.shape}, '
                    f'expected {parameter.shape}'
                )
            loaded[name] = value
            continue

        matched_stack = False
        parts = name.split('.')
        for position, part in enumerate(parts):
            if not part.isdigit():
                continue
            stacked_parts = list(parts)
            stacked_parts[position] = 'stacked'
            stacked_name = '.'.join(stacked_parts)
            if stacked_name not in parameters:
                continue

            parameter = parameters[stacked_name]
            layer_index = int(part)
            if layer_index >= parameter.shape[0]:
                raise ValueError(
                    f'Adapter layer index {layer_index} is out of range for '
                    f'{stacked_name!r}'
                )
            expected_shape = parameter.shape[1:]
            if value.shape != expected_shape:
                raise ValueError(
                    f'Adapter tensor {name!r} has shape {value.shape}, '
                    f'expected {expected_shape}'
                )
            entry = stacked_layers.setdefault(
                stacked_name,
                {
                    'values': np.empty(
                        parameter.shape,
                        dtype=value.dtype,
                    ),
                    'indices': set(),
                },
            )
            if layer_index in entry['indices']:
                raise ValueError(
                    f'Duplicate adapter layer tensor: {name}'
                )
            entry['values'][layer_index] = value
            entry['indices'].add(layer_index)
            matched_stack = True
            break

        if not matched_stack:
            unexpected.append(name)

    if unexpected:
        preview = ', '.join(sorted(unexpected)[:8])
        raise ValueError(
            f'Adapter checkpoint contains unexpected tensors: {preview}'
        )

    for name, entry in stacked_layers.items():
        parameter = parameters[name]
        expected_indices = set(range(parameter.shape[0]))
        missing_indices = expected_indices - entry['indices']
        if missing_indices:
            missing = ', '.join(map(str, sorted(missing_indices)))
            raise ValueError(
                f'Adapter checkpoint is missing layers {missing} for {name!r}'
            )
        loaded[name] = entry['values']

    missing = sorted(set(parameters) - set(loaded))
    if missing:
        preview = ', '.join(missing[:8])
        raise ValueError(
            f'Adapter checkpoint is missing tensors: {preview}'
        )

    for name, value in loaded.items():
        parameter = parameters[name]
        array = jnp.asarray(value, dtype=parameter.dtype)
        sharding = getattr(parameter.value, 'sharding', None)
        if sharding is not None:
            array = jax.device_put(array, sharding)
        parameter.value = array

    model.peft_config = dict(config)
    return model


@Takt.register_peft_merger(LoRALinear)
def _merge_lora(module: Any, *, dtype: Any, quant: Any, module_path: str) -> Any:
    base_layer = module.base_layer
    weight = getattr(base_layer, 'weight', None)
    if weight is None:
        raise TypeError(
            f'LoRA base module {module_path} has no mergeable weight'
        )

    base_value = weight.value
    if isinstance(base_value, qwix.QArray):
        base_value = qwix.dequantize(base_value)

    if dtype is None:
        target_dtype = base_value.dtype
    else:
        target_dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(target_dtype, jnp.floating):
        raise TypeError(
            'Merged LoRA dtype must be floating-point; use quant= for '
            'quantized output'
        )

    delta = jnp.matmul(
        module.lora_A.value.astype(jnp.float32),
        module.lora_B.value.astype(jnp.float32),
    ).reshape(base_value.shape)
    merged = (
        base_value.astype(jnp.float32)
        + delta * module.scaling
    ).astype(target_dtype)

    weight.value = merged
    weight.quantization = None
    if quant is not None:
        rule = resolve_quantization_rule(
            quant,
            module_path,
        )
        if rule is not None:
            weight.value = quantize_linear_weight(
                merged,
                weight,
                rule,
            )
            weight.quantization = rule

    return base_layer


__all__ = []
