# Copyright 2026 Shinapri.
# Copyright 2024 The Flax Authors.
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

"""Utilities for working with jit and partitioned models."""
from __future__ import annotations

from typing import Any, Self, Sequence
from collections.abc import Mapping
import jax
import collections
import threading
import dataclasses
import contextlib
import contextvars

from taktiny.utils.typing import LogicalRules, MeshAxisName

class _UnassignedAxis:
    def __repr__(self) -> str: return 'UnassignedAxis'
    def __bool__(self) -> bool: return False

_unassigned_axis = _UnassignedAxis()

@dataclasses.dataclass
class _AxisRules(threading.local):
    rules: LogicalRules = ()

_axis_rules = _AxisRules()

def get_logical_axis_rules() -> LogicalRules:
    return _axis_rules.rules

def set_logical_axis_rules(rules: LogicalRules) -> None:
    _axis_rules.rules = tuple(rules)

class map_logical_axis_names(contextlib.ContextDecorator):
    """Context manager and decorator for mapping logical axis names to mesh axis names.
    
    Can also be called as a function without a 'with' block to set rules globally.
    """
    def __init__(
        self, 
        map_names: Mapping[str, MeshAxisName] | Sequence[tuple[str, MeshAxisName]] | None
    ) -> None:
        if map_names is not None and not isinstance(map_names, (Mapping, Sequence)):
            raise TypeError('map_names must be a mapping (dict) or a sequence of tuples.')
        
        rules: list[tuple[str, MeshAxisName]] = []
        if isinstance(map_names, Mapping):
            rules = list(map_names.items())
        elif map_names is not None:
            rules = list(map_names)
            
        self._rules = tuple(rules)
        
        # Apply globally immediately upon instantiation
        current_rules = get_logical_axis_rules()
        self._prev_rules = current_rules
        new_rules = self._rules + current_rules
        set_logical_axis_rules(new_rules)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Revert ONLY if used in a with block
        set_logical_axis_rules(self._prev_rules)

def _mesh_assignment_free(new_assignment: Any, existing_assignments: Any) -> bool:
    new = set(jax.tree_util.tree_leaves(new_assignment))
    existing = set(jax.tree_util.tree_leaves(existing_assignments))
    new.discard(jax.sharding.PartitionSpec.UNCONSTRAINED)
    new.discard(None)
    if existing.intersection(new):
        return False
    return True

def _logical_to_mesh_axes(
    array_dim_names: Sequence[str | None] | None,
    rules: LogicalRules | None = None,
) -> list[_UnassignedAxis | None | str | tuple[str, ...]] | None:
    if array_dim_names is None:
        return None
    if rules is None:
        rules = get_logical_axis_rules()
    axis_name_counts = collections.Counter(array_dim_names)
    dups = tuple(k for k, v in axis_name_counts.items() if v > 1 and isinstance(k, str))
    if dups:
        raise ValueError(f'Unsupported: Dimensions {dups} occur more than once.')
    if not isinstance(rules, (tuple, list)):
        raise TypeError('Unknown axis rule specification type.')
        
    result: list[_UnassignedAxis | None | str | tuple[str, ...]]
    result = [(_unassigned_axis if isinstance(name, str) else name) for name in array_dim_names]
    
    for rule_model_name, rule_mesh_names in rules:
        if rule_model_name in array_dim_names:
            pos = array_dim_names.index(rule_model_name)
            if _mesh_assignment_free(rule_mesh_names, result) and result[pos] == _unassigned_axis:
                result[pos] = rule_mesh_names
    return result

def logical_to_mesh_axes(
    array_dim_names: Sequence[str | None] | None,
    rules: LogicalRules | None = None,
) -> jax.sharding.PartitionSpec | None:
    result = _logical_to_mesh_axes(array_dim_names, rules)
    if result is None:
        return None
    result = [None if x is _unassigned_axis else x for x in result]
    return jax.sharding.PartitionSpec(*result)

def with_logical_partitioning(
    initializer: Any, 
    axis_names: Sequence[str | None] | None = None,
    partition_spec: Any = None
) -> Any:
    """Wraps an initializer to instantly shard its output across the active JAX mesh
    using logical axis rules, without allocating on a single device first.
    
    Args:
        initializer: A standard JAX initializer function (key, shape, dtype).
        axis_names: The logical axis names for the parameter.
        partition_spec: Optional explicit partition spec to use if axis_names is omitted or unmapped.
        
    Returns:
        A wrapped initializer function that automatically applies JIT and out_shardings
        if a global JAX mesh is active.
    """
    def wrapper(key, shape, dtype=None):
        import jax
        import jax.numpy as jnp
        if dtype is None:
            dtype = jnp.float32
            
        resolved_spec = partition_spec
        if axis_names is not None:
            mapped_axes = logical_to_mesh_axes(axis_names)
            if mapped_axes is not None:
                resolved_spec = mapped_axes
                
        if resolved_spec is not None:
            try:
                # JIT the initializer with the partition spec to instantly distribute it
                sharded_fn = jax.jit(initializer, static_argnums=(1, 2), out_shardings=resolved_spec)
                return sharded_fn(key, shape, dtype)
            except RuntimeError as e:
                # Fallback if no global jax.set_mesh() context is active
                if "requires a non-empty mesh in context" in str(e):
                    return initializer(key, shape, dtype)
                raise
        return initializer(key, shape, dtype)
    return wrapper
