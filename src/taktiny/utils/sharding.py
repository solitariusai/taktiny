# Copyright 2026 Shinapri
# Copyright 2025-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long, disable=bare-except, consider-using-generator
"""Utilities for sharding and JIT partitioning in Taktiny."""
from __future__ import annotations

from typing import Any

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from taktiny.utils import spmd


def remove_size_one_mesh_axis(spec: Any, mesh: Any) -> Any:
    """
    Removes mesh axes from a PartitionSpec (P) where the axis size is 1.

    This is a common optimization to simplify sharding by excluding redundant axes.
    Function originally from jax._src.core:
    https://github.com/jax-ml/jax/blob/main/jax/_src/core.py
    """
    if spec is None:
        return None
    new_spec = []
    for s in spec:
        if s is None or s == P.UNCONSTRAINED:
            new_spec.append(s)
        elif isinstance(s, tuple):
            new_spec.append(tuple(i for i in s if mesh.shape.get(i, 1) != 1))
        else:
            new_spec.append(None if mesh.shape.get(s, 1) == 1 else s)
    return P(*new_spec, unreduced=spec.unreduced, reduced=spec.reduced)

def logical_to_mesh_axes(logical_names: Any, mesh: Any, rules: Any=None) -> Any:
    """Remove size one mesh axes given logical names."""
    tensor_spec = spmd.logical_to_mesh_axes(logical_names, rules=rules)
    return remove_size_one_mesh_axis(tensor_spec, mesh)


def logical_to_mesh(tree: Any, mesh: Any, rules: Any=None) -> Any:
  """Remove size one mesh axes given logical pspec pytree."""
  if tree is None:
      return None
  return jax.tree.map(
      lambda x: logical_to_mesh_axes(x, mesh, rules=rules),
      tree,
      is_leaf=lambda x: isinstance(x, P),
  )


def logical_to_mesh_sharding(tree: Any, mesh: Any, rules: Any=None) -> Any:
    """Return sharding pytree given logical specs pytree"""
    if tree is None:
        return None
    return jax.tree.map(
        lambda x: NamedSharding(mesh, x),
        logical_to_mesh(tree, mesh, rules=rules),
        is_leaf=lambda x: isinstance(x, P),
    )


def create_sharding(mesh: Any, logical_names: Any, rules: Any=None) -> Any:
    """Create NamedSharding with given logical names."""
    return NamedSharding(mesh, logical_to_mesh_axes(logical_names, mesh, rules=rules))


__all__ = [
  'remove_size_one_mesh_axis',
  'logical_to_mesh',
  'logical_to_mesh_axes',
  'logical_to_mesh_sharding',
  'create_sharding',
]