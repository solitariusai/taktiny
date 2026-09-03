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
"""Shared type aliases and protocols used across TakTiny."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import enum
from os import PathLike as OSPathLike
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike as JaxArrayLike, DTypeLike
import qwix


Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = JaxArrayLike
Activation: TypeAlias = str | Callable[[Array], Array]
DType: TypeAlias = DTypeLike
Initializer: TypeAlias = Callable[..., Array]
PRNGKey: TypeAlias = jax.Array
PyTree: TypeAlias = Any
Shape: TypeAlias = Sequence[int]
Axes: TypeAlias = int | Sequence[int]
AxisName: TypeAlias = str | None
AxisNames: TypeAlias = tuple[AxisName, ...]
MeshAxisName: TypeAlias = str | tuple[str, ...] | None
LogicalRules: TypeAlias = Sequence[tuple[str, MeshAxisName]]
Sharding: TypeAlias = NamedSharding | PartitionSpec | None
MeshLike: TypeAlias = Mesh | None
PathLike: TypeAlias = str | OSPathLike[str]
Batch: TypeAlias = Mapping[str, PyTree]
MutableBatch: TypeAlias = dict[str, PyTree]
StateDict: TypeAlias = dict[str, PyTree]
ParameterDict: TypeAlias = dict[str, Any]
ModuleFactory: TypeAlias = Callable[..., Any]
LossFn: TypeAlias = Callable[[Any, Batch], Array]
QuantConfig: TypeAlias = str | qwix.QuantizationRule | qwix.PtqProvider | Sequence[qwix.QuantizationRule] | None
T = TypeVar('T')

@runtime_checkable
class StatefulIterator(Protocol[T]):
    """Iterator whose cursor can be checkpointed and restored."""

    def __iter__(self) -> Iterator[T]: ...

    def __next__(self) -> T: ...

    def get_state(self) -> PyTree: ...

    def set_state(self, state: PyTree) -> None: ...

@runtime_checkable
class EpochAware(Protocol):
    """Data source that supports deterministic epoch selection."""
    def set_epoch(self, epoch: int) -> None: ...


__all__ = [
    'Array',
    'ArrayLike',
    'Activation',
    'Axes',
    'AxisName',
    'AxisNames',
    'Batch',
    'DType',
    'EpochAware',
    'Initializer',
    'LogicalRules',
    'LossFn',
    'MeshAxisName',
    'MeshLike',
    'ModuleFactory',
    'MutableBatch',
    'PRNGKey',
    'ParameterDict',
    'PathLike',
    'PyTree',
    'QuantConfig',
    'Shape',
    'Sharding',
    'StateDict',
    'StatefulIterator',
]
