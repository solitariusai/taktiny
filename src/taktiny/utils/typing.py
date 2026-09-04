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
from os import PathLike as OSPathLike
from typing import Any, Protocol, runtime_checkable

import jax
import qwix
from jax.lax import ConvGeneralDilatedDimensionNumbers, PrecisionLike
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike as JaxArrayLike
from jax.typing import DTypeLike

type Array = jax.Array
type ArrayLike = JaxArrayLike
type Activation = str | Callable[[Array], Array]
type DType = DTypeLike
type PRNGKey = jax.Array
type PyTree = Any
type Shape = Sequence[int]
type GenericShape = int | Sequence[int]
type Axes = int | Sequence[int]
type AxisName = str | None
type AxisNames = tuple[AxisName, ...]
type MeshAxisName = str | tuple[str, ...] | None
type LogicalRules = Sequence[tuple[str, MeshAxisName]]
type Sharding = NamedSharding | PartitionSpec | None
type MeshLike = Mesh | None
type PathLike = str | OSPathLike[str]
type Batch = Mapping[str, PyTree]
type MutableBatch = dict[str, PyTree]
type StateDict = dict[str, PyTree]
type ParameterDict = dict[str, Any]
type ModuleFactory = Callable[..., Any]
type LossFn = Callable[[Any, Batch], Array]
type QuantConfig = str | qwix.QuantizationRule | qwix.PtqProvider | Sequence[qwix.QuantizationRule] | None
type MetaData = dict[str, Any] | Sequence[tuple[str, Any]]


@runtime_checkable
class StatefulIterator[T](Protocol):
    """Iterator whose cursor can be checkpointed and restored."""

    def __iter__(self) -> Iterator[T]: ...

    def __next__(self) -> T: ...

    def get_state(self) -> PyTree: ...

    def set_state(self, state: PyTree) -> None: ...

@runtime_checkable
class EpochAware(Protocol):
    """Data source that supports deterministic epoch selection."""
    def set_epoch(self, epoch: int) -> None: ...

class Initializer(Protocol):
    def __call__(
        self,
        key: Array,
        shape: Sequence[int | Any],
        dtype: DType | None = None,
        out_sharding: NamedSharding | PartitionSpec | None = None
    ) -> Array:
        ...

class DotGeneral(Protocol):
    def __call__(
        self,
        lhs: ArrayLike,
        rhs: ArrayLike,
        dimension_numbers: tuple[
            tuple[Sequence[int], Sequence[int]],
            tuple[Sequence[int], Sequence[int]],
        ],
        precision: PrecisionLike,
        preferred_element_type: DTypeLike | None,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        ...

class ConvGeneralDilated(Protocol):
    def __call__(
        self,
        lhs: Array,
        rhs: Array,
        window_strides: Sequence[int],
        padding: str | Sequence[tuple[int, int]],
        lhs_dilation: Sequence[int] | None = None,
        rhs_dilation: Sequence[int] | None = None,
        dimension_numbers: ConvGeneralDilatedDimensionNumbers | None = None,
        feature_group_count: int = 1,
        batch_group_count: int = 1,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        out_sharding: NamedSharding | PartitionSpec | None = None,
    ) -> Array:
        ...

__all__ = [
    'Activation',
    'Array',
    'ArrayLike',
    'Axes',
    'AxisName',
    'AxisNames',
    'Batch',
    'ConvGeneralDilated',
    'DType',
    'DotGeneral',
    'EpochAware',
    'GenericShape',
    'Initializer',
    'LogicalRules',
    'LossFn',
    'MeshAxisName',
    'MeshLike',
    'MetaData',
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
    'StatefulIterator'
]
