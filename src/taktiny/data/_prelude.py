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
"""Utilities for adapting random-access datasets to Grain."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from absl import flags
from absl.flags import UnparsedFlagAccessError
import grain.python as grain
from grain._src.core.transforms import FlatMap as _FlatMapTransform


class Map(grain.MapTransform):
    """Adapt a regular callable to a Grain map transformation."""

    def __init__(self, function: Callable[[Any], Any]) -> None:
        if not callable(function):
            raise TypeError('function must be callable')

        super().__init__()
        self.function = function

    def map(self, element: Any) -> Any:
        return self.function(element)


class Filter(grain.FilterTransform):
    """Adapt a regular callable to a Grain filter transformation."""

    def __init__(self, function: Callable[[Any], bool]) -> None:
        if not callable(function):
            raise TypeError('function must be callable')

        super().__init__()
        self.function = function

    def filter(self, element: Any) -> bool:
        return self.function(element)


class IndexMap(grain.MapWithIndexTransform):
    """Adapt a regular callable to a Grain map with index transformation."""

    def __init__(self, function: Callable[[int, Any], Any]) -> None:
        if not callable(function):
            raise TypeError('function must be callable')

        super().__init__()
        self.function = function

    def map_with_index(self, idx: int, element: Any) -> Any:
        return self.function(idx, element)


class RandomMap(grain.RandomMapTransform):
    """Adapt a regular callable to a Grain random map transformation."""

    def __init__(self, function: Callable[[Any], Any]) -> None:
        if not callable(function):
            raise TypeError('function must be callable')

        super().__init__()
        self.function = function

    def map_with_index(self, element: Any, rng: Any) -> Any:
        return self.function(element, rng)


class _Unbatch(_FlatMapTransform):
    def __init__(self, max_fan_out: int) -> None:
        self.max_fan_out = max_fan_out

    def flat_map(self, element: Iterable[Any]) -> Iterable[Any]:
        return element


class BatchMap:
    """Apply one callable to buffered rows and emit rows individually.

    ``BatchMap`` expands into native Grain batch, map, and flat-map
    transformations. This preserves the cursor within a mapped batch when a
    dataloader iterator is checkpointed.
    """

    def __init__(
        self,
        function: Callable[[Sequence[Any]], Any],
        batch_size: int,
        *,
        drop_remainder: bool = False,
    ) -> None:
        if not callable(function):
            raise TypeError('function must be callable')
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError('batch_size must be a positive integer')
        if not isinstance(drop_remainder, bool):
            raise TypeError('drop_remainder must be a boolean')

        self.function = function
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder

    @staticmethod
    def _rows_from_mapping(
        columns: Mapping[str, Sequence[Any]],
        expected_size: int,
    ) -> tuple[dict[str, Any], ...]:
        normalized = {}
        for key, values in columns.items():
            if isinstance(values, (str, bytes)):
                raise ValueError(
                    f'batched output field {key!r} must have one value per row'
                )
            try:
                size = len(values)
            except TypeError as error:
                raise ValueError(
                    f'batched output field {key!r} must have one value per row'
                ) from error
            if size != expected_size:
                raise ValueError(
                    f'batched output field {key!r} returned {size} rows; '
                    f'expected {expected_size}'
                )
            normalized[key] = values

        return tuple(
            {
                key: values[index]
                for key, values in normalized.items()
            }
            for index in range(expected_size)
        )

    def _map_batch(self, rows: Sequence[Any]) -> tuple[Any, ...]:
        output = self.function(rows)
        if isinstance(output, Mapping):
            return self._rows_from_mapping(output, len(rows))
        if isinstance(output, (str, bytes)):
            raise TypeError(
                'batched map output must contain one result per input row'
            )
        try:
            output = tuple(output)
        except TypeError as error:
            raise TypeError(
                'batched map output must be a sequence or mapping'
            ) from error
        if len(output) != len(rows):
            raise ValueError(
                f'batched map returned {len(output)} rows; expected {len(rows)}'
            )
        return output

    def grain_operations(self) -> tuple[Any, ...]:
        """Return the native Grain transformations for this operation."""
        return (
            grain.Batch(
                self.batch_size,
                drop_remainder=self.drop_remainder,
                batch_fn=list,
            ),
            Map(self._map_batch),
            _Unbatch(self.batch_size),
        )


def _expand_operations(operations: Iterable[Any]) -> tuple[Any, ...]:
    expanded = []
    for operation in operations:
        expand = getattr(operation, 'grain_operations', None)
        if callable(expand):
            expanded.extend(_expand_operations(tuple(expand())))
        else:
            expanded.append(operation)
    return tuple(expanded)


def _prepare_grain_workers() -> None:
    """Allow Grain workers to start when Abseil flags are still unparsed."""
    if flags.FLAGS.is_parsed():
        return

    # Grain 0.2.18 reads this FlagHolder while constructing its worker pool.
    # Reading a holder before absl.app.run() raises in notebooks and regular
    # Python programs. The underlying Flag exposes the same live value without
    # requiring TakTiny to parse the application's complete flag registry.
    from grain._src.core import profiler

    name = '_GRAIN_ENABLE_MULTIPROCESS_WORKER_PROFILING'
    holder = getattr(profiler, name, None)
    if holder is None:
        return
    try:
        holder.value
    except UnparsedFlagAccessError:
        setattr(profiler, name, flags.FLAGS[holder.name])


class DatasetUtils:
    """Construct Grain dataloaders without prescribing data semantics."""

    @classmethod
    def from_datasets(
        cls,
        source: Sequence[Any],
        *,
        operations: Sequence[Any] = (),
        sampler: grain.Sampler | None = None,
        shuffle: bool = False,
        seed: int = 0,
        num_epochs: int | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
        worker_count: int | None = 0,
        worker_buffer_size: int = 1,
    ) -> grain.DataLoader:
        """Create a Grain loader from a random-access dataset.

        ``operations`` are applied exactly in the supplied order. Mapping,
        filtering, packing, batching, and collation therefore remain separate
        concerns and can be composed using Grain transformations or custom
        Grain operations.

        When ``sampler`` is omitted, an :class:`grain.IndexSampler` is created
        from the remaining sampling arguments. Supplying ``sampler`` transfers
        sampling and sharding responsibility entirely to that object. The
        default ``num_epochs=None`` creates an unbounded loader.
        """
        if not hasattr(source, '__getitem__'):
            raise TypeError('source must support random access')

        if operations is None or isinstance(operations, (str, bytes)):
            raise TypeError('operations must be a sequence')
        try:
            operations = tuple(operations)
        except TypeError as error:
            raise TypeError('operations must be a sequence') from error
        operations = _expand_operations(operations)

        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be a boolean')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError('seed must be an integer')
        if (
            num_epochs is not None
            and (
                isinstance(num_epochs, bool)
                or not isinstance(num_epochs, int)
                or num_epochs < 1
            )
        ):
            raise ValueError('num_epochs must be a positive integer or None')
        if (
            isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or shard_count < 1
        ):
            raise ValueError('shard_count must be a positive integer')
        if (
            isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or not 0 <= shard_index < shard_count
        ):
            raise ValueError(
                'shard_index must be between zero and shard_count - 1'
            )
        if (
            worker_count is not None
            and (
                isinstance(worker_count, bool)
                or not isinstance(worker_count, int)
                or worker_count < 0
            )
        ):
            raise ValueError('worker_count must be non-negative or None')
        if (
            isinstance(worker_buffer_size, bool)
            or not isinstance(worker_buffer_size, int)
            or worker_buffer_size < 1
        ):
            raise ValueError('worker_buffer_size must be a positive integer')

        if sampler is None:
            try:
                num_records = len(source)
            except TypeError as error:
                raise TypeError(
                    'source must have a finite length when sampler is omitted'
                ) from error

            sampler = grain.IndexSampler(
                num_records=num_records,
                num_epochs=num_epochs,
                shard_options=grain.ShardOptions(
                    shard_index=shard_index,
                    shard_count=shard_count,
                    drop_remainder=False,
                ),
                shuffle=shuffle,
                seed=seed,
            )

        if worker_count is None or worker_count > 0:
            _prepare_grain_workers()

        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=operations,
            worker_count=worker_count,
            worker_buffer_size=worker_buffer_size,
        )


__all__ = [
    'DatasetUtils',
    'Map',
    'BatchMap',
    'Filter',
    'IndexMap',
    'RandomMap',
]
