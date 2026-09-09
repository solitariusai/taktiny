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
"""Load already-available, random-access data without prescribing a modality."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol

import grain.python as grain
import numpy as np
from absl import flags
from absl.flags import UnparsedFlagAccessError
from grain._src.python.dataset import base as dataset_base

from taktiny.data.transforms import Batch, _expand_operations


class RandomAccessSource(Protocol):
    """Structural interface for finite sources; no framework inheritance needed."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> Any: ...


class _ShardSampler:
    """Present a shard as a local sampler, including uneven and empty tails.

    Grain 0.2.18 DataLoader floors sampler length / shard_count even when
    drop_remainder=False. Do the partitioning here and disable its second
    partitioning step. Keep global RNGs but expose local traversal indices.
    """

    def __init__(
        self, sampler: grain.IndexSampler | None,
        shard_index: int, shard_count: int,
    ) -> None:
        self.sampler = sampler
        self.shard_index = shard_index
        self.shard_count = shard_count
        self._shard_options = grain.NoSharding()

    def __len__(self) -> int:
        size = len(self.sampler) if self.sampler is not None else 0
        if size == sys.maxsize:
            return sys.maxsize
        return max(0, (size - self.shard_index + self.shard_count - 1) // self.shard_count)

    def __getitem__(self, index: int) -> grain.RecordMetadata:
        if index < 0 or index >= len(self) or self.sampler is None:
            raise IndexError(index)
        record = self.sampler[index * self.shard_count + self.shard_index]
        return replace(record, index=index)

    def __repr__(self) -> str:
        return (f'_ShardSampler({self.sampler!r}, shard_index={self.shard_index}, '
                f'shard_count={self.shard_count})')


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
        holder.value  # noqa: B018
    except UnparsedFlagAccessError:
        setattr(profiler, name, flags.FLAGS[holder.name])


class DataLoader(grain.DataLoader):
    """Preprocess and iterate over caller-provided, random-access records.

    Records may be arrays, mappings, tuples, images, audio, strings, or custom
    objects. The source is never downloaded, decoded, or copied into memory.
    Supply a list, array, already-loaded dataset, or an object implementing
    __len__ and integer __getitem__. This loader does not accept streaming
    generators; use a streaming backend directly, or explicitly materialize a
    finite stream with list(source) if it fits in memory.

    Args:
        source: Caller-owned random-access data, not a repository ID or path.
        operations: Ordered Taktiny or native Grain operations. Use Map for a
            record callable, RandomMap for augmentation, and Filter to drop
            records. Operations are lazy and run when the loader is iterated.
        batch_size: Optional final batch size, applied after all operations.
            None emits records unchanged. For operations after batching, put
            Batch directly in operations and leave this argument unset.
        drop_remainder: Drop an incomplete final batch; requires batch_size.
        collate_fn: Optional function(rows) for final batching. None uses Grain
            stacking; list preserves ragged or custom objects without padding.
            Requires batch_size. Collation runs independently in each worker.
        sampler: Optional native Grain sampler. When provided, it owns sampling,
            epochs, seed, and sharding; the corresponding convenience arguments
            are ignored after validation.
        shuffle: Shuffle indices using seed; defaults to False.
        seed: Unsigned 32-bit integer seed for sampling and RandomMap augmentation.
        num_epochs: Positive epoch count (default 1); None repeats indefinitely.
            Each new iterator starts from the beginning unless state is restored.
        shard_index: This process's data shard, in [0, shard_count).
        shard_count: Number of data shards. Shards may have unequal lengths;
            this is input partitioning, not JAX device-array sharding.
        worker_count: Child workers; 0 runs locally, None lets Grain choose.
            Sources and transforms must be serializable when workers are used.
        worker_buffer_size: Positive per-worker prefetch buffer size.

    Iterators retain Grain's get_state()/set_state() checkpoint API. Restore
    against the same source and pipeline. Custom iterator operations retain
    their own Grain checkpoint limitations. Transforms should be deterministic
    apart from RandomMap's supplied RNG and should not mutate source records.

    Example:
        >>> from taktiny.data import DataLoader, MapFields
        >>> rows = [{'value': 255, 'label': 0}, {'value': 0, 'label': 1}]
        >>> loader = DataLoader(rows, operations=[
        ...     MapFields({'value': lambda x: x / 255})], batch_size=2)
        >>> next(iter(loader))['value'].tolist()
        [1.0, 0.0]
    """
    def __init__(
        self,
        source: dataset_base.RandomAccessDataSource | Any,
        *,
        operations: Sequence[Any] = (),
        batch_size: int | None = None,
        drop_remainder: bool = False,
        collate_fn: Callable[[Sequence[Any]], Any] | None = None,
        sampler: grain.Sampler | None = None,
        shuffle: bool = False,
        seed: int = 0,
        num_epochs: int | None = 1,
        shard_index: int = 0,
        shard_count: int = 1,
        worker_count: int | None = 0,
        worker_buffer_size: int = 1,
    ) -> None:
        """Create a Grain loader from a random-access dataset.

        ``operations`` are applied exactly in the supplied order. Mapping,
        filtering, packing, batching, and collation therefore remain separate
        concerns and can be composed using Grain transformations or custom
        Grain operations.

        When ``sampler`` is omitted, an :class:`grain.IndexSampler` is created
        from the remaining sampling arguments. Supplying ``sampler`` transfers
        sampling and sharding responsibility entirely to that object. The
        default ``num_epochs=1`` creates a single-epoch (finite) loader; pass
        ``None`` for an unbounded loader.
        """
        _validate_source(source)

        if operations is None or isinstance(operations, (str, bytes)):
            raise TypeError('operations must be a sequence')

        try:
            operations = tuple(operations)
        except TypeError as error:
            raise TypeError('operations must be a sequence') from error

        operations = _expand_operations(operations)
        if not isinstance(drop_remainder, bool):
            raise TypeError('drop_remainder must be a boolean')

        if batch_size is None:
            if drop_remainder or collate_fn is not None:
                raise ValueError('drop_remainder and collate_fn require batch_size')
        else:
            operations += (Batch(batch_size, drop_remainder=drop_remainder,
                                 collate_fn=collate_fn),)

        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be a boolean')

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError('seed must be an integer')

        if not 0 <= seed < 2**32:
            raise ValueError('seed must be an unsigned 32-bit integer')

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

            base_sampler = grain.IndexSampler(
                num_records=num_records,
                num_epochs=num_epochs,
                shard_options=grain.NoSharding(),
                shuffle=shuffle,
                seed=seed,
            ) if num_records else None
            sampler = _ShardSampler(base_sampler, shard_index, shard_count)

        if worker_count is None or worker_count > 0:
            _prepare_grain_workers()

        super().__init__(
            data_source=source,
            sampler=sampler,
            operations=operations,
            worker_count=worker_count,
            worker_buffer_size=worker_buffer_size,
        )



def _validate_source(source: Any) -> None:
    if isinstance(source, (str, bytes, Mapping)):
        raise TypeError('source must contain records, not a path, repository ID, or column mapping')
    if not hasattr(source, '__getitem__'):
        raise TypeError('source must support random access; materialize finite iterables explicitly')


def train_validation_split(
    source: RandomAccessSource,
    validation_size: float,
    *,
    shuffle: bool = True,
    seed: int = 0,
) -> tuple[Any, Any]:
    """Split a random-access source into ``(train, validation)`` views.

    ``validation_size`` may be a count (``int``) or a fraction (``float``
    in ``(0, 1)``). The returned views are random-access and can be passed
    directly to DataLoader. Fractions are rounded to the nearest integer;
    both splits must be nonempty. The source itself is not copied or shuffled.
    """
    _validate_source(source)
    if not isinstance(shuffle, bool):
        raise TypeError('shuffle must be a boolean')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError('seed must be an integer')

    count = len(source)
    if isinstance(validation_size, bool):
        raise TypeError('validation_size must be an int or float')
    if isinstance(validation_size, float):
        if not 0.0 < validation_size < 1.0:
            raise ValueError('float validation_size must be in (0, 1)')
        validation_count = round(count * validation_size)
    elif isinstance(validation_size, int):
        if not 0 < validation_size < count:
            raise ValueError(
                f'int validation_size must be in (0, {count})'
            )
        validation_count = validation_size
    else:
        raise TypeError('validation_size must be an int or float')

    if not 0 < validation_count < count:
        raise ValueError('validation_size must leave both splits nonempty')

    indices = np.arange(count)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(count)
    train = _IndexedView(source, indices[validation_count:])
    validation = _IndexedView(source, indices[:validation_count])
    return train, validation


class _IndexedView:
    """Random-access view over a subset of a source's indices."""

    def __init__(self, source: RandomAccessSource, indices: np.ndarray) -> None:
        self._source = source
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return _IndexedView(self._source, self._indices[index])
        return self._source[int(self._indices[index])]


__all__ = ['DataLoader', 'RandomAccessSource', 'train_validation_split']
