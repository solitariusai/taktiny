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
"""Modality-independent, composable Grain transformations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal

import grain.python as grain
import numpy as np
from grain._src.core.transforms import FlatMap as _FlatMapTransform


class Map(grain.MapTransform):
    """Apply a callable to one record and return its replacement.

    Records may be arrays, mappings, or arbitrary objects. The function decides
    what fields survive; Map does not merge its result with the original input.
    Use directly as a callable or as a DataLoader operation. Functions should
    avoid mutating source records and be serializable when using worker processes.

    Args:
        function: Callable receiving one record and returning one output record.

    Example:
        >>> from taktiny.data import DataLoader, Map
        >>> operation = Map(lambda row: {'value': row['value'] * 2})
        >>> operation({'value': 3})
        {'value': 6}
        >>> list(DataLoader([{'value': 1}, {'value': 2}], operations=[operation]))
        [{'value': 2}, {'value': 4}]
    """

    def __init__(self, function: Callable[[Any], Any]) -> None:
        if not callable(function):
            raise TypeError('function must be callable')

        super().__init__()
        self.function = function

    def map(self, element: Any) -> Any:
        return self.function(element)

    def __call__(self, element: Any) -> Any:
        return self.map(element)

class Filter(grain.FilterTransform):
    """Keep records for which a predicate returns True.

    Within DataLoader, rejected records are omitted without replacing or
    modifying accepted records. Calling Filter directly returns the predicate
    result, not the input record. Put it before Batch to filter individual rows.

    Args:
        function: Callable receiving one record and returning a boolean.

    Example:
        >>> from taktiny.data import DataLoader, Filter
        >>> keep = Filter(lambda value: value >= 0)
        >>> keep(-1)
        False
        >>> list(DataLoader([-1, 0, 2], operations=[keep]))
        [0, 2]
    """

    def __init__(self, function: Callable[[Any], bool]) -> None:
        if not callable(function):
            raise TypeError('function must be callable')

        super().__init__()
        self.function = function

    def filter(self, element: Any) -> bool:
        return self.function(element)

    def __call__(self, element: Any) -> bool:
        return self.filter(element)

class IndexMap(grain.MapWithIndexTransform):
    """Map records together with their sampler traversal indices.

    The index is not necessarily the original source row key when shuffling,
    nor a dense output index after filtering. After batching, it belongs to
    the final contributing record. The default loader's sharded sampler uses
    local traversal indices. Use map_with_index(index, element) for a direct
    call; __call__ adapts an iterator of Grain Records for DataLoader.

    Args:
        function: Callable receiving (index, record) and returning a replacement
            record. Existing fields are preserved only if the function keeps them.

    Example:
        >>> from taktiny.data import DataLoader, IndexMap
        >>> operation = IndexMap(lambda index, value: {'index': index, 'value': value})
        >>> list(DataLoader([10, 20], operations=[operation], shuffle=False))
        [{'index': 0, 'value': 10}, {'index': 1, 'value': 20}]
    """

    def __init__(self, function: Callable[[int, Any], Any]) -> None:
        if not callable(function):
            raise TypeError('function must be callable')

        super().__init__()
        self.function = function

    def map_with_index(self, index: int, element: Any) -> Any:
        return self.function(index, element)

    def __call__(self, records: Iterable[grain.Record]) -> Iterator[grain.Record]:
        # Grain DataLoader does not dispatch MapWithIndexTransform natively.
        for record in records:
            yield grain.Record(
                metadata=record.metadata,
                data=self.map_with_index(record.metadata.index, record.data),
            )

class RandomMap(grain.RandomMapTransform):
    """Apply random augmentation using Grain's per-record NumPy Generator.

    The loader's sampler supplies the RNG. Use it instead of global random
    state for reproducible results and checkpoint restoration. With the default
    sampler, seed controls the stream and a new iterator starts it again.
    Direct calls require an explicit NumPy Generator, not a JAX random key.

    Args:
        function: Callable receiving (record, rng) and returning one replacement
            record. Avoid mutating the original record.

    Example:
        >>> from taktiny.data import DataLoader, RandomMap
        >>> augment = RandomMap(lambda value, rng: value + int(rng.integers(10)))
        >>> loader = DataLoader([1, 2, 3], operations=[augment], seed=42)
        >>> first = list(loader)
        >>> first == list(loader)
        True
    """

    def __init__(self, function: Callable[[Any, np.random.Generator], Any]) -> None:
        if not callable(function):
            raise TypeError('function must be callable')

        super().__init__()
        self.function = function

    def random_map(self, element: Any, rng: np.random.Generator) -> Any:
        return self.function(element, rng)

    def __call__(self, element: Any, rng: np.random.Generator) -> Any:
        return self.random_map(element, rng)

class Compose(grain.MapTransform):
    """Apply deterministic record functions from left to right.

    Accepts callables and Grain MapTransforms. With no functions this is an
    identity. Works directly on an example or as a DataLoader operation, and
    never iterates over examples implicitly. Use Filter, RandomMap, and Batch
    as separate loader operations, not inside Compose.

    Args:
        *functions: Ordered callables or Grain MapTransforms. Each receives
            the preceding function's result. An empty composition is an identity.

    Example:
        >>> from taktiny.data import Compose
        >>> Compose(lambda x: x + 1, lambda x: x * 2)(3)
        8
    """

    def __init__(self, *functions: Callable[[Any], Any] | grain.MapTransform) -> None:
        normalized = []
        for function in functions:
            if isinstance(function, grain.MapTransform):
                normalized.append(function.map)
            elif isinstance(function, (grain.FilterTransform, grain.RandomMapTransform,
                                       grain.MapWithIndexTransform)):
                raise TypeError('Compose only accepts deterministic record maps')
            elif callable(function):
                normalized.append(function)
            else:
                raise TypeError('Compose functions must be callable or MapTransforms')
        self.functions = tuple(normalized)

    def map(self, element: Any) -> Any:
        for function in self.functions:
            element = function(element)
        return element

    def __call__(self, element: Any) -> Any:
        return self.map(element)

class MapFields(grain.MapTransform):
    """Transform selected top-level mapping values, preserving other fields.

    Each function receives its field's value, not the complete record. Missing
    keys raise KeyError. A new dictionary is returned; unselected values are
    shared, not deep-copied. Functions should avoid mutating their inputs.
    Use Map for nested structures, renaming, or computations across fields.

    Args:
        functions: Mapping from top-level field keys to value-transforming
            callables. The mapping is copied at construction. An empty mapping
            returns a shallow copy of each input record.

    Example:
        >>> from taktiny.data import MapFields
        >>> MapFields({'value': lambda x: x / 255})({'value': 255, 'label': 2})
        {'value': 1.0, 'label': 2}
    """

    def __init__(self, functions: Mapping[Any, Callable[[Any], Any]]) -> None:
        if not isinstance(functions, Mapping):
            raise TypeError('functions must be a mapping of field keys to callables') 

        if any(not callable(function) for function in functions.values()):
            raise TypeError('field functions must be callable')

        self.functions = dict(functions)

    def map(self, element: Mapping[Any, Any]) -> dict[Any, Any]:
        if not isinstance(element, Mapping):
            raise TypeError('MapFields expects a mapping record')
        result = dict(element)
        for key, function in self.functions.items():
            result[key] = function(element[key])
        return result

    def __call__(self, element: Mapping[Any, Any]) -> dict[Any, Any]:
        return self.map(element)

class Batch(grain.Batch):
    """Group consecutive records, optionally using collate_fn(rows).

    The default stacks matching leaves into NumPy arrays, preserving nested
    structure. It does not pad ragged data. Pass collate_fn=list to retain raw
    rows, or a custom callable for padding, audio, images, or arbitrary objects.
    With multiple workers, batching occurs within each worker independently.

    Args:
        batch_size: Positive maximum number of consecutive records in a batch.
        drop_remainder: If True, discard an incomplete final batch. Defaults to
            False, which emits it with a smaller leading batch dimension.
        collate_fn: Optional callable receiving a sequence of rows and returning
            any batch structure. None uses Grain's default leaf-wise stacking.

    Example:
        >>> from taktiny.data import Batch, DataLoader
        >>> loader = DataLoader([1, 2, 3], operations=[Batch(2)])
        >>> [batch.tolist() for batch in loader]
        [[1, 2], [3]]
        >>> ragged = DataLoader([[1], [2, 3]], operations=[Batch(2, collate_fn=list)])
        >>> list(ragged)
        [[[1], [2, 3]]]
    """

    def __init__(
        self, 
        batch_size: int, 
        *, 
        drop_remainder: bool = False,
        collate_fn: Callable[[Sequence[Any]], Any] | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if not isinstance(drop_remainder, bool):
            raise TypeError('drop_remainder must be a boolean')
        if collate_fn is not None and not callable(collate_fn):
            raise TypeError('collate_fn must be callable or None')
        super().__init__(batch_size, drop_remainder=drop_remainder, batch_fn=collate_fn)

class FlatMap(_FlatMapTransform):
    """Expand each example into zero or more examples, up to max_fan_out.

    function(element) returns an iterable of records, not a column mapping.
    Expansion is bounded so Grain can checkpoint within an expanded record.
    Use this for windows, patches, segments, or any cardinality-changing map.

    Args:
        function: Callable receiving one record and returning an iterable of
            output records. Strings, bytes, and column mappings are not accepted
            as that iterable; wrap such records in a list instead.
        max_fan_out: Positive upper bound on outputs per input record. Exceeding
            it raises ValueError. An empty output iterable drops the input.

    Example:
        >>> from taktiny.data import DataLoader, FlatMap
        >>> split = FlatMap(lambda values: values, max_fan_out=3)
        >>> list(DataLoader([[1, 2], [], [3]], operations=[split]))
        [1, 2, 3]
        >>> split.flat_map([4, 5])
        (4, 5)
    """

    def __init__(self, function: Callable[[Any], Iterable[Any]], *, max_fan_out: int) -> None:
        if not callable(function):
            raise TypeError('function must be callable')
        if isinstance(max_fan_out, bool) or not isinstance(max_fan_out, int) or max_fan_out < 1:
            raise ValueError('max_fan_out must be a positive integer')
        self.function = function
        self.max_fan_out = max_fan_out

    def flat_map(self, element: Any) -> tuple[Any, ...]:
        output = self.function(element)
        if isinstance(output, (str, bytes, Mapping)):
            raise TypeError('FlatMap output must be an iterable of records')
        rows = []
        for row in output:
            if len(rows) == self.max_fan_out:
                raise ValueError('FlatMap output exceeds max_fan_out')
            rows.append(row)
        return tuple(rows)

class _Unbatch(_FlatMapTransform):
    def __init__(self, max_fan_out: int) -> None:
        self.max_fan_out = max_fan_out

    def flat_map(self, element: Sequence[Any]) -> Sequence[Any]:
        return element

class BatchMap:
    """Apply one callable to buffered rows and emit rows individually.

    ``BatchMap`` expands into native Grain batch, map, and flat-map
    transformations. This preserves the cursor within a mapped batch when a
    dataloader iterator is checkpointed.

    Unlike Batch, the output remains a stream of individual records. The
    function must preserve row count and order. Use FlatMap when changing
    cardinality; use Batch for final training batches.

    Args:
        function: Callable receiving a sequence of raw rows, not stacked columns.
            Return either one output per row, or a mapping of columns whose
            values each have the same length as the input buffer.
        batch_size: Positive maximum number of input rows per function call.
        drop_remainder: Drop an incomplete final input buffer when True. With
            the default False, the function also receives the smaller buffer.

    Example:
        >>> from taktiny.data import BatchMap, DataLoader
        >>> operation = BatchMap(lambda rows: {'value': [x * 2 for x in rows]}, 2)
        >>> list(DataLoader([1, 2, 3], operations=[operation]))
        [{'value': 2}, {'value': 4}, {'value': 6}]
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
                raise TypeError(
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

class ApplyTemplate(grain.MapTransform):
    """Format nested templates from mapping records and attach the result.

    String leaves use str.format_map with the input record. Mapping values,
    lists, and tuples are formatted recursively; mapping keys and other leaves
    are kept as-is. A new record dictionary is returned, preserving other
    fields. Missing formatting fields raise KeyError. No tokenizer or text
    model is involved; templates can describe paths, metadata, or messages.

    Args:
        template: String or nested structure to format. Containers are rebuilt
            for each record; arbitrary non-container leaves are shared.
        format_fn: Optional keyword-only callable applied once to the complete
            formatted structure. None leaves it unchanged. The callable should
            avoid mutating shared template leaves.
        return_key: Nonempty output field name; defaults to 'template'. An existing
            field with this name is replaced in the returned record only.

    Example:
        >>> from taktiny.data import ApplyTemplate
        >>> operation = ApplyTemplate({'path': '{folder}/{name}.png'}, return_key='asset')
        >>> operation({'folder': 'images', 'name': 'cat'})['asset']
        {'path': 'images/cat.png'}
        >>> join = ApplyTemplate(['{first}', '{last}'], format_fn=' '.join, return_key='name')
        >>> join({'first': 'Ada', 'last': 'Lovelace'})['name']
        'Ada Lovelace'
    """

    def __init__(
        self,
        template: Any,
        *,
        format_fn: Callable[[Any], Any] | None = None,
        return_key: str = 'template',
    ) -> None:
        
        if format_fn is not None and not callable(format_fn):
            raise TypeError('format_fn must be callable or None')

        if not isinstance(return_key, str) or not return_key:
            raise TypeError('return_key must be a non-empty string')

        super().__init__()
        self.template = template
        self.return_key = return_key
        self.format_fn = format_fn

    @classmethod
    def _format(cls, template: Any, element: Mapping[str, Any]) -> Any:
        if isinstance(template, str):
            return template.format_map(element)

        if isinstance(template, Mapping):
            return {
                key: cls._format(value, element)
                for key, value in template.items()
            }

        if isinstance(template, list):
            return [cls._format(value, element) for value in template]

        if isinstance(template, tuple):
            return tuple(cls._format(value, element) for value in template)

        return template

    def map(self, element: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(element, Mapping):
            raise TypeError('ApplyTemplate expects mapping records')

        value = self._format(self.template, element)
        if self.format_fn is not None:
            value = self.format_fn(value)

        result = dict(element)
        result[self.return_key] = value
        return result

    def __call__(self, element: Mapping[str, Any]) -> dict[str, Any]:
        return self.map(element)

class Pack:
    """Concatenate aligned array fields into fixed-length records.

    Args:
        length: Positive number of steps along each field's packing axis.
        keys: One field name or a nonempty sequence of field names to pack.
            Only these fields and
            requested generated fields are emitted; other input fields are
            discarded because multiple source records can share one pack.
        axis: Integer packing axis for all fields, or a mapping containing one
            axis per key. Negative axes count from the end of each array.
        padding_values: Per-field scalar padding values; all default to zero.
            Padding is cast to the field's dtype, which is preserved.
        position_key: Optional output name for a 1-D int32 vector of positions
            within each source record. Positions reset at record boundaries
            and continue across split fragments. Padding positions are zero.
        mask_key: Optional name of a shared 1-D input/output validity mask.
            If present in an input, nonzero entries select steps from every
            field before packing. Missing input masks mean all steps are valid.
            Output masks are int32, one for data and zero for padding. Positions
            refer to the compressed sequence after mask selection.
        overflow: 'split' preserves all steps, carrying overflow into subsequent
            packs. 'truncate' fills the current pack and discards the rest of
            that source record, even if it could fill another pack.
        drop_remainder: Drop the final partial pack instead of right-padding it.

    Fields must have equal lengths along their respective packing axes within
    each record. Each field must retain its dtype, rank, and non-packing shape
    across nonempty records. Different fields may have different dtypes/shapes.
    No token, label, or language-model conventions are applied. Inputs are not
    mutated; empty records are skipped. Arrays are processed with NumPy on CPU.

    Use pack(records) for a lazy iterable of ordinary mapping records, or put
    this operation before Batch in a DataLoader. State belongs to each iterator,
    not the Pack object. In a loader, each worker/shard packs independently.
    Like the legacy PackSequences iterator operation, buffered packing state is
    not captured by Grain checkpoints: mid-pack resume is not supported.

    Packing runs on individual records before the loader creates a batch.
    For a record shaped (time, channels), axis=0 packs time; final batching
    produces (batch, length, channels). No batch axis is assumed inside Pack.

    Example:
        >>> import numpy as np
        >>> from taktiny.data import DataLoader
        >>> from taktiny.data.transforms import Pack
        >>> packer = Pack(4, keys=('audio',), mask_key='valid')
        >>> records = [{'audio': np.ones((3, 2), dtype=np.float32)}]
        >>> result = next(packer.pack(records))
        >>> result['audio'].shape
        (4, 2)
        >>> result['valid'].tolist()
        [1, 1, 1, 0]
        >>> loader = DataLoader(records, operations=[Pack(4, keys='audio')], batch_size=2)
        >>> next(iter(loader))['audio'].shape
        (1, 4, 2)
    """

    def __init__(
        self,
        length: int,
        *,
        keys: str | Sequence[str],
        axis: int | Mapping[str, int] = 0,
        padding_values: Mapping[str, Any] | None = None,
        position_key: str | None = None,
        mask_key: str | None = None,
        overflow: Literal['split', 'truncate'] = 'split',
        drop_remainder: bool = False,
    ) -> None:
        if isinstance(keys, str):
            keys = (keys,)
            
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError('length must be a positive integer')
        if (not isinstance(keys, Sequence) or isinstance(keys, (str, bytes))
                or not keys or any(not isinstance(key, str) or not key for key in keys)):
            raise TypeError('keys must be a non-empty sequence of strings')
        if len(set(keys)) != len(keys):
            raise ValueError('keys must not contain duplicates')
        for name, value in [('position_key', position_key), ('mask_key', mask_key)]:
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f'{name} must be a non-empty string or None')
        output_keys = (*keys, *(k for k in (position_key, mask_key) if k is not None))
        if len(set(output_keys)) != len(output_keys):
            raise ValueError('packing output field names must be unique')
        if isinstance(axis, Mapping):
            if set(axis) != set(keys):
                raise ValueError('axis must contain exactly one entry per packed key')
            axes = dict(axis)
        else:
            axes = dict.fromkeys(keys, axis)
        if any(isinstance(a, bool) or not isinstance(a, int) for a in axes.values()):
            raise TypeError('axis values must be integers')
        if padding_values is not None and not isinstance(padding_values, Mapping):
            raise TypeError('padding_values must be a mapping or None')
        if set(padding_values or ()) - set(keys):
            raise ValueError('padding_values contains unknown fields')
        padding = {key: 0 for key in keys}
        padding.update(padding_values or {})
        if any(np.ndim(value) != 0 for value in padding.values()):
            raise ValueError('padding_values must contain scalars')
        if overflow not in ('split', 'truncate'):
            raise ValueError('overflow must be "split" or "truncate"')
        if not isinstance(drop_remainder, bool):
            raise TypeError('drop_remainder must be a boolean')

        self.length = length
        self.keys = tuple(keys)
        self.axes = axes
        self.padding_values = padding
        self.position_key = position_key
        self.mask_key = mask_key
        self.overflow = overflow
        self.drop_remainder = drop_remainder

    def _normalize_record(self, value: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], int]:
        if not isinstance(value, Mapping):
            raise TypeError('Pack expects mapping records')
        arrays: dict[str, np.ndarray] = {}
        length = None
        for key in self.keys:
            if key not in value:
                raise KeyError(f'packed record is missing {key!r}')
            array = np.asarray(value[key])
            axis = self.axes[key]
            if not -array.ndim <= axis < array.ndim:
                raise ValueError(f'axis={axis} is out of range for {key!r} with rank {array.ndim}')
            array = np.moveaxis(array, axis, 0)
            if length is not None and len(array) != length:
                raise ValueError('all packed fields must have equal lengths along their packing axes')
            length = len(array)
            arrays[key] = array
        assert length is not None
        if self.mask_key is not None and self.mask_key in value:
            mask = np.asarray(value[self.mask_key])
            if mask.ndim != 1:
                raise ValueError(f'{self.mask_key!r} must be one-dimensional before packing')
            if len(mask) != length:
                raise ValueError('mask and packed fields must have equal lengths')
            valid = mask.astype(bool, copy=False)
            arrays = {key: array[valid] for key, array in arrays.items()}
            length = int(np.count_nonzero(valid))
        return arrays, length

    def _finish_pack(
        self, values: Mapping[str, Sequence[np.ndarray]], positions: Sequence[int],
    ) -> dict[str, np.ndarray]:
        padding = self.length - len(positions)
        packed = {}
        for key in self.keys:
            array = np.concatenate(values[key], axis=0)
            if padding:
                array = np.pad(array, ((0, padding),) + ((0, 0),) * (array.ndim - 1),
                               constant_values=self.padding_values[key])
            packed[key] = np.moveaxis(array, 0, self.axes[key])
        if self.position_key is not None:
            packed[self.position_key] = np.pad(np.asarray(positions, dtype=np.int32), (0, padding))
        if self.mask_key is not None:
            packed[self.mask_key] = np.pad(np.ones(len(positions), dtype=np.int32), (0, padding))
        return packed

    def pack(self, records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, np.ndarray]]:
        """Lazily pack plain records without a DataLoader or materializing a stream."""
        wrapped = (grain.Record(grain.RecordMetadata(index=i), value)
                   for i, value in enumerate(records))
        for record in self(wrapped):
            yield record.data

    def __call__(self, records: Iterable[grain.Record]) -> Iterator[grain.Record]:
        """Grain iterator-operation adapter; use pack() for ordinary records."""
        values: dict[str, list[np.ndarray]] = {key: [] for key in self.keys}
        positions: list[int] = []
        schema: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        metadata = None
        for record in records:
            arrays, length = self._normalize_record(record.data)
            if not length:
                continue
            for key, array in arrays.items():
                actual = (array.shape[1:], array.dtype)
                if key in schema and schema[key] != actual:
                    raise ValueError(f'{key!r} must preserve its non-packing shape and dtype across records')
                schema[key] = actual
            offset = 0
            while offset < length:
                stop = min(offset + self.length - len(positions), length)
                for key in self.keys:
                    values[key].append(arrays[key][offset:stop])
                positions.extend(range(offset, stop))
                metadata = record.metadata.remove_record_key()
                offset = stop
                if len(positions) == self.length:
                    output = self._finish_pack(values, positions)
                    values = {key: [] for key in self.keys}
                    positions = []
                    yield grain.Record(metadata, output)
                if self.overflow == 'truncate':
                    break
        if positions and not self.drop_remainder:
            assert metadata is not None
            yield grain.Record(metadata, self._finish_pack(values, positions))


__all__ = [
    'ApplyTemplate',
    'Batch',
    'BatchMap',
    'Compose',
    'Filter',
    'FlatMap',
    'IndexMap',
    'Map',
    'MapFields',
    'Pack',
    'RandomMap',
]
