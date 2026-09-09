# Data Loading & Transforms

`taktiny.data` turns records you provide into batches. It uses Grain for
iteration and supports arrays, mappings, images, audio, text, and other Python
objects. It does not download datasets or choose how to decode your files.

The usual order is: read a record, transform it, then batch it. Packing and
custom collation are optional steps, not assumptions about your data.

## Start with records

Pass `DataLoader` a random-access source: a list, an array, an already-loaded
dataset, or an object with `__len__` and integer `__getitem__`. A record is one
item returned by `source[index]`; it need not be a dictionary.

This example normalizes small image arrays while keeping their labels:

```python
import numpy as np
from taktiny.data import DataLoader, MapFields

records = [
    {"image": np.full((4, 4, 1), value, dtype=np.uint8), "label": label}
    for value, label in [(0, 0), (128, 1), (255, 1)]
]

loader = DataLoader(
    records,
    operations=[MapFields({"image": lambda x: x.astype(np.float32) / 255})],
    batch_size=2,
)

batches = list(loader)
assert batches[0]["image"].shape == (2, 4, 4, 1)
assert batches[0]["label"].tolist() == [0, 1]
assert batches[1]["image"].shape == (1, 4, 4, 1)
```

Operations run lazily as you iterate. By default, the loader preserves source
order, makes one pass, and runs in the current process (`worker_count=0`).
Final batching stacks matching leaves into NumPy arrays. It does not move them
to JAX devices; transfer numeric batches explicitly with `jax.device_put` when
needed.

Do not pass a repository ID, a file path, or a column dictionary as the source.
Load or wrap the data yourself first. Streaming generators are not supported by
`DataLoader`; use a streaming backend directly, or materialize a finite stream
only if it fits in memory.

## Choose an operation

Operations in `operations=[...]` run in the order you supply them. Taktiny's
wrappers can be mixed with native Grain operations.

| Operation | What it receives and does |
| --- | --- |
| `Map(fn)` | Passes a whole record to `fn` and emits the result. |
| `MapFields({...})` | Transforms selected top-level fields, preserving the others. |
| `Compose(...)` | Chains record-level callables or Grain map transforms. |
| `Filter(fn)` | Keeps a record when `fn(record)` is true. |
| `IndexMap(fn)` | Calls `fn(index, record)` with Grain's supplied index. |
| `RandomMap(fn)` | Calls `fn(record, rng)` with a NumPy random generator. |
| `FlatMap(fn, max_fan_out=...)` | Expands a record into a bounded number of records. |
| `BatchMap(fn, batch_size=...)` | Processes buffered rows together, then emits individual rows. |
| `Batch(...)` | Groups records into a batch at that point in the pipeline. |
| `ApplyTemplate(...)` | Formats a template using a mapping record's fields. |
| `Pack(...)` | Concatenates aligned array fields into fixed-length records. |

Use `Map` for computations involving several fields, renaming keys, or nested
structures. Use `MapFields` when each selected field can be handled separately.
Transforms should return new values rather than mutate source records;
`MapFields` makes a shallow copy, not a deep copy of untouched values.

## Control where batching happens

`DataLoader(..., batch_size=32)` appends batching **after every operation**.
To run an operation on a completed batch, put `Batch` inside the pipeline and
leave the loader's `batch_size` unset:

```python
import numpy as np
from taktiny.data import Batch, DataLoader, Map

loader = DataLoader(
    [1, 2, 3, 4],
    operations=[
        Map(np.float32),
        Batch(2),
        Map(lambda batch: batch - batch.mean()),
    ],
)
assert [batch.tolist() for batch in loader] == [[-0.5, 0.5], [-0.5, 0.5]]
```

Setting both an explicit `Batch` and the loader's `batch_size` batches twice.
Use `drop_remainder=True` on the relevant batch operation when you need every
batch to have the same leading dimension. Otherwise the final batch can be
smaller. With multiple workers, batching happens independently in each worker.

### Ragged records

Default batching does not pad unequal array shapes. Supply a `collate_fn` that
pads or combines your records, or use `list` to keep them as individual objects:

```python
import numpy as np
from taktiny.data import DataLoader

loader = DataLoader(
    [np.arange(2), np.arange(5)],
    batch_size=2,
    collate_fn=list,
)
batch = next(iter(loader))
assert [row.shape for row in batch] == [(2,), (5,)]
```

A custom collator receives a sequence of rows, not a dictionary of stacked
columns. Its return value becomes the batch.

### Buffered preprocessing is not final batching

`BatchMap` is useful when a decoder or another preprocessing function can
process several records at once. Its function receives raw rows and must
preserve their count and order. It can return rows or a mapping of columns:

```python
from taktiny.data import BatchMap, DataLoader

loader = DataLoader(
    [1, 2, 3],
    operations=[BatchMap(lambda rows: {"value": [x * 2 for x in rows]}, 2)],
)
assert list(loader) == [{"value": 2}, {"value": 4}, {"value": 6}]
```

The output is still a stream of individual records. Add final batching
separately if needed. Use `FlatMap`, not `BatchMap`, to change the row count.

## Pack aligned sequences

`Pack` concatenates steps from several records into fixed-length arrays. It is
not text-specific: the packing axis could represent audio samples, sensor
readings, frames, or token positions.

Put `Pack` **before batching**. Its default `axis=0` refers to an individual
record, not a batch. A record shaped `(time, channels)` becomes
`(length, channels)` after packing, then `(batch, length, channels)` after
batching.

```python
import numpy as np
from taktiny.data import DataLoader, Pack

records = [
    {
        "signal": np.full((steps, 2), value, dtype=np.float32),
        "target": np.full(steps, value, dtype=np.int32),
        "name": name,
    }
    for steps, value, name in [(3, 1, "first"), (2, 2, "second")]
]

loader = DataLoader(
    records,
    operations=[Pack(
        4,
        keys=("signal", "target"),
        mask_key="valid",
        position_key="position",
    )],
    batch_size=2,
)
batch = next(iter(loader))

assert batch["signal"].shape == (2, 4, 2)
assert batch["target"].tolist() == [[1, 1, 1, 2], [2, 0, 0, 0]]
assert batch["valid"].tolist() == [[1, 1, 1, 1], [1, 0, 0, 0]]
assert batch["position"].tolist() == [[0, 1, 2, 0], [1, 0, 0, 0]]
assert "name" not in batch
```

Both selected fields stay aligned. Within each input record, they must have
the same length along their packing axes. Across records, each field must
keep its dtype and non-packing shape. Use an `axis` mapping when different
fields store their sequence dimension on different axes.

Only selected fields and requested generated fields survive packing. In this
example, `name` is discarded: one output record can contain parts of several
inputs, so there is no longer a single name to attach to it. If metadata must
remain aligned, represent it as a per-step array and include it in `keys`.

By default, `overflow="split"` carries remaining steps into the next pack, and
the last partial pack is padded with zeros. `padding_values` sets per-field
padding values. `overflow="truncate"` instead discards the remainder of an
input record once the current pack fills.

`mask_key` emits a one-dimensional validity mask per pack. If an input already
has that mask field, its nonzero entries select steps from every packed field
before packing. `position_key` resets positions at input boundaries and
continues them across split fragments. It does not create segment IDs or an
attention mask; any model-specific boundary handling is your responsibility.

`Pack(drop_remainder=True)` drops a partial **pack**. The loader's
`drop_remainder=True` drops a partial **batch of packs**. These are independent.
To pack a plain iterable without a loader, use `packer.pack(records)`.

## Reproducible random transforms

Use `RandomMap`'s supplied generator for augmentation instead of a global RNG.
The loader's `seed` controls sampling and these random transforms:

```python
import numpy as np
from taktiny.data import DataLoader, RandomMap

def add_noise(record, rng):
    noise = rng.normal(0, 0.01, size=record.shape).astype(record.dtype)
    return record + noise

loader = DataLoader(
    [np.zeros(3, dtype=np.float32) for _ in range(4)],
    operations=[RandomMap(add_noise)],
    shuffle=True,
    seed=42,
    batch_size=2,
)
first_pass = list(loader)
second_pass = list(loader)
assert all(np.array_equal(a, b) for a, b in zip(first_pass, second_pass))
```

Each new iterator starts from the beginning. `num_epochs=1` is the default;
set a larger count for multiple passes within one iterator, or `None` for an
unbounded iterator. Keep the source and pipeline unchanged when comparing runs.

## Split a source

`train_validation_split` returns random-access views without copying the
records or changing the source's order. `validation_size` accepts an integer
count or a fraction between zero and one; both resulting splits must be nonempty.

```python
from taktiny.data import DataLoader, train_validation_split

train, validation = train_validation_split(
    list(range(10)), validation_size=0.2, seed=42,
)
assert (len(train), len(validation)) == (8, 2)
train_loader = DataLoader(train, shuffle=True, seed=42, batch_size=4)
validation_loader = DataLoader(validation, batch_size=2)
```

## Workers, shards, and resuming

Start with `worker_count=0` while developing a pipeline. A positive value uses
child workers; your source and transforms must be serializable. Use a guarded
`if __name__ == "__main__":` entry point in multiprocessing scripts.
`worker_buffer_size` controls each worker's prefetch buffer.

For multiple processes, set `shard_index` and `shard_count` to assign separate
input indices to each process. This partitions input records; it does not
create sharded JAX arrays. Shards can have unequal lengths, and workers and
shards batch and pack independently. Account for unequal step counts if your
training loop requires processes to advance together.

If you pass a native Grain `sampler`, it owns sampling, epochs, randomness,
and sharding; the loader's corresponding convenience settings no longer
control those choices.

Loader iterators expose Grain's `get_state()` and `set_state()` methods:

```python
import numpy as np
from taktiny.data import DataLoader

loader = DataLoader(list(range(8)), batch_size=2)
iterator = iter(loader)
next(iterator)
state = iterator.get_state()
expected = next(iterator)

resumed = iter(loader)
resumed.set_state(state)
assert np.array_equal(next(resumed), expected)
```

Restore against the same source and pipeline. This state is separate from a
model checkpoint. **`Pack` does not checkpoint its buffered packing state**, so
do not rely on exact mid-pack resume. Custom Grain iterator operations can
have their own checkpoint limitations.

## Format metadata or text

`ApplyTemplate` formats string leaves inside strings, dictionaries, lists, or
tuples. It preserves other record fields and stores the result under
`return_key` (default: `"template"`). It does not read the resulting path or
tokenize text.

```python
from taktiny.data import ApplyTemplate

make_asset = ApplyTemplate(
    {"path": "{folder}/{name}.png", "label": "{name}"},
    return_key="asset",
)
record = make_asset({"folder": "images", "name": "cat"})
assert record["asset"] == {"path": "images/cat.png", "label": "cat"}
assert record["name"] == "cat"
```

Use `format_fn` to process the complete formatted result. Missing template
fields raise `KeyError`; an existing `return_key` is replaced in the returned
record.

For complete signatures, see the [loader reference](../api/data/loader.md)
and [transform reference](../api/data/transforms.md).
