# Data Pipelines

The trainer consumes an iterable of model-ready batches. Taktiny does not hide
text processing inside `DatasetConfig`; reusable data operations live in
`taktiny.data_utils` and compose with Grain.

## Build a Grain Dataloader

`DatasetUtils.from_datasets` accepts a finite random-access source, such as a
Hugging Face `Dataset`, and applies operations exactly in the supplied order.

```python
from datasets import load_dataset
import grain.python as grain
import numpy as np

from taktiny.data_utils import (
    ApplyTemplate,
    BatchMap,
    CausalLMBatch,
    DatasetUtils,
    PackSequences,
)

dataset = load_dataset("openai/gsm8k", "main", split="train")

template = [
    {"role": "user", "content": "{question}"},
    {"role": "assistant", "content": "{answer}"},
]


def render(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False)


def tokenize_batch(rows):
    encoded = tokenizer(
        [row["template"] for row in rows],
        add_special_tokens=False,
    )
    return [
        {
            "input_ids": np.asarray(token_ids, dtype=np.int32),
            "labels": np.asarray(token_ids, dtype=np.int32),
        }
        for token_ids in encoded["input_ids"]
    ]


dataloader = DatasetUtils.from_datasets(
    dataset,
    operations=[
        ApplyTemplate(template, render),
        BatchMap(tokenize_batch, batch_size=256),
        PackSequences(2048, overflow="split", drop_remainder=True),
        grain.Batch(batch_size=4, drop_remainder=True),
        CausalLMBatch(),
    ],
    shuffle=True,
    seed=42,
    num_epochs=1,
)
```

`BatchMap` buffers rows for one vectorized tokenizer call, then emits one
tokenized record per input row. This avoids invoking a tokenizer separately for
every record while preserving a row-oriented stream for packing.

## Packing Order

`PackSequences` must run after tokenization and before `grain.Batch`. Its input
fields must be one-dimensional and equal in length. It emits fixed-length
records containing reset `position_ids`; the causal model uses those resets to
keep packed examples independent without storing a quadratic block-diagonal
mask in the dataloader.

`CausalLMBatch` runs after batching. It masks labels at reset positions, while
padding labels produced by `PackSequences` already use `-100`, so next-token
loss does not cross example or padding boundaries.

The overflow modes are:

- `split`: divide a source sequence longer than the target length into chunks
- `truncate`: retain only the first target-length chunk

`drop_remainder` controls the final incomplete packed record. It does not
choose the overflow policy.

## Generic Operations

The data namespace exports:

- `Map`: adapt a row callable to a Grain map transform
- `BatchMap`: batch rows, call a vectorized function, and unbatch its result
- `ApplyTemplate`: recursively format strings in mappings, lists, and tuples
- `PackSequences`: greedily construct fixed-length token records
- `CausalLMBatch`: prepare packed labels after batching

For image, audio, or other tasks, use `Map`, native Grain operations, or custom
transforms and emit the batch structure expected by the task loss. Text
operations are not required by `Trainer`.

## Sampling, Sharding, and Workers

```python
dataloader = DatasetUtils.from_datasets(
    dataset,
    operations=operations,
    shuffle=True,
    seed=42,
    num_epochs=1,
    shard_index=process_index,
    shard_count=process_count,
    worker_count=2,
    worker_buffer_size=2,
)
```

When `sampler` is omitted, Taktiny builds a Grain `IndexSampler` from the source
length and these sampling options. Passing a custom sampler transfers sampling
and source-sharding responsibility to that sampler.

Keep `num_epochs=1` when `TrainingConfig.epochs` controls training epochs.
Larger values make one iteration of this dataloader traverse the source more
than once and therefore multiply the trainer's outer epoch count.

`worker_count=0` processes records in the caller. A positive count uses Grain
workers; `worker_buffer_size` controls buffered outputs per worker. More workers
help only when preprocessing is the bottleneck and may increase host memory.

The dataloader length is available only when Grain and its operations can
derive it. Packing changes the number of output records based on tokenized
lengths, so an exact progress total may be unknown without preprocessing the
entire source first.

## Trainer Boundary

Wrap the resulting loader without duplicating preprocessing settings:

```python
from taktiny import DatasetConfig

dataset_config = DatasetConfig(
    train_dataloader=dataloader,
    validation_dataloader=validation_dataloader,
    batch_sharding=batch_sharding,
    prefetch_size=2,
)
```

Stateful iterators are not prefetched, preserving their checkpoint cursor.
Exact resume uses `get_state` and `set_state` when available; otherwise the
trainer relies on deterministic ordering and replays/skips consumed batches.
