# Getting Started

## Installation

Taktiny requires Python 3.11 or newer.

```bash
uv add taktiny
```

Install the JAX build appropriate for the CPU, GPU, or TPU environment. Model
loading also requires enough host memory for checkpoint decoding and enough
device memory for parameters, temporary computations, and KV caches.

## Load and Generate

The following example uses an implemented Qwen2 checkpoint:

```python
from taktiny import Maestro
from transformers import AutoTokenizer

repo = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(repo)
model = Maestro.from_pretrained(
    repo,
    dtype="bfloat16",
    use_list=False,
)

batch = tokenizer("Once upon a time", return_tensors="np")
output_ids = model.generate(
    batch.input_ids,
    attention_mask=batch.attention_mask,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    seed=42,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

`use_list=False` stores decoder parameters in `nn.SeqStack` and executes the
layers through `jax.lax.scan`. `use_list=True` stores separate layer objects
and executes a Python-unrolled layer loop. Both modes preserve sequential
decoder semantics.

## Stream Text

`generate` accepts a Transformers-compatible streamer. Streaming currently
requires batch size one.

```python
from transformers import TextStreamer

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

output_ids = model.generate(
    batch.input_ids,
    attention_mask=batch.attention_mask,
    max_new_tokens=64,
    temperature=0.7,
    streamer=streamer,
    seed=42,
)
```

Use `model.stream_generate(...)` directly when token IDs are needed one decode
step at a time instead of decoded text.

## Inspect Without Loading Weights

`Maestro.eval_shape` downloads or reads only configuration metadata and builds
abstract `jax.ShapeDtypeStruct` leaves. It does not download checkpoint shards
or allocate parameter buffers.

```python
from taktiny import Maestro

abstract_model = Maestro.eval_shape(
    "Qwen/Qwen2.5-0.5B",
    use_list=False,
)
print(abstract_model)
```

Abstract construction does not estimate compilation memory, runtime
temporaries, optimizer state, activations, or KV-cache memory.

## Quantized Loading

Uniform weight-only quantization can be selected with a dtype shortcut:

```python
model = Maestro.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    dtype="int4",
    use_list=False,
)
```

The shortcut stores matching weights as Qwix arrays while model computation
uses BF16 by default. Use `quant=` for selective rules; see
[Models and Checkpoints](models_and_checkpoints.md#quantized-loading).

## Verify the Checkout

The offline suite uses tiny deterministic fixtures:

```bash
uv run pytest
```

It covers checkpoint mappings, model parity, cached decoding, training,
serialization, PEFT, data operations, kernel entry points, and runtime
interfaces without downloading pretrained weights.
