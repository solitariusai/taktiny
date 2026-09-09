# Generation

## Native Causal Generation

Implemented causal models share `TransformerCausalLM.generate`:

```python
output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=128,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.05,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    seed=42,
)
```

The returned array includes the original padded `input_ids` followed by newly
generated token IDs. `attention_mask` must have the same two-dimensional shape
as `input_ids`; it supplies the per-sequence prompt lengths for padded batches.
Each row keeps its own cached position and completion state.

Generation performs one prompt prefill and then cached one-token decoding. It
supports temperature sampling, top-k, top-p, repetition penalty, multiple EOS
IDs, and deterministic integer seeds.

## Streaming

Pass a Transformers-compatible streamer to the same method:

```python
from transformers import TextStreamer

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=128,
    streamer=streamer,
    seed=42,
)
```

`generate` calls `streamer.put(...)` with the prompt and each generated token,
then calls `streamer.end()`. Transformers `TextStreamer` supports batch size
one. `TextIteratorStreamer` may be consumed from another thread in the usual
Transformers pattern.

For token-level iteration without a text streamer:

```python
for token_ids in model.stream_generate(
    input_ids,
    max_new_tokens=128,
    attention_mask=attention_mask,
    seed=42,
):
    consume(token_ids)  # shape: [batch, 1]
```

## Forward Context

The lower-level causal call returns `(logits, context)`:

```python
logits, context = model(
    input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
)
```

`TransformerContext` carries key/value caches, cached positions, and causal
mode. Passing explicit two-dimensional `position_ids` also supports packed
training examples. Most inference callers should use `generate`, which creates
and updates the context internally.
