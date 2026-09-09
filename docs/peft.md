# PEFT

`Takt` transforms an existing model. The implemented PEFT method is LoRA;
`Takt` itself does not run a training loop.

## Apply LoRA

```python
from taktiny import Takt
from taktiny.peft import LoraConfig

model = Takt.apply_peft(
    model,
    LoraConfig(
        target_modules=["q_proj", "v_proj"],
        rank=8,
        alpha=16.0,
    ),
)
```

`target_modules` accepts one regex string or a sequence of regex strings. Each
pattern is searched against the complete module path. Matching linear modules
are replaced by `nn.LoRALinear`. Base parameters are frozen and only `lora_A`
and `lora_B` remain trainable.

## QLoRA-Style Training

```python
from taktiny import Maestro

model = Maestro.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype="int4",
)
model = Takt.apply_peft(model, lora_config)
```

This is weight-only PTQ followed by LoRA training: frozen base weights remain
Qwix INT4, quantized linear operations use the Qwix path, and floating adapter
parameters are updated. It is not full-weight QAT.

## Save and Load an Adapter

A model containing LoRA writes adapter tensors and reconstruction metadata
instead of the full base model:

```python
paths = model.save_pretrained(
    "./adapter",
    max_shard_size="500MB",
)
```

The directory includes `config.json`, `adapter_config.json`, and one or more
`adapter_model*.safetensors` files. Scanned layer state is expanded to numbered
layer keys in the checkpoint.

Load an adapter into a compatible base model:

```python
model = Takt.load_peft(model, "./adapter", local=True)
```

If the base model has no PEFT wrappers, `load_peft` applies them from
`adapter_config.json`. If wrappers already exist, their paths, rank, alpha, and
shapes must match; incompatible structure raises an error instead of silently
loading partial state.

Hub repositories, revisions, private tokens, subfolders, and sharded adapter
indexes are supported by `Takt.load_peft`.

## Merge an Adapter

```python
model = Takt.merge_peft(model, dtype="bfloat16")
```

Merging computes the adapter contribution in float32, writes it into each base
linear module, removes the wrappers, and restores the original trainable flags.
Pass `quant=` to requantize merged weights. The operation mutates and returns
the same model, and raises when no registered mergeable PEFT module exists.

## Upload

`model.push_to_hub(...)` detects whether the model contains LoRA and uploads
the adapter checkpoint in that case. Merge first when the target repository
should contain a complete standalone model.
