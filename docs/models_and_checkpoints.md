# Models and Checkpoints

`Maestro` is the registry-backed entry point for constructing pretrained
Taktiny models. It reads the single Hugging Face architecture declared in
`config.json`, resolves the corresponding class, and delegates construction or
checkpoint loading to that class.

## Loading a Model

```python
from taktiny import Maestro

model = Maestro.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype="bfloat16",
    use_list=False,
)
```

The main arguments are:

- `repo_or_path`: a Hugging Face repository ID or local checkpoint directory
- `local=True`: read a local directory instead of the Hub
- `dtype`: floating parameter dtype, or a uniform Qwix shortcut such as
  `"int8"`, `"int4"`, or `"nf4"`
- `quant`: a Qwix rule, provider, qtype, or sequence of selective rules
- `mesh`: a `jax.sharding.Mesh` or a mapping of mesh axis names to sizes
- `sharding_rules`: logical parameter axes mapped to mesh axes
- `use_list`: select independent decoder layers (`True`) or a scanned
  `nn.SeqStack` (`False`)

Additional keyword arguments are forwarded to the selected architecture.

## Abstract Inspection

`eval_shape` constructs the selected model under `jax.eval_shape`:

```python
from taktiny import Maestro

abstract_model = Maestro.eval_shape(
    "google/gemma-2-2b",
    use_list=False,
)
print(abstract_model)
```

It reads or downloads `config.json`, but does not download checkpoint shards or
allocate the parameter buffers. The result contains `jax.ShapeDtypeStruct`
leaves and must not be used for a real forward pass.

## Implemented Causal Families

The current `experiment` implementation has working causal model paths for:

| Family | Hugging Face architecture |
| --- | --- |
| Llama | `LlamaForCausalLM` |
| Original Qwen | `QwenForCausalLM`, `QWenLMHeadModel` |
| Qwen2 | `Qwen2ForCausalLM` |
| Qwen3 | `Qwen3ForCausalLM` |
| Gemma | `GemmaForCausalLM` |
| Gemma2 | `Gemma2ForCausalLM` |
| Gemma3 text model | `Gemma3ForCausalLM` |

Gemma2 and Gemma3 alternate local and full attention. In scanned mode their
per-layer differences are represented as stacked array state so every layer
has the same PyTree structure.

Several additional names are present in the internal repertoire while their
architecture-specific implementations are still placeholders. In particular,
registration of DeepSeek, GPT-OSS, Llama 4, Gemma 4, Qwen MoE/Next/3.5, or a
diffusion class should not be treated as a support claim.

Use the registry for discovery, while keeping that distinction in mind:

```python
from taktiny import Maestro

names = Maestro.available()
classes = Maestro.list()
is_registered = Maestro.supported("LlamaForCausalLM")
```

## Quantized Loading

A quantized `dtype` is a uniform weight-storage shortcut:

```python
model = Maestro.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype="int4",
)
```

Matching weights are quantized as checkpoint tensors are loaded instead of
first materializing the complete dense model. Qwix `QArray` weights use
`qwix.dot_general` in `nn.Linear`; default activation and computation dtype is
BF16 for this shortcut.

For selective quantization, pass `quant=`. Explicit rules take precedence when
combined with a quantized `dtype`; the dtype shortcut is the fallback for
unmatched eligible modules. Embeddings may also be quantized when a rule
selects them, but embedding lookup dequantizes only selected rows.

Quantization reduces persistent parameter storage. It does not guarantee a
proportional reduction in activation, attention, compilation, optimizer, or
temporary dequantization memory.

## Saving and Reloading

```python
saved_paths = model.save_pretrained(
    "./checkpoint",
    max_shard_size="2GB",
)

model.load_pretrained("./checkpoint")
```

`save_pretrained` always writes `config.json`. A model without LoRA writes all
parameters; a model containing `LoRALinear` modules writes adapter tensors and
`adapter_config.json`. `nn.SeqStack` parameters are expanded to conventional
`model.layers.0`, `model.layers.1`, and subsequent keys. Shards follow the
Transformers naming convention and an index is written when required.

`load_pretrained` restores a full Taktiny-native local checkpoint into an
already constructed compatible model. Use `Takt.load_peft` for adapters.

Upload either form with:

```python
url = model.push_to_hub(
    "username/model-name",
    max_shard_size="2GB",
)
```

## Adding an Architecture

Architecture registration is an internal extension point. A usable
integration needs more than a registry entry: it needs an architecture class,
decoder layer, configuration normalization, checkpoint key mapping, tied
weight behavior, and parity tests.

```python
from taktiny.maestro._livret import repertoire

repertoire.register("MyModelForCausalLM", MyModel)
```

Registering multiple architecture names to one implementation is valid; the
registry is many-to-one.
