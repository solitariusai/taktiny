# Taktiny

Taktiny is an experimental neural-network library built directly on JAX. It
provides object-oriented modules that remain valid JAX PyTrees, a small set of
transformer building blocks, `Maestro` for loading compatible Hugging Face
checkpoints, and `Takt` for transforming existing model instances.

The project is under active development. APIs, checkpoint mappings, and model
coverage may change between revisions.

## Highlights

- Stateful `nn.Module` and `nn.Parameter` objects registered as JAX PyTrees
- Native support for `jax.jit`, `jax.value_and_grad`, and Optax
- Safetensors checkpoint loading and saving
- Abstract model construction through `jax.eval_shape`
- Reusable transformer decoder, model, and causal-LM components
- KV-cached autoregressive generation
- Logical parameter axes and optional JAX mesh sharding
- Qwix weight-only PTQ and registry-backed PEFT transformations

## Requirements

- Python 3.12 or newer
- JAX 0.10.2 or newer

## Testing

The offline test suite uses tiny deterministic checkpoints and does not
download pretrained weights:

```bash
uv run pytest
```

It checks Hugging Face logit parity for Llama, Qwen2, Gemma, Gemma2, and
Gemma3, together with full-versus-cached decoding, checkpoint-key coverage,
sliding-window boundaries, BF16 loading, and Qwix INT8/INT4 loading. Original Qwen
uses legacy Hugging Face remote code that is incompatible with Transformers
5.13; its offline tests cover checkpoint mapping and cached decoding instead.


## Quick Start

The following example loads a Qwen2 checkpoint and generates text:

```python
from taktiny import Maestro
from transformers import AutoTokenizer

repo = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(repo)
model = Maestro.from_pretrained(repo)

input_ids = tokenizer.encode(
    "Once upon a time",
    return_tensors="np",
)

output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

Model loading and generation materialize checkpoint parameters and KV caches.
Choose a checkpoint and dtype that fit the available device and host memory.

## Quantized Loading

Checkpoint weights can be quantized as they are loaded. Taktiny stores matching
linear weights and embedding tables as Qwix `QArray` values:

```python
import qwix

from taktiny import Maestro

rules = [
    qwix.QuantizationRule(
        module_path=r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        weight_qtype="int8",
    ),
]

model = Maestro.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    dtype="bfloat16",
    quant=rules,
)
```

For a uniform weight-only format, `dtype="int8"` and `dtype="int4"` remain
shortcuts. The quantized dtype describes linear-weight and embedding-table
storage. Activations, quantization scales, and operation outputs remain BF16
by default, while Qwix selects the available dot implementation for the
quantized operands.

The shortcut can be combined with explicit rules. Explicit rules are checked
first, and the dtype is used as the fallback for unmatched modules:

```python
model = Maestro.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    dtype="int4",
    quant=[
        qwix.QuantizationRule(
            module_path=r".*self_attn\.q_proj",
            weight_qtype="int8",
        ),
    ],
)
```

## Implemented Architectures

| Hugging Face architecture | Taktiny class | Status |
| --- | --- | --- |
| `LlamaForCausalLM` | `Llama` | Implemented |
| `QWenLMHeadModel` | `Qwen` | Implemented |
| `Qwen2ForCausalLM` | `Qwen2` | Implemented |
| `GemmaForCausalLM` | `Gemma` | Implemented |
| `Gemma3ForCausalLM` | `Gemma3` | Implemented |

Other architecture names may appear in the internal repertoire as development
placeholders. Registration alone does not mean that checkpoint loading or
inference is implemented.

You can inspect the architecture registry with:

```python
from taktiny import Maestro

print(Maestro.available())
```

## Inspecting Shapes

`Maestro.eval_shape` constructs an abstract model from repository
configuration without allocating parameter buffers or downloading checkpoint
weights:

```python
from taktiny import Maestro

abstract_model = Maestro.eval_shape("Qwen/Qwen2.5-0.5B")
print(abstract_model)
```

This is useful for inspecting parameter counts, shapes, and dtypes before
loading a checkpoint. It does not estimate temporary compilation memory or KV
cache usage.

## Applying PEFT

`Takt` applies registered transformations to an existing model. PEFT methods
are selected by configuration type, allowing additional adapter families to
share one public entry point:

```python
from taktiny import LoraConfig, Takt

model = Takt.apply_peft(
    model,
    LoraConfig(
        target_modules=["q_proj", "v_proj"],
        rank=16,
        alpha=32,
    ),
)
```

PEFT transformations currently mutate the supplied model and return the same
instance. LoRA is implemented; additional configuration types can register
their own implementations with `Takt.register_peft`.

## Building Modules

Taktiny modules keep parameters directly on the object while participating in
JAX transformations:

```python
import jax

from taktiny import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        rngs: nn.Rngs,
    ):
        self.input = nn.Linear(
            in_features,
            hidden_features,
            rngs=rngs,
        )
        self.output = nn.Linear(
            hidden_features,
            out_features,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.output(jax.nn.silu(self.input(x)))


model = MLP(64, 128, 10, rngs=nn.Rngs(42))
jitted_model = jax.jit(model)
```

Parameters can be inspected or restored through flat and nested state
dictionaries:

```python
flat_state = model.flat_state_dict()
state = model.state_dict()

model.load_state_dict(state)
```

Models derived from `PretrainedModel` can also write Safetensors checkpoints:

```python
model.save_pretrained("./checkpoint")
```

## Defining A Transformer Family

`TransformerDecoderLayer` creates modules in the order supplied by the family
implementation. Normalization modules transform the active hidden state;
attention and MLP modules are applied as residual branches.

```python
from taktiny import nn
from taktiny.cosettes.common import (
    TransformerCausalLM,
    TransformerDecoderLayer,
)
from taktiny.layers import Attention, GateMLP


class ExampleDecoderLayer(TransformerDecoderLayer):
    def __init__(self, config, rngs: nn.Rngs):
        super().__init__(
            config,
            rngs=rngs,
            input_layernorm=nn.RMSNorm,
            self_attn=Attention,
            post_attention_layernorm=nn.RMSNorm,
            mlp=GateMLP,
        )


class ExampleForCausalLM(TransformerCausalLM):
    def __init__(
        self,
        config,
        rngs: nn.Rngs = None,
        mesh=None,
        sharding_rules=None,
    ):
        if rngs is None:
            rngs = nn.Rngs(42)

        super().__init__(
            config,
            rngs=rngs,
            decoder=ExampleDecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
        )
```

Checkpoint-facing attribute names such as `self_attn`, `input_layernorm`, and
`mlp` should match the source checkpoint wherever possible. This minimizes
weight-mapping rules.

## Training

The experimental `Trainer` accepts native Taktiny models and uses Optax for
updates:

```python
import optax

from taktiny import DatasetConfig, Trainer, TrainingConfig

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    training_config=TrainingConfig(
        epochs=1,
        max_steps=1_000,
        optimizer=optax.adamw(3e-4),
        log_interval=10,
        jit_compile=True,
    ),
    dataset_config=DatasetConfig(dataloader=train_batches),
)

trainer.train()
```

When no dataloader is supplied, `DatasetConfig` can load a Hugging Face
dataset. `process_fn` runs once on the loaded dataset and may return either
train data or a `(train, validation)` pair. Non-streaming data is exposed
through resumable Grain iterators; streaming data remains an HF iterable.

```python
dataset_config = DatasetConfig(
    repo_id='open-r1/OpenThoughts-114k-math',
    process_fn=prepare_dataset,
    streaming=False,
)
```

For gated repositories, `HF_TOKEN` takes precedence over `hf_token`. Repository
loading, token resolution, preprocessing, and Grain wrapping are all skipped
when `dataloader` is supplied explicitly.

### Supervised Fine-Tuning

`SFTTrainer` specializes the same training loop with causal language-model
loss, tokenization, dynamic padding, and optional sequence packing:

```python
from transformers import AutoTokenizer

from taktiny import SFTDatasetConfig, SFTTrainer, SFTTrainingConfig

tokenizer = AutoTokenizer.from_pretrained(model_repo)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model,
    training_config=SFTTrainingConfig(
        epochs=1,
        learning_rate=2e-4,
        assistant_only_loss=True,
        jit_compile=True,
    ),
    dataset_config=SFTDatasetConfig(
        repo_id='open-r1/OpenThoughts-114k-math',
        tokenizer=tokenizer,
        process_fn=prepare_open_thoughts,
        batch_size=8,
        max_length=1024,
        padding='longest',
        packing=False,
    ),
)
trainer.train()
```

Supported records contain one of:

- `input_ids`, with optional `labels` and `attention_mask`
- `text`
- `messages`
- `prompt` and `completion`

Prompt-completion records use completion-only loss by default. Conversational
records can use `assistant_only_loss=True`; explicit pretokenized `labels`
always take precedence. Set `packing=True` to fill fixed-length sequences.
Packed examples receive block-diagonal attention masks, so examples in the same
sequence cannot attend to each other.

`process_fn` runs once on a dataset loaded through `repo_id`. Use
`formatting_fn` for per-record conversion. Set `skip_prepare_dataset=True` only
when the supplied dataloader already yields complete SFT batches containing
`input_ids`, `attention_mask`, and `labels`.

Rematerialization is configured on models that understand their own layer
boundaries:

```python
model.enable_remat()
trainer.train()
```

Scheduled checkpoints can preserve exact stochastic and dataloader progress:

```python
training_config = TrainingConfig(
    seed=42,
    output_dir='checkpoints/run',
    save_steps=100,
    save_async=True,
)
```

`save_async=True` captures a stable host snapshot and overlaps its
serialization with later training. Checkpoints are first written to a sibling
temporary directory and become visible only after an atomic rename. Single-host
checkpoints use portable Safetensors, including lossless Qwix component
metadata. Multi-host jobs collectively save distributed model and optimizer
state with Orbax, while RNG and stateful-dataloader cursors are stored per host.

Loss functions that use dropout or another stochastic operation may opt into
the Trainer RNG without changing deterministic loss functions:

```python
def loss_fn(model, batch, *, rng):
    logits = model(batch['input_ids'], rng=rng)
    return cross_entropy(logits, batch['labels'])
```

Callbacks and validation metrics can be added without changing the training
step:

```python
from taktiny import TensorBoardCallback, WandbCallback

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    training_config=training_config,
    dataset_config=dataset_config,
    compute_metrics=lambda params, batch: {
        'accuracy': accuracy(params, batch),
    },
    callbacks=[
        TensorBoardCallback(log_dir='runs/experiment'),
        WandbCallback(project='taktiny'),
    ],
)
```

Callback objects may implement any of `on_step_end`, `on_log`, `on_save`, or
`on_evaluate`. TensorBoard and W&B dependencies remain optional and are
available through the `tensorboard`, `wandb`, or `reporting` package extras.

Trainer consumes the supplied dataloader directly. At each epoch it uses the
first available `set_epoch` hook on the dataloader, its sampler, or its
dataset. Iterators exposing both `get_state` and `set_state` are checkpointed
as bytes or JSON and restored without replaying consumed batches. Other
iterators retain the epoch-and-batch-position resume fallback.

The trainer currently uses heuristic parameter freezing for large and
quantized parameters. Review the trainable parameter set before using it for a
real training run.

## Project Layout

```text
src/taktiny/
├── nn/                 Object-oriented JAX modules and parameters
├── layers/             Attention, feed-forward, positional, and vision layers
├── cosettes/           Reusable model implementations
├── maestro/            Architecture registry and checkpoint orchestration
├── takt/               Existing-model transformations and PEFT methods
├── trainer/            Experimental training utilities
└── utils/              Sharding, quantization, typing, and weight mapping
```

## License

Taktiny is distributed under the Apache License 2.0. See
[`LICENSE.md`](LICENSE.md).
