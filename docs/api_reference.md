# API Reference

This page is a compact map of the current `experiment` API. Architecture
internals under `taktiny.cosettes` and raw kernel modules remain lower-level
extension points.

## Top Level

```python
from taktiny import (
    DatasetConfig,
    Maestro,
    ModelConfig,
    Takt,
    TensorBoardCallback,
    Trainer,
    TrainerCallback,
    TrainingConfig,
    WandbCallback,
    kernels,
    layers,
    nn,
    peft,
    tt,
)
```

| Symbol | Purpose |
| --- | --- |
| `Maestro` | Resolve a registered architecture, inspect it abstractly, or load a checkpoint |
| `ModelConfig` | Attribute-based representation of Hugging Face configuration JSON |
| `Takt` | Transform an existing model with PEFT, load adapters, or merge adapters |
| `Trainer` | Generic Optax/JAX training, evaluation, reporting, and checkpoint lifecycle |
| `TrainingConfig` | Optimizer, schedule, numeric, evaluation, logging, and save settings |
| `DatasetConfig` | Existing train/validation dataloaders, batch sharding, and prefetch settings |
| `nn` | Module and primitive neural-network namespace |
| `layers` | Transformer and higher-level layer namespace |
| `kernels` | Experimental low-level kernel namespace |
| `tt` | Module-aware `vmap` and `scan` transformations |
| `peft` | PEFT configuration compatibility module |

Import `LoraConfig` explicitly from `taktiny.peft`:

```python
from taktiny.peft import LoraConfig, PeftConfig
```

## Maestro

```python
Maestro.available()
Maestro.list()
Maestro.supported(architecture_name)
Maestro.from_pretrained(
    repo_or_path,
    mesh=None,
    sharding_rules=None,
    local=False,
    dtype=None,
    quant=None,
    **kwargs,
)
Maestro.eval_shape(
    repo_or_path,
    mesh=None,
    sharding_rules=None,
    local=False,
    **kwargs,
)
```

`supported` currently means registered. See
[Models and Checkpoints](models_and_checkpoints.md#implemented-causal-families)
for the smaller implemented and tested family list.

## Model Lifecycle

`PretrainedModel` supplies:

```python
model.save_pretrained(path, max_shard_size="5GB")
model.load_pretrained(path)
model.push_to_hub(
    repo_id,
    commit_message=None,
    commit_description=None,
    private=None,
    token=None,
    revision=None,
    create_pr=False,
    max_shard_size="5GB",
)
```

`TransformerCausalLM` supplies:

```python
logits, context = model(
    input_ids,
    attention_mask=None,
    position_ids=None,
    ctx=None,
    logits_to_keep=0,
)

output_ids = model.generate(
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    seed=42,
    attention_mask=None,
    repetition_penalty=1.0,
    eos_token_id=None,
    pad_token_id=None,
    streamer=None,
)

iterator = model.stream_generate(
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    seed=42,
    attention_mask=None,
    repetition_penalty=1.0,
    eos_token_id=None,
    pad_token_id=None,
)

model.enable_remat()
```

Core implementation classes are available from `taktiny.cosettes._common`:
`TransformerContext`, `TransformerDecoderLayer`, `TransformerModel`,
`TransformerCausalLM`, `TransformerConditionalGeneration`, `DiffusionLM`, and
`DiffusionIM`. `PretrainedModel` lives in `taktiny.cosettes._base`.

## PEFT

```python
LoraConfig(
    target_modules,
    rank=8,
    alpha=8.0,
    rngs=None,
)

Takt.apply_peft(model, config)
Takt.load_peft(
    model,
    path_or_repo,
    local=None,
    token=None,
    revision=None,
    subfolder=None,
    rngs=None,
)
Takt.merge_peft(model, dtype=None, quant=None)
```

`target_modules` contains regular-expression patterns searched against full
module paths.

## Trainer

```python
Trainer(
    model,
    training_config,
    dataset_config,
    loss_fn=loss_fn,
    callbacks=None,
    compute_metrics=None,
)

trainer.train(resume_from_checkpoint=None)
trainer.evaluate()
```

`TrainingConfig` fields:

- execution: `epochs`, `max_steps`, `seed`, `jit_compile`, `donate_batch`
- optimization: `learning_rate`, `schedule`, `optimizer`, `weight_decay`
- gradients: `gradient_accumulation_steps`, `max_grad_norm`,
  `skip_non_finite`
- FP16 scaling: `loss_scale`, `initial_loss_scale`,
  `loss_scale_growth_interval`
- logging: `log_interval`
- saving: `output_dir`, `save_steps`, `save_total_limit`, `save_at_end`,
  `save_optimizer_state`, `save_async`, `max_shard_size`
- evaluation: `eval_strategy`, `eval_steps`, `metric_for_best_model`,
  `greater_is_better`, `load_best_model_at_end`

`DatasetConfig` fields are `train_dataloader`, `validation_dataloader`,
`batch_sharding`, `shuffle`, `seed`, and `prefetch_size`.

Loss helpers from `taktiny.trainer`:

```python
cross_entropy_loss(
    logits,
    labels,
    mask=None,
    ignore_index=-100,
    reduction="mean",
)
causal_lm_loss(model, batch, ignore_index=-100)
```

Callback classes are `TrainerCallback`, `TensorBoardCallback`, and
`WandbCallback`. Lifecycle hooks are `on_train_begin`, `on_step_end`, `on_log`,
`on_save`, `on_evaluate`, and `on_train_end`.

## Data Utilities

Import from `taktiny.data_utils`:

```python
from taktiny.data_utils import (
    ApplyTemplate,
    BatchMap,
    CausalLMBatch,
    DatasetUtils,
    Map,
    PackSequences,
)
```

```python
DatasetUtils.from_datasets(
    source,
    operations=(),
    sampler=None,
    shuffle=False,
    seed=0,
    num_epochs=None,
    shard_index=0,
    shard_count=1,
    worker_count=0,
    worker_buffer_size=1,
)
```

`Map` adapts a row callable. `BatchMap` buffers rows for one batched callable
and unbatches its result. `ApplyTemplate` formats nested templates.
`PackSequences` packs one-dimensional sequence fields before batching.
`CausalLMBatch` masks labels at reset positions after batching.

## Neural-Network Primitives

Common symbols from `taktiny.nn` include:

- state: `Module`, `Parameter`, `Rngs`
- dense operations: `Linear`, `Embedding`
- normalization: `RMSNorm`, `LayerNorm`, `GroupNorm`
- PEFT wrapper: `LoRALinear`
- containers: `List`, `Sequential`, `SeqStack`, `Stack`

`SeqStack` invokes a supplied scan body. `Stack` directly vmaps the stored
module call and accepts `in_axes` and `out_axes`.

## Layers and Kernel Entry Points

Common symbols from `taktiny.layers` include `Attention`, `GateMLP`,
`FusedGateMLP`, `MoeFFN`, `RotaryEmbedding`, and architecture-specific norms.

```python
Attention.apply(query, key, value, kernel="dot_product", **kwargs)
FusedGateMLP.apply(lhs, rhs, group_sizes, kernel="gmm", **kwargs)
MoeFFN.apply(lhs, rhs, group_sizes, kernel="gmm", **kwargs)
Embedding.apply(operand, indices, kernel="gather_reduce", **kwargs)
```

Attention kernel names are `dot_product`, `flash`, `ragged`, `splash`, and
`ring`. Refer to [Kernels](kernels.md) before choosing a non-JAX
path.

## Experimental Runtime and RL APIs

The optional vLLM wrapper is under `taktiny.ensembles.vllm`:

```python
from taktiny.ensembles.vllm import VLLM

runtime = VLLM(model, auto_start=True, **engine_options)
runtime.generate(...)
runtime.sync()
runtime.close()
```

The bundled local engine currently targets GPU. Native Taktiny execution on
vLLM TPU is deliberately `NotImplemented`; supply a custom `VLLMEngine` to use
a future or external TPU integration.

`taktiny.trainer` also exports `PolicyRuntime`, `Rollout`, and the abstract
`RLBaseTrainer`. The base owns policy versioning, synchronization boundaries,
and generic Trainer integration. It does not implement GRPO, DPO, PPO, reward
calculation, or rollout-to-batch conversion; concrete subclasses must provide
those algorithm-specific hooks.

See [Experimental APIs](experimental.md) for the precise support boundaries.
