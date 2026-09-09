# Training

`Trainer` is a general Optax/JAX training loop. It receives a model, execution
configuration, existing dataloaders, and a loss function; it does not assume a
language-model dataset format.

## Basic Causal-LM Training

```python
from taktiny import DatasetConfig, Trainer, TrainingConfig
from taktiny.trainer import causal_lm_loss

trainer = Trainer(
    model,
    TrainingConfig(
        epochs=1,
        max_steps=100,
        learning_rate=2e-4,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        output_dir="./outputs",
        save_steps=50,
        save_total_limit=2,
        save_optimizer_state=True,
    ),
    DatasetConfig(
        train_dataloader=dataloader,
        prefetch_size=2,
    ),
    loss_fn=causal_lm_loss,
)

trainer.train()
```

`causal_lm_loss` expects `input_ids` and `labels`, plus optional
`attention_mask` and `position_ids`. Supply a custom `loss_fn(model, batch)` for
other tasks. A loss may also accept `rng=` for trainer-managed randomness.

## Optimizer and Schedule

The default optimizer is Optax AdamW using `learning_rate` and `weight_decay`.
Pass an Optax schedule through `schedule=`:

```python
import optax

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=2e-4,
    warmup_steps=100,
    decay_steps=1000,
    end_value=2e-5,
)

config = TrainingConfig(schedule=schedule)
```

Alternatively, pass a complete Optax gradient transformation through
`optimizer=`. A custom optimizer owns its learning rate. Trainer LR logging is
available only when a separate schedule is also supplied.

## Gradient and Numeric Controls

`TrainingConfig` supports:

- `gradient_accumulation_steps`
- `max_grad_norm` for global clipping
- `skip_non_finite` for NaN/Inf update rejection
- `loss_scale` as a positive number or `"dynamic"`
- `initial_loss_scale` and `loss_scale_growth_interval`

Dynamic scaling makes FP16 training more robust but cannot guarantee numerical
stability for every architecture and optimizer.

Supported causal models configure decoder rematerialization directly:

```python
model.enable_remat()
```

Call it before training. Rematerialization recomputes decoder layers during the
backward pass to reduce retained activation memory. It does not reduce model or
optimizer-state storage.

## Multi-Device Batches

Load or place model parameters with a JAX mesh, then give
`DatasetConfig.batch_sharding` either one `Sharding` for every batch leaf or a
matching sharding PyTree.

Mesh axis names such as `fsdp` and `tp` are descriptive only. Actual behavior
comes from the parameter logical-axis rules and batch partition specs. A model
mesh that shards only parameters is not automatically batch sharding, and vice
versa.

## Evaluation and Best Checkpoint

```python
config = TrainingConfig(
    output_dir="./outputs",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

dataset_config = DatasetConfig(
    train_dataloader=train_loader,
    validation_dataloader=validation_loader,
)
```

`eval_strategy` accepts `"no"`, `"steps"`, or `"epoch"`. Supply
`compute_metrics` to add task metrics. `greater_is_better` may override the
direction inferred for best-model selection.

## Callbacks and Reporting

```python
from taktiny import TensorBoardCallback, WandbCallback

callbacks = [
    TensorBoardCallback("./outputs/runs"),
    WandbCallback(project="taktiny"),
]
```

Custom callbacks may implement `on_train_begin`, `on_step_end`, `on_log`,
`on_save`, `on_evaluate`, and `on_train_end`. TensorBoard and W&B callbacks
report both training and evaluation logs.

## Checkpoint and Resume

```python
trainer.train(resume_from_checkpoint="latest")
```

Checkpoints use atomic directory publication and optionally asynchronous saves.
They include model or adapter weights, configuration, trainer state and log
history, RNG state, dataloader position when available, and Orbax optimizer
state when enabled. Multi-host saves coordinate processes before publishing a
checkpoint.

Resume restores the global step, epoch position, log history, best-checkpoint
metadata, loss-scaling state, RNG, model/adapters, optimizer, and dataloader
cursor. `save_total_limit` prunes older completed checkpoints.
