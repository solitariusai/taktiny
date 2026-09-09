# Trainer & Callbacks

The `taktiny.trainer` package provides a general-purpose training loop for JAX models. It integrates Optax optimization, gradient accumulation, mixed-precision loss scaling, evaluation, exponential moving averages (EMA), device placement, callbacks, and Orbax checkpointing.

The trainer does not own the data pipeline. You provide:

* a Taktiny model,
* a loss function,
* an iterable of training batches,
* and the desired training configuration.

This keeps loading, preprocessing, batching, and sampling independent from the training loop.

## Basic Training

A minimal training setup defines a model and loss function, then constructs a `Trainer`.

```python
import jax.numpy as jnp
import optax

from taktiny import nn
from taktiny.trainer import DatasetConfig, Trainer, TrainingConfig


class MLP(nn.Module):
    def __init__(self, *, rngs: nn.Rngs):
        self.dense = nn.Linear(32, 4, rngs=rngs)

    def __call__(self, x):
        return self.dense(x)


def loss_fn(model, batch):
    logits = model(batch["image"])
    return jnp.mean((logits - batch["label"]) ** 2)


model = MLP(rngs=nn.Rngs(0))

train_data = [
    {
        "image": jnp.ones(32),
        "label": jnp.zeros(4),
    }
] * 10

learning_rate = 3e-4

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    training_config=TrainingConfig(
        max_steps=1000,
        optimizer=optax.adamw(learning_rate),
        schedule=optax.constant_schedule(learning_rate),
        output_dir="/tmp/checkpoints",
        log_interval=50,
        save_steps=500,
    ),
    dataset_config=DatasetConfig(
        train_dataloader=train_data,
        prefetch_size=2,
    ),
)

trainer.train()
```

## Loss Functions

The trainer calls the provided loss function with the current model and batch:

```python
def loss_fn(model, batch):
    logits = model(batch["image"])
    return jnp.mean((logits - batch["label"]) ** 2)
```

Loss functions may also accept an `rng` keyword argument. When present, the trainer supplies a per-step PRNG key:

```python
def loss_fn(model, batch, *, rng):
    ...
```

This is useful when the training step contains stochastic operations such as dropout or random augmentation.

The returned value should be a scalar JAX array representing the loss to minimize.

## Training Configuration

`TrainingConfig` defines optimization, evaluation, numerical-stability, and checkpointing behavior.

| Category          | Options                                                                             |
| ----------------- | ----------------------------------------------------------------------------------- |
| **Optimization**  | `optimizer`, `learning_rate`, `schedule`, `weight_decay`                            |
| **Gradients**     | `gradient_accumulation_steps`, `max_grad_norm`, `compute_grad_norm`                 |
| **Precision**     | `loss_scale`, `initial_loss_scale`, `loss_scale_growth_interval`, `skip_non_finite` |
| **Averaging**     | `ema_decay`                                                                         |
| **Evaluation**    | `eval_strategy`, `eval_steps`, `metric_for_best_model`, `greater_is_better`         |
| **Checkpointing** | `output_dir`, `save_steps`, `save_total_limit`, `save_at_end`, `save_async`         |
| **Execution**     | `max_steps`, `jit_compile`, `seed`, `log_interval`                                  |

### Gradient Accumulation

Set `gradient_accumulation_steps` to accumulate gradients across multiple batches before applying an optimizer update:

```python
TrainingConfig(
    gradient_accumulation_steps=4,
)
```

With an accumulation factor of `4`, the trainer evaluates four micro-batches before performing one optimizer step.

This is useful when the desired effective batch size does not fit in device memory.

### Gradient Clipping

Use `max_grad_norm` to clip gradients by their global norm before applying optimizer updates:

```python
TrainingConfig(
    max_grad_norm=1.0,
)
```

Gradient norm computation can be disabled separately with `compute_grad_norm=False`.

### Loss Scaling

Loss scaling can improve numerical stability when training with low-precision arithmetic.

For dynamic loss scaling:

```python
TrainingConfig(
    loss_scale="dynamic",
)
```

The trainer scales the loss before differentiation and unscales the resulting gradients before optimization.

When non-finite gradients are detected, the optimizer update can be skipped and the dynamic loss scale adjusted automatically.

A fixed scale may also be provided:

```python
TrainingConfig(
    loss_scale=32768.0,
)
```

Additional dynamic-scaling behavior can be controlled with:

* `initial_loss_scale`
* `loss_scale_growth_interval`
* `skip_non_finite`

### Exponential Moving Average

Set `ema_decay` to maintain an exponential moving average of model parameters:

```python
TrainingConfig(
    ema_decay=0.999,
)
```

EMA weights provide a smoothed version of the model parameters and may be used during evaluation or checkpoint selection.

The decay value must be between `0` and `1`.

## Evaluation

Evaluation can be disabled or scheduled periodically.

For step-based evaluation:

```python
TrainingConfig(
    eval_strategy="steps",
    eval_steps=100,
)
```

Supported evaluation strategies are:

* `"no"` — disable automatic evaluation,
* `"steps"` — evaluate every `eval_steps`,
* `"epoch"` — evaluate at epoch boundaries when supported by the data source.

A validation iterable must be supplied through `DatasetConfig` when evaluation is enabled:

```python
DatasetConfig(
    train_dataloader=train_loader,
    validation_dataloader=validation_loader,
)
```

The metric used to determine the best checkpoint can be configured with:

```python
TrainingConfig(
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```

Set `load_best_model_at_end=True` to restore the best checkpoint after training.

## Dataset Configuration

`DatasetConfig` describes the batch iterables consumed by the trainer.

```python
dataset_config = DatasetConfig(
    train_dataloader=train_loader,
    validation_dataloader=validation_loader,
    prefetch_size=2,
)
```

The trainer accepts generic Python iterables, including:

* lists,
* generators,
* custom iterable datasets,
* and `taktiny.data.DataLoader` instances.

Loading, preprocessing, batching, shuffling, and sampling remain the responsibility of the data pipeline.

### Batch Sharding

Use `batch_sharding` to place incoming batches according to a JAX sharding specification:

```python
DatasetConfig(
    train_dataloader=train_loader,
    batch_sharding=batch_sharding,
)
```

`batch_sharding` may be a single sharding object applied to batch leaves or a PyTree matching the structure of the batch.

This allows data placement to integrate with `Mesh`, `NamedSharding`, and other distributed JAX configurations.

### Prefetching

`prefetch_size` controls how many batches may be prepared ahead of the training loop:

```python
DatasetConfig(
    train_dataloader=train_loader,
    prefetch_size=4,
)
```

Set it to `0` to disable prefetching.

## Checkpointing

Taktiny uses Orbax for checkpoint management.

Enable periodic checkpointing with:

```python
TrainingConfig(
    output_dir="./checkpoints",
    save_steps=500,
    save_total_limit=3,
)
```

Additional options include:

* `save_at_end` — save a checkpoint when training finishes,
* `save_async` — use asynchronous Orbax checkpoint writes,
* `save_optimizer_state` — include optimizer state for exact training resumption,
* `load_best_model_at_end` — restore the best checkpoint after training.

If optimizer state is omitted, a checkpoint may still restore model weights but cannot reproduce the exact optimizer state required to resume training.

## Callbacks

Callbacks extend the training loop without modifying the trainer itself.

```python
trainer = Trainer(
    ...,
    callbacks=[
        MyCallback(),
    ],
)
```

Custom callbacks can subclass `TrainerCallback` and implement lifecycle hooks such as step completion or evaluation events.

```python
from taktiny.trainer import TrainerCallback


class MyCallback(TrainerCallback):
    def on_step_end(self, *args, **kwargs):
        ...
```

Built-in reporting integrations include:

### TensorBoard

`TensorBoardCallback` writes training and evaluation metrics for visualization in TensorBoard.

```python
from taktiny.trainer import TensorBoardCallback
```

### Weights & Biases

`WandbCallback` reports training metrics to Weights & Biases.

```python
from taktiny.trainer import WandbCallback
```

These integrations are optional dependencies and can be installed separately when needed.
