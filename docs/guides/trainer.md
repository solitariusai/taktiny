# Training with `Trainer`

:::{note}
`Trainer`, its configuration, and its callbacks are experimental. Their
interfaces may change between releases.
:::

{py:class}`Trainer<taktiny.trainer.Trainer>` runs the
optimization loop for a model, loss function, and iterable of batches that you
provide. Add evaluation, gradient accumulation, callbacks, or Orbax
checkpoints as the training run needs them. Build the data pipeline with
{py:class}`DataLoader<taktiny.data.DataLoader>` or any iterable that yields
ready-to-use batches.

## Train a model

Each item in `train_batches` below is already a batch with a leading batch
dimension. The Trainer applies four optimizer steps:

```python
import jax.numpy as jnp
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


train_batches = [
    {"image": jnp.ones((8, 32)), "label": jnp.zeros((8, 4))}
    for _ in range(4)
]

trainer = Trainer(
    model=MLP(rngs=nn.Rngs(0)),
    loss_fn=loss_fn,
    training_config=TrainingConfig(
        max_steps=4,
        learning_rate=3e-4,
        log_interval=1,
    ),
    dataset_config=DatasetConfig(
        train_dataloader=train_batches,
        prefetch_size=0,
    ),
)

trainer.train()
```

With `optimizer` unset, Trainer uses AdamW with `learning_rate` and
`weight_decay`. To use another Optax transformation, pass it as `optimizer`:

```python
import optax

optimizer = optax.adam(3e-4)
config = TrainingConfig(max_steps=4, optimizer=optimizer)
```

For a changing learning rate, pass `schedule=...` to `TrainingConfig` with
the default optimizer, or pass the schedule directly to a custom Optax
optimizer.

## Write the loss function

The loss function receives the current model and one batch and returns a
scalar JAX array. Add an `rng` keyword parameter when the training step uses
randomness; Trainer supplies a per-step key:

```python
import jax


def loss_fn(model, batch, *, rng):
    noise = 0.01 * jax.random.normal(rng, batch["image"].shape)
    prediction = model(batch["image"] + noise)
    return jnp.mean((prediction - batch["label"]) ** 2)
```

For extra training metrics, return `(loss, metrics)` and construct Trainer with
`loss_has_aux=True`. The metrics mapping is included in step logs.

## Configure optimization

`TrainingConfig` groups the controls for optimization, evaluation, numerical
stability, and checkpointing. Common options are:

| Category          | Options                                                                             |
| ----------------- | ----------------------------------------------------------------------------------- |
| **Optimization**  | `optimizer`, `learning_rate`, `schedule`, `weight_decay`                            |
| **Gradients**     | `gradient_accumulation_steps`, `max_grad_norm`, `compute_grad_norm`                 |
| **Precision**     | `loss_scale`, `initial_loss_scale`, `loss_scale_growth_interval`, `skip_non_finite` |
| **Averaging**     | `ema_decay`                                                                         |
| **Evaluation**    | `eval_strategy`, `eval_steps`, `metric_for_best_model`, `greater_is_better`         |
| **Checkpointing** | `output_dir`, `save_steps`, `save_total_limit`, `save_at_end`, `save_async`         |
| **Execution**     | `max_steps`, `jit_compile`, `seed`, `log_interval`                                  |

### Gradient accumulation

Set `gradient_accumulation_steps` to accumulate gradients across multiple
batches before applying an optimizer update:

```python
TrainingConfig(
    gradient_accumulation_steps=4,
)
```

With an accumulation factor of `4`, Trainer evaluates four micro-batches
before each optimizer step. This provides a larger effective batch size while
keeping each individual batch small.

### Gradient clipping

Use `max_grad_norm` to clip gradients by their global norm before applying
optimizer updates:

```python
TrainingConfig(
    max_grad_norm=1.0,
)
```

`compute_grad_norm=False` skips norm tracking when clipping is unset. Clipping
still computes the norm it needs.

### Loss scaling

Loss scaling can improve numerical stability with low-precision arithmetic.

For dynamic loss scaling:

```python
TrainingConfig(
    loss_scale="dynamic",
)
```

Trainer scales the loss before differentiation and unscales the resulting
gradients before optimization.

With the default `skip_non_finite=True`, a non-finite step skips its optimizer
update and dynamic scaling adjusts the scale.

A fixed scale may also be provided:

```python
TrainingConfig(
    loss_scale=32768.0,
)
```

`initial_loss_scale` and `loss_scale_growth_interval` control how a dynamic
scale starts and grows.

### Exponential moving average

Set `ema_decay` to maintain an exponential moving average of model parameters:

```python
TrainingConfig(
    ema_decay=0.999,
)
```

The decay must satisfy `0 < ema_decay < 1`. After training starts,
`trainer.ema` returns an independent model containing the averaged weights.
Use that model when you want to evaluate or save the EMA version.

## Evaluation

Provide a validation iterable and choose when Trainer evaluates it. For
step-based evaluation:

```python
validation_batches = train_batches[:1]

config = TrainingConfig(
    eval_strategy="steps",
    eval_steps=2,
)

dataset_config = DatasetConfig(
    train_dataloader=train_batches,
    validation_dataloader=validation_batches,
)
```

The strategies are `"no"` (the default), `"steps"`, and `"epoch"`. The epoch
strategy evaluates at the end of each completed pass through the training
iterable. For a model-selection metric, set:

```python
config = TrainingConfig(
    max_steps=4,
    eval_strategy="steps",
    eval_steps=2,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    output_dir="./checkpoints",
)
```

`load_best_model_at_end=True` restores the best checkpoint after training.
Checkpoint selection uses the metric named by `metric_for_best_model`.

## Supply batches

`DatasetConfig` holds training and optional validation iterables. Each item
should already have the shape and fields expected by your loss function:

```python
dataset_config = DatasetConfig(
    train_dataloader=train_batches,
    prefetch_size=2,
)
```

Lists of batches, generators, custom iterables, and
{py:class}`DataLoader<taktiny.data.DataLoader>` all work here. For data
preparation and batching, see the [data guide](data.md).

### Batch sharding

Use `batch_sharding` to place incoming batches on a JAX mesh. A single
`Sharding` applies to every array leaf; a matching PyTree can give different
fields different placements:

```python
from jax.sharding import NamedSharding, PartitionSpec as P

dataset_config = DatasetConfig(
    train_dataloader=train_batches,
    batch_sharding=NamedSharding(mesh, P("data")),
)
```

Here `mesh` is a JAX mesh with a `"data"` axis, as in the [SPMD guide](spmd.md).
This spec partitions the leading batch dimension on that axis. You can also
configure placement in `DataLoader`; choose one placement point for a pipeline.

### Prefetching

`prefetch_size` controls how many batches Trainer prepares ahead of the
training loop:

```python
dataset_config = DatasetConfig(
    train_dataloader=train_batches,
    prefetch_size=4,
)
```

Use `0` for direct iteration. Benchmark the value with your data pipeline and
device transfer time.

## Checkpointing

Trainer saves checkpoints with Orbax. Set an output directory and a save
interval to keep snapshots during a run:

```python
config = TrainingConfig(
    max_steps=4,
    learning_rate=3e-4,
    output_dir="./checkpoints",
    save_steps=2,
    save_total_limit=3,
)
```

`save_at_end=True` also writes at the end of training, and `save_async=True`
overlaps checkpoint writes with training. Checkpoints include optimizer state by
default. To resume, construct a Trainer with the same model structure, optimizer,
and data order, then call:

```python
trainer.train(resume_from_checkpoint="latest")
```

`"latest"` selects the highest-numbered checkpoint in `output_dir`; you can
also pass a checkpoint directory. Use `save_optimizer_state=False` for a
weight-and-trainer-state snapshot when restoring the optimizer trajectory is
unnecessary. The [checkpoint guide](checkpoint.md) covers weight-only model
checkpoints separately.

## Callbacks

Callbacks receive training events and logs. For example, collect the reported
loss each time Trainer logs:

```python
from taktiny.trainer import TrainerCallback


class LossHistory(TrainerCallback):
    def __init__(self):
        self.values = []

    def on_log(self, trainer, logs):
        if logs["loss"] is not None:
            self.values.append(logs["loss"])


history = LossHistory()
```

Pass `callbacks=[history]` when constructing Trainer. Other hooks include
`on_train_begin`, `on_step_end`, `on_save`, `on_evaluate`, and `on_train_end`.

### TensorBoard

`TensorBoardCallback` writes training and evaluation metrics for visualization:

```python
from taktiny.trainer import TensorBoardCallback
```

### Weights & Biases

`WandbCallback` reports training metrics to Weights & Biases:

```python
from taktiny.trainer import WandbCallback
```

Install the corresponding `tensorboard` or `wandb` optional dependency before
using these callbacks.
