# Trainer & Callbacks

The `taktiny.trainer` package provides an end-to-end training loop for JAX models with Optax optimizer management, automatic checkpointing via Orbax, and logging.

## Basic Training Loop

```python
import optax
from taktiny.trainer import Trainer, TrainingConfig, DatasetConfig

trainer = Trainer(
    model=model,
    loss_fn=my_loss_fn,
    training_config=TrainingConfig(
        max_steps=1000,
        optimizer=optax.adamw(learning_rate=3e-4),
        log_interval=50,
        jit_compile=True,
    ),
    dataset_config=DatasetConfig(dataloader=my_dataloader),
)

trainer.train()
```

## Callbacks & Logging

You can attach logging and evaluation callbacks:

- `TensorBoardCallback`: Exports scalar training loss and metrics to TensorBoard logs.
- `WandbCallback`: Syncs metrics, configurations, and gradients to Weights & Biases.
