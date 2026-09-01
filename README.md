# Taktiny

Taktiny is an experimental neural-network library for JAX. It provides object-oriented modules that are registered as JAX PyTrees, basic network layers, and a training loop.

The project is currently in development and APIs are subject to change.

## Features

- Stateful `nn.Module` and `nn.Parameter` objects
- PEFT adapters via `Takt` (LoRA, DoRA, AdaLoRA, LoHa, LoKr, VeRA)
- Weight-only PTQ using `qwix`
- An experimental trainer using Optax, with checkpointing and logging
- Compatible with standard JAX transformations (`jit`, `vmap`, `value_and_grad`)

## Requirements

- Python 3.12+
- JAX 0.10.2+

## Installation & Development

To set up a local development environment:

```bash
uv sync --frozen --group dev
```

To run the offline test suite on the CPU:

```bash
uv run --frozen pytest
```

## Building Models

Taktiny modules hold their parameters directly while acting as standard JAX PyTrees.

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
        self.input = nn.Linear(in_features, hidden_features, rngs=rngs)
        self.output = nn.Linear(hidden_features, out_features, rngs=rngs)

    def __call__(self, x):
        return self.output(jax.nn.silu(self.input(x)))

model = MLP(64, 128, 10, rngs=nn.Rngs(42))
jitted_model = jax.jit(model)

dummy_input = jax.numpy.ones((1, 64))
output = jitted_model(dummy_input)
```

## Applying PEFT

The `Takt` namespace provides functions for applying PEFT adapters to an existing model.

```python
from taktiny import Takt
from taktiny.takt.peft import LoRAAdapter

model = Takt.apply_peft(
    model,
    LoRAAdapter(
        target_modules=["input"],
        rank=16,
        alpha=32,
    ),
)
```

## Training

The `Trainer` class wraps an Optax optimizer and a Taktiny model for training.

```python
import optax
from taktiny import Trainer, TrainingConfig, DatasetConfig

trainer = Trainer(
    model=model,
    loss_fn=my_loss_fn,
    training_config=TrainingConfig(
        max_steps=1000,
        optimizer=optax.adamw(3e-4),
        log_interval=10,
        jit_compile=True,
    ),
    dataset_config=DatasetConfig(dataloader=my_batches),
)

trainer.train()
```

## Project Layout

```text
src/taktiny/
├── nn/                 JAX modules, layers, and parameters
├── takt/               Model transformations and PEFT implementations
├── trainer/            Training loop utilities and callbacks
└── utils/              Typing, state dictionaries, and transform helpers
```

## License

Taktiny is distributed under the Apache License 2.0. See [`LICENSE.md`](LICENSE.md).
