# Taktiny

Taktiny modules are Python objects registered as JAX PyTrees. The library does
not assume a particular model architecture or dataset format, and its
components can be used independently.

The project is experimental and APIs may change.

[Quickstart](docs/getting_started/quickstart.md) ·
[Guides](docs/index.md) ·
[API reference](docs/api/nn.md)

## What’s included

- **Models:** `Module` and `Parameter`, linear and convolutional layers,
  embeddings, normalization, recurrent layers, attention, and other neural
  network components.
- **Data:** Grain-backed data loading, transforms, batching, and packing for
  caller-provided records.
- **Training:** An Optax-based trainer with evaluation, callbacks, gradient
  accumulation, and Orbax checkpoints.
- **Sharding:** Partition specifications and logical axis mappings for JAX
  device meshes.
- **Adapters:** LoRA, DoRA, AdaLoRA, LoHa, LoKr, and VeRA.
- **Quantization:** Quantization utilities backed by Qwix.

The data, training, and model APIs can also be used separately with existing
JAX code.

## Installation

Taktiny requires Python **3.12+** and JAX **0.10.2+**.

Install with `uv`:

```bash
uv add git+https://github.com/solitariusai/taktiny.git@experiment
```

Or with `pip`:

```bash
pip install git+https://github.com/solitariusai/taktiny.git@experiment
```

## Define a model

Modules hold their parameters directly and are registered as JAX PyTrees.

```python
import jax
import jax.numpy as jnp
from taktiny import nn


class MLP(nn.Module):
    def __init__(self, *, rngs: nn.Rngs):
        self.hidden = nn.Linear(8, 32, rngs=rngs)
        self.output = nn.Linear(32, 1, rngs=rngs)

    def __call__(self, x):
        return self.output(jax.nn.silu(self.hidden(x)))


model = MLP(rngs=nn.Rngs(0))

jit_model = jax.jit(model)
output = jit_model(jnp.ones((4, 8)))

assert output.shape == (4, 1)
```

Passing the model as an argument to a compiled function makes its parameters
part of the function inputs rather than capturing them in a closure.

## Prepare data and train

The following example trains the model above on in-memory records.

```python
import numpy as np
import optax

from taktiny.data import DataLoader
from taktiny.trainer import DatasetConfig, Trainer, TrainingConfig


inputs = np.random.default_rng(0).normal(size=(32, 8)).astype(np.float32)
records = [{"x": x, "y": x.sum(keepdims=True)} for x in inputs]

loader = DataLoader(
    records,
    batch_size=8,
    shuffle=True,
    seed=0,
    num_epochs=None,
)


def loss_fn(model, batch):
    return jnp.mean((model(batch["x"]) - batch["y"]) ** 2)


trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    training_config=TrainingConfig(
        max_steps=20,
        optimizer=optax.adam(1e-3),
        log_interval=10,
    ),
    dataset_config=DatasetConfig(
        train_dataloader=loader,
    ),
)

trainer.train()

assert trainer.global_step == 20
```

`Trainer` accepts iterables of batches. Checkpointing is optional.

See the [trainer guide](docs/guides/trainer.md) for evaluation, saving, and
resuming.

## Apply an adapter

Adapters can be applied to matching module paths.

```python
from taktiny.takt import LoRAAdapter, Takt


adapted = MLP(rngs=nn.Rngs(1))

adapted = Takt.apply_adapter(
    adapted,
    LoRAAdapter(
        targets="hidden",
        rank=4,
        alpha=8,
        rngs=nn.Rngs(2),
    ),
)

assert adapted(jnp.ones((4, 8))).shape == (4, 1)
```

`targets` accepts module-path regex patterns. Applying an adapter freezes
existing parameters and adds trainable adapter parameters.

See the [PEFT guide](docs/guides/peft.md).

## Documentation

- [Quickstart](docs/getting_started/quickstart.md)
- [Data loading and transforms](docs/guides/data.md)
- [Sharding](docs/guides/spmd.md)
- [Training and checkpoints](docs/guides/trainer.md)
- [PEFT and adapters](docs/guides/peft.md)
- [Tutorials](docs/tutorial)

## Development

Run the test suite on CPU:

```bash
make test
```

Project layout:

```text
src/taktiny/
├── nn/        Modules, layers, parameters, and RNG utilities
├── data/      Loading and preprocessing
├── takt/      Adapter injection
├── trainer/   Training, evaluation, callbacks, and checkpoints
└── utils/     Sharding, transforms, quantization, and typing
```

## License

Taktiny is distributed under the Apache License 2.0.
See [`LICENSE.md`](LICENSE.md).