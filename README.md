# Taktiny

Taktiny is an experimental deep learning framework built directly on JAX. It provides object-oriented modules that remain valid JAX PyTrees, a robust set of neural network building blocks, and an end-to-end training loop integration. 

The project is currently focused on providing low-level, composable components for modern deep learning architectures. 

## Highlights

- **Object-Oriented Modeling**: Stateful `nn.Module` and `nn.Parameter` objects registered natively as JAX PyTrees.
- **PEFT Ecosystem**: Native support for Parameter-Efficient Fine-Tuning adapters including LoRA, DoRA, AdaLoRA, LoHa, LoKr, and VeRA via `Takt`.
- **Quantization**: Built-in support for weight-only post-training quantization through integration with `qwix`.
- **Full-Lifecycle Trainer**: Experimental `Trainer` abstraction integrating tightly with Optax for full distributed training, checkpointing, and logging (TensorBoard/WandB).
- **JAX Native**: Seamless compatibility with `jax.jit`, `jax.value_and_grad`, `jax.vmap`, and hardware mesh sharding.
- **Rust/PyO3 Extensions**: Hardware-accelerated subroutines using Rust bindings.

## Requirements

- Python 3.12+
- JAX 0.10.2+

## Installation & Development

To set up a local development environment:

```bash
uv sync --frozen --group dev
uv run maturin develop
```

To run the offline test suite on the CPU:

```bash
uv run --frozen pytest
```

## Quick Start: Building Models

Taktiny modules keep parameters directly on the object while participating seamlessly in JAX functional transformations. 

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

# Forward pass
dummy_input = jax.numpy.ones((1, 64))
output = jitted_model(dummy_input)
```

## Applying PEFT Adapters (Takt)

Taktiny features native `Takt` PEFT layers for injecting trainable adapters into existing topologies.

```python
from taktiny import Takt
from taktiny.takt.peft import LoRAAdapter

# Target the `input` linear layer in the previously built MLP model
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

The experimental `Trainer` connects Taktiny modules directly to Optax for update steps, checkpointing, and evaluation metrics.

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
├── nn/                 Core JAX object-oriented layers, activations, and parameters
├── takt/               Model transformations and PEFT implementations
├── trainer/            Experimental training loop utilities and callbacks
└── utils/              Typing, state dictionaries, and transform helpers

src/taktinylib/         Rust extension Python wrappers
lib/                    Core Rust implementations
```

## License

Taktiny is distributed under the Apache License 2.0. See [`LICENSE.md`](LICENSE.md).
