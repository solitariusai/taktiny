# Quickstart

This guide walks through the basic Taktiny workflow: defining a model, using JAX transformations, preparing data, writing a training loop, applying PEFT adapters, and running inference.

## 1. Define a Model

Taktiny models inherit from {py:class}`nn.Module<taktiny.nn.Module>`. Parameters are stored directly by modules and participate in the JAX PyTree system.

Random state is provided explicitly through {py:class}`nn.Rngs<taktiny.nn.Rngs>`.

```python
import jax
import jax.numpy as jnp

from taktiny import nn


class Classifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        *,
        rngs: nn.Rngs,
    ):
        self.fc1 = nn.Linear(in_features, hidden_features, rngs=rngs)
        self.norm = nn.LayerNorm(hidden_features)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.fc2(x)


model = Classifier(
    in_features=16,
    hidden_features=32,
    num_classes=4,
    rngs=nn.Rngs(42),
)
```

## 2. Use JAX Transformations

Taktiny modules are JAX PyTrees, so they can be passed directly through standard JAX transformations.

```python
x = jnp.ones((8, 16))

jit_model = jax.jit(model)
logits = jit_model(x)

print(logits.shape)
# (8, 4)
```

The model itself may also be passed as an argument to transformed functions:

```python
@jax.jit
def forward(model, x):
    return model(x)


logits = forward(model, x)
```

The same PyTree representation allows Taktiny modules to participate in transformations such as `jax.grad`, `jax.value_and_grad`, and `jax.vmap`.

## 3. Prepare Data

`taktiny.data` provides a {py:class}`DataLoader<taktiny.data.DataLoader>` built around composable data operations.

```python
from taktiny.data import DataLoader, train_validation_split


records = [
    {
        "x": jnp.asarray(
            [i * 0.1 + j for j in range(16)],
            dtype=jnp.float32,
        ),
        "y": jnp.asarray(i % 4, dtype=jnp.int32),
    }
    for i in range(128)
]

train_records, validation_records = train_validation_split(
    records,
    validation_size=0.25,
    seed=42,
)

train_loader = DataLoader(
    train_records,
    batch_size=16
)

validation_loader = DataLoader(
    validation_records,
    batch_size=16
)
```

The data pipeline is independent from the training loop. Any iterable that yields compatible batches can be used instead.

## 4. Train the Model

Define a loss function, then use JAX to compute its gradients. Taktiny's
{py:class}`Optimizer<taktiny.takt.Optimizer>` wraps an Optax transformation and
keeps its state alongside the model in the training loop.

```python
import optax

from taktiny.takt import Optimizer


def loss_fn(model, batch):
    logits = model(batch["x"])

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch["y"],
    )

    return jnp.mean(loss)


optimizer = Optimizer(model, optax.adam(1e-3))


@jax.jit
def train_step(model, optimizer, batch):
    loss, gradients = jax.value_and_grad(loss_fn)(model, batch)
    model = optimizer.update(model, gradients)
    return model, optimizer, loss


for epoch in range(3):
    for batch in train_loader:
        model, optimizer, loss = train_step(model, optimizer, batch)
    print(f"epoch {epoch + 1}: last batch loss {float(loss):.4f}")

validation_loss = jnp.mean(jnp.stack([
    loss_fn(model, batch) for batch in validation_loader
]))
print(f"validation loss: {float(validation_loss):.4f}")
```

The model and optimizer are both returned by `train_step`, so their updated
states are passed into the next step. Any iterable yielding compatible batches
can replace `train_loader`.

The experimental {py:class}`Trainer<taktiny.trainer.Trainer>` is available if
you prefer a configurable training loop; see the [Trainer guide](../guides/trainer.md).

## 5. Optional: Add LoRA

You can skip this step and use the trained classifier as it is. To fine-tune
with LoRA, wrap the existing linear layers directly with
{py:class}`nn.LoRALinear<taktiny.nn.LoRALinear>`:

```python
model.fc1 = nn.LoRALinear(
    model.fc1,
    rank=4,
    alpha=8.0,
    rngs=nn.Rngs(101),
    bias=False,
)
model.fc2 = nn.LoRALinear(
    model.fc2,
    rank=4,
    alpha=8.0,
    rngs=nn.Rngs(102),
    bias=False,
)
```

Wrapping a layer does not automatically freeze its base parameters. Create a
new optimizer that selects only the LoRA kernels, then reuse `train_step`:

```python
optimizer = Optimizer(
    model,
    optax.adam(5e-4),
    include=[r"fc[12]\.lora_[AB]\.kernel"],
)

for batch in train_loader:
    model, optimizer, loss = train_step(model, optimizer, batch)
```

The selection also prevents weight decay or other optimizer updates from
changing the base layers. The experimental adapter framework can automate
matching and replacement across larger models; see the [PEFT guide](../guides/peft.md).

## 6. Evaluation and Inference

Switch the model to evaluation mode before inference:

```python
model.eval()
```

Modules whose behavior depends on training state, such as dropout, will use their evaluation behavior.

Inference can then be compiled normally with JAX:

```python
@jax.jit
def predict(model, x):
    logits = model(x)
    return jnp.argmax(logits, axis=-1)


x = jnp.ones((1, 16))

prediction = predict(model, x)

print(prediction)
```

## 7. Save and Restore Weights

Choose either format below. Safetensors creates a single file; Orbax creates a
checkpoint directory.

### Safetensors

Install Safetensors:

```bash
uv add safetensors
# Or, with pip:
pip install safetensors
```

Save the model's flat parameter dictionary directly:

```python
from safetensors.flax import save_file

save_file(model.flat_state_dict(), "classifier.safetensors")
```

To restore the file into a compatible model:

```python
from safetensors.flax import load_file

model.load_flat_state_dict(load_file("classifier.safetensors"))
```

### Orbax

Orbax is already a Taktiny dependency, so no additional install is needed:

```python
from pathlib import Path
import orbax.checkpoint as ocp

checkpoint_dir = Path("classifier_checkpoint").resolve()
with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(checkpoint_dir, model.state_dict())
    checkpointer.wait_until_finished()
```

Restore the checkpoint using the model's current state as the target structure:

```python
with ocp.StandardCheckpointer() as checkpointer:
    state = checkpointer.restore(checkpoint_dir, target=model.state_dict())

model.load_state_dict(state)
```

Both formats save weights, not the model definition or optimizer state. To load
them in a new session, first construct the same model structure (including LoRA
layers if you added them). Choose a new directory for each Orbax save. For a
resumable training checkpoint, see the
[checkpoint guide](../guides/checkpoint.md).

Because Taktiny models remain JAX PyTrees throughout the workflow, the same model object can move between initialization, transformation, training, adaptation, and inference without requiring a separate functional parameter representation.
