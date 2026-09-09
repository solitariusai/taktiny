# Quickstart

This guide walks through the basic Taktiny workflow: defining a model, using JAX transformations, preparing data, training, applying PEFT adapters, and running inference.

## 1. Define a Model

Taktiny models inherit from `nn.Module`. Parameters are stored directly by modules and participate in the JAX PyTree system.

Random state is provided explicitly through `nn.Rngs`.

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
        self.dropout = nn.Dropout(rate=0.1, rngs=rngs)
        self.fc2 = nn.Linear(hidden_features, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
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

`taktiny.data` provides a `DataLoader` built around composable data operations.

```python
from taktiny.data import Batch, DataLoader, train_validation_split


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
    test_size=0.25,
    seed=42,
)

train_loader = DataLoader(
    train_records,
    operations=[Batch(batch_size=16)],
)

validation_loader = DataLoader(
    validation_records,
    operations=[Batch(batch_size=16)],
)
```

The data pipeline is independent from the training loop. Any iterable that yields compatible batches can be used instead.

## 4. Train the Model

`Trainer` combines a model, loss function, optimizer configuration, and batch iterables into a training loop.

```python
import optax

from taktiny.trainer import DatasetConfig, Trainer, TrainingConfig


def loss_fn(model, batch):
    logits = model(batch["x"])

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch["y"],
    )

    return jnp.mean(loss)


trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    training_config=TrainingConfig(
        max_steps=50,
        learning_rate=1e-3,
        log_interval=10,
        eval_strategy="steps",
        eval_steps=25,
        output_dir="./checkpoints/classifier",
        save_at_end=True,
    ),
    dataset_config=DatasetConfig(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
    ),
)

trainer.train()
```

Taktiny handles optimization while leaving model definition, loss computation, and data preparation explicit.

Training behavior such as gradient accumulation, clipping, EMA, loss scaling, evaluation, and checkpointing can be configured through `TrainingConfig`.

## 5. Apply PEFT Adapters

Adapters can be applied to an existing model through `Takt`.

For example, LoRA can replace selected linear modules while freezing the parameters that already belong to the base model.

```python
from taktiny.takt import LoRAAdapter, Takt


adapter = LoRAAdapter(
    targets=["fc1", "fc2"],
    rank=4,
    alpha=8.0,
    rngs=nn.Rngs(101),
)

model = Takt.apply_adapter(model, adapter)
```

The resulting model is still an ordinary Taktiny module and can be passed to the same training infrastructure:

```python
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    training_config=TrainingConfig(
        max_steps=20,
        learning_rate=5e-4,
        log_interval=5,
    ),
    dataset_config=DatasetConfig(
        train_dataloader=train_loader,
    ),
)

trainer.train()
```

Other adapters can be applied through the same interface without changing the surrounding model or training code.

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

Because Taktiny models remain JAX PyTrees throughout the workflow, the same model object can move between initialization, transformation, training, adaptation, and inference without requiring a separate functional parameter representation.
