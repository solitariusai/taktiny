# Quickstart

This guide demonstrates how to build, train, fine-tune, and evaluate a neural network with Taktiny.

---

## 1. Defining a Model

In Taktiny, models inherit from `taktiny.nn.Module`. Parameters are instances of `taktiny.nn.Parameter` and are registered as JAX PyTrees. Random state is passed explicitly using `taktiny.nn.Rngs`:

```python
import jax
import jax.numpy as jnp
from taktiny import nn

class Classifier(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int, *, rngs: nn.Rngs):
        self.fc1 = nn.Linear(in_features, hidden, rngs=rngs)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(rate=0.1, rngs=rngs)
        self.fc2 = nn.Linear(hidden, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)

# Initialize with seeded random state
rngs = nn.Rngs(42)
model = Classifier(in_features=16, hidden=32, num_classes=4, rngs=rngs)
```

---

## 2. Functional JIT and Gradients

Because `nn.Module` instances are pure JAX PyTrees, you can pass them directly to `jax.jit`, `jax.vmap`, and `jax.grad`:

```python
# JIT-compile the model forward pass
jitted_forward = jax.jit(lambda m, x: m(x))
x = jnp.ones((8, 16))
logits = jitted_forward(model, x)
print("Output logits shape:", logits.shape)  # (8, 4)
```

---

## 3. Data Loading and Batching

Taktiny provides a fast data loading pipeline via `taktiny.data.DataLoader` and composable operations:

```python
from taktiny.data import DataLoader, Batch, train_validation_split

# Generate synthetic dataset
raw_data = [
    {
        "x": jnp.array([float(i * 0.1 + j) for j in range(16)]),
        "y": jnp.array(i % 4, dtype=jnp.int32),
    }
    for i in range(128)
]

# Split into train and validation sets
train_records, val_records = train_validation_split(raw_data, test_size=0.25, seed=42)

# Build DataLoaders
train_loader = DataLoader(train_records, operations=[Batch(batch_size=16)])
val_loader = DataLoader(val_records, operations=[Batch(batch_size=16)])
```

---

## 4. Automated Training with `Trainer`

Use `taktiny.trainer.Trainer` to handle the training loop, loss optimization, metrics logging, and checkpointing:

```python
import optax
from taktiny.trainer import Trainer, TrainingConfig, DatasetConfig

# Define loss function
def loss_fn(m: Classifier, batch: dict[str, jax.Array]) -> jax.Array:
    logits = m(batch["x"])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch["y"])
    return jnp.mean(loss)

# Configure hyperparameters
training_config = TrainingConfig(
    max_steps=50,
    learning_rate=1e-3,
    log_interval=10,
    eval_strategy="steps",
    eval_steps=25,
    save_at_end=True,
    output_dir="./checkpoints/classifier",
)

dataset_config = DatasetConfig(
    train_dataloader=train_loader,
    validation_dataloader=val_loader,
)

# Launch training
trainer = Trainer(
    model=model,
    training_config=training_config,
    dataset_config=dataset_config,
    loss_fn=loss_fn,
)

trainer.train()
```

---

## 5. Parameter-Efficient Fine-Tuning (PEFT)

To adapt a model without full retraining, inject low-rank adapters via `taktiny.takt.Takt`. Base weights are automatically frozen (`trainable=False`), so only adapter parameters are updated:

```python
from taktiny.takt import Takt, LoRAAdapter

# Define LoRA adapter targeting linear layers
lora = LoRAAdapter(
    rank=4,
    alpha=8.0,
    targets=["fc1", "fc2"],
    rngs=nn.Rngs(101),
)

adapted_model = Takt.apply_adapter(model, lora)

# Train only adapter weights
peft_config = TrainingConfig(
    max_steps=20,
    learning_rate=5e-4,
    log_interval=5,
    output_dir="./checkpoints/lora_finetuned",
)

peft_trainer = Trainer(
    model=adapted_model,
    training_config=peft_config,
    dataset_config=dataset_config,
    loss_fn=loss_fn,
)

peft_trainer.train()
```

---

## 6. Checkpoint Restoration and Inference

Restore checkpoints with Orbax and run fast inference:

```python
# Restore saved checkpoint
trainer.restore_checkpoint("./checkpoints/classifier/checkpoint_50")

# Set to eval mode (e.g. disables dropout)
model.eval()

# Fast JIT inference
@jax.jit
def predict(m, x):
    return jnp.argmax(m(x), axis=-1)

sample_input = jnp.ones((1, 16))
prediction = predict(model, sample_input)
print("Predicted class:", prediction)
```
