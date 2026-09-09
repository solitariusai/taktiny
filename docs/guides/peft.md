# Parameter-Efficient Fine-Tuning (PEFT)

The `taktiny.takt` package provides adapters that transform existing models by injecting low-rank or parameter-efficient modules while automatically freezing the pre-existing base weights.

Adapters in Taktiny are designed with JAX's SPMD paradigm in mind. They correctly inherit logical axis names and partition specifications from the base layers they wrap, support quantized base weights via Qwix, and naturally separate trainable and frozen states for checkpointing.

## Injecting an Adapter

In Taktiny, adapters are not standalone modules; they are applied dynamically to an existing model using `Takt.apply_adapter()`.

```python
from taktiny import nn
from taktiny.takt import Takt, LoRAAdapter

# 1. Start with an existing base model
model = nn.Sequential([
    nn.Linear(64, 128, rngs=nn.Rngs(0)),
    nn.Linear(128, 10, rngs=nn.Rngs(1)),
])

# 2. Configure the adapter
# "targets" can be a single regex string or a list of regex patterns.
adapter = LoRAAdapter(
    targets=".*",
    rank=8,
    alpha=16.0,
    rngs=nn.Rngs(42),
)

# 3. Apply the adapter
adapted_model = Takt.apply_adapter(model, adapter)
```

When an adapter is applied, `apply_adapter`:
1. Searches the model for target layers (e.g., `nn.Linear`) that match the regex pattern.
2. Replaces those layers with specialized wrappers (like `LoRALinear`) that encapsulate the original frozen layer.
3. Automatically sets `trainable = False` on all pre-existing parameters across the entire model.
4. Leaves the newly injected parameters as `trainable = True`.

## Quantization and Qwix Support

Taktiny adapters fully support mixed-precision and quantized fine-tuning (QLoRA/QPEFT).

Because the specialized wrappers separate the base weight computation from the adapter update, the base `Linear` module can be quantized using Qwix. When you run a forward pass, the base layer will dequantize its weights and compute its contribution, while the trainable adapter parameters remain in pure floating point (e.g., `bfloat16` or `float32`).

You do not need special configuration for QPEFT: simply quantize the model before or after applying the adapter.

## Sharding and SPMD

Adapters gracefully handle JAX's distributed sharding. 

When you apply an adapter to a layer, the new adapter parameters must decide how they will be sharded across the `Mesh`. By default, if you don't explicitly pass sharding arguments to the adapter, Taktiny uses the following inheritance logic:
1. **Logical Axis Names**: The adapter inherits the logical labels (e.g., `("input", "output")`) from the base layer and resolves them using the current mapping rules. 
2. **Physical Partition Specs**: If the base layer doesn't have logical names, the adapter inherits its exact physical `PartitionSpec`.

If you want the adapter to use a *different* sharding layout than the base layer, you can provide `partition_spec` or `axis_names` directly to the adapter constructor.
To force an adapter to be fully replicated across all devices while the base layer remains sharded, explicitly pass `partition_spec=PartitionSpec()`:

```python
from jax.sharding import PartitionSpec

adapter = LoRAAdapter(
    targets=".*",
    rank=4,
    partition_spec=PartitionSpec(),  # Force replicated adapter parameters
    rngs=nn.Rngs(42),
)
```

## Supported Adapters

Taktiny includes several state-of-the-art PEFT methods. They all inherit from `AdapterBase`.

| Adapter | Mechanism | Best For |
| --- | --- | --- |
| **`LoRAAdapter`** | `base(x) + (alpha/rank) * B(A(x))` | Standard fine-tuning tasks. |
| **`DoRAAdapter`** | Decouples magnitude and direction of the weights. Adapts direction via LoRA and learns a separate output magnitude parameter. | High-performance fine-tuning matching full-rank updates. |
| **`AdaLoRAAdapter`** | An SVD-style adapter: `base(x) + (alpha/rank) * B(E * A(x))`. | Strict parameter budgets. |
| **`LoHaAdapter`** | Hadamard product adapter with `delta W = (A1 B1) * (A2 B2)`. | Computer vision and stable diffusion. |
| **`LoKrAdapter`** | Kronecker product weight adapter: `delta W = kron(W1, W2)`. | Extremely high-dimensional target layers. |
| **`VeRAAdapter`** | Uses frozen random projections shared by all targets, learning only diagonal rank/output scales. | Maximum parameter efficiency. |

### Shared Projections with VeRA

`VeRAAdapter` drastically reduces trainable parameters by sharing state across multiple target layers. Taktiny's `prepare` hook automatically computes the maximum input and output dimensions across all matching layers. It creates shared, frozen random projections (`vera_A` and `vera_B`) that are reused across all targets.

### Post-Step Updates (AdaLoRA)

A few adapters require periodic maintenance during training. `AdaLoRAAdapter` dynamically masks out less important singular values based on importance scores, and it requires an orthogonal regularization penalty. 

To support this, `Takt` retains the applied adapter objects as static metadata on the model (`model._takt_adapters`). Your training loop or callback can invoke `Takt.update_adapters()` after an optimizer step to execute these hooks:

```python
# During your training loop, after optax.apply_updates:
Takt.update_adapters(adapted_model, step=current_step)
```

You can compute custom penalties by iterating over the model tree. For example, to add AdaLoRA's orthogonal loss:

```python
from taktiny.nn.modules.peft import AdaLoRALinear

adapter_loss = 0.0
for layer in adapted_model.flat_children():
    if isinstance(layer, AdaLoRALinear):
        adapter_loss += layer.orthogonal_loss()
```

## Checkpointing and Resuming

Taktiny's trainer separates parameters into `trainable_params` and `frozen_params` internally. Because `Takt.apply_adapter` sets the base parameters to `trainable = False`, you can trivially extract just the adapter weights for saving:

```python
# To save only the adapter parameters:
adapter_params = {
    path: param 
    for path, param in adapted_model.flat_parameter_dict().items() 
    if param.trainable
}
```

When resuming, load the base model checkpoint first, apply the adapter architecture, and then inject the loaded adapter parameters.

## A complete training example

This snippet demonstrates creating a model, applying LoRA, and using the Taktiny `Trainer` to fine-tune it. The trainer natively respects `trainable` flags and will only optimize the adapter.

```python
import jax.numpy as jnp
import optax
from taktiny import nn
from taktiny.takt import Takt, LoRAAdapter
from taktiny.trainer import Trainer, TrainingConfig, DatasetConfig

class MLP(nn.Module):
    def __init__(self, *, rngs: nn.Rngs):
        self.dense1 = nn.Linear(32, 64, rngs=rngs)
        self.dense2 = nn.Linear(64, 4, rngs=rngs)

    def __call__(self, x):
        return self.dense2(jax.nn.relu(self.dense1(x)))

# 1. Initialize the base model
model = MLP(rngs=nn.Rngs(0))

# 2. Inject the adapter
adapter = LoRAAdapter(
    targets="dense.*",
    rank=4,
    rngs=nn.Rngs(1),
)
model = Takt.apply_adapter(model, adapter)

# 3. Dummy dataset and loss
def dummy_loss(model_params, batch, **kwargs):
    x, y = batch["x"], batch["y"]
    # Taktiny models are PyTrees! The trainer passes the updated model 
    # as the first argument, so you just call it directly.
    logits = model_params(x)
    return jnp.mean((logits - y) ** 2)

dummy_data = [{"x": jnp.ones(32), "y": jnp.zeros(4)} for _ in range(10)]

# 4. Train only the adapter
trainer = Trainer(
    model=model,
    loss_fn=dummy_loss,
    training_config=TrainingConfig(
        max_steps=5,
        optimizer=optax.adamw(learning_rate=1e-3),
    ),
    dataset_config=DatasetConfig(dummy_data, batch_size=2),
)

trainer.train()
```
