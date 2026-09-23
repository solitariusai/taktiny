# Parameter-efficient fine-tuning

:::{note}
PEFT layers and `Takt` adapter APIs are experimental. Their interfaces may
change between releases.
:::

Taktiny's PEFT layers add a small trainable update to an existing
{py:class}`nn.Linear<taktiny.nn.Linear>` layer. You can wrap layers yourself,
or use `Takt.apply_adapter()` to find and replace
matching layers throughout a model. The base model and adapter parameters
remain separate, which makes it possible to optimize and save just the update.

## Wrap a layer with LoRA

Start with a model whose linear layers you want to adapt:

```python
import jax
import jax.numpy as jnp

from taktiny import nn


class MLP(nn.Module):
    def __init__(self, *, rngs: nn.Rngs):
        self.dense1 = nn.Linear(32, 64, rngs=rngs)
        self.dense2 = nn.Linear(64, 4, rngs=rngs)

    def __call__(self, x):
        return self.dense2(jax.nn.relu(self.dense1(x)))


model = MLP(rngs=nn.Rngs(0))
model.dense1 = nn.LoRALinear(
    model.dense1, rank=4, alpha=8.0, rngs=nn.Rngs(1), bias=False,
)
model.dense2 = nn.LoRALinear(
    model.dense2, rank=4, alpha=8.0, rngs=nn.Rngs(2), bias=False,
)
```

Each wrapper keeps the original layer as `base` and adds two low-rank
projections, `lora_A` and `lora_B`. It computes the base output plus a scaled
LoRA update. The update initially contributes zero, so wrapping the layer
does not change its output before training.

Wrapping a layer does **not** by itself stop gradients through its base. Select
the parameters to update in the optimizer. Here, the regular expression
matches only the LoRA kernels:

```python
import optax

from taktiny.takt import Optimizer

optimizer = Optimizer(
    model,
    optax.adam(1e-3),
    include=[r"dense[12]\.lora_[AB]\.kernel"],
)


def loss_fn(model, x, y):
    return jnp.mean((model(x) - y) ** 2)


@jax.jit
def train_step(model, optimizer, x, y):
    loss, gradients = jax.value_and_grad(loss_fn)(model, x, y)
    model = optimizer.update(model, gradients)
    return model, optimizer, loss


x = jnp.ones((2, 32))
y = jnp.zeros((2, 4))
model, optimizer, loss = train_step(model, optimizer, x, y)
```

The optimizer creates state and applies updates only for selected leaves;
unselected base weights also receive no weight decay. The optimizer selection
does not itself prevent `jax.value_and_grad` from calculating gradients for
the base parameters.

## Inject adapters by path

For a model with many target layers, `Takt.apply_adapter()` can do the
replacement. `targets` accepts one regular expression or a list of them, and
each expression is searched against a module path:

```python
from taktiny.takt import LoRAAdapter, Takt

base_model = MLP(rngs=nn.Rngs(3))
adapter = LoRAAdapter(
    targets=[r"^dense1$", r"^dense2$"],
    rank=4,
    alpha=8.0,
    rngs=nn.Rngs(4),
)
adapted_model = Takt.apply_adapter(base_model, adapter)
```

`apply_adapter()` changes `base_model` in place and returns it. It marks
pre-existing parameters as `trainable=False` and leaves new adapter parameters
trainable. This flag is metadata used by training code that honors it, such as
Taktiny's experimental Trainer; with a manual Optax loop, select the leaves
explicitly as in the first example.

The convenience adapter classes target linear layers. Their corresponding
`nn.*Linear` wrappers can also be used directly:

| Adapter helper | Direct wrapper | Update |
| --- | --- | --- |
| `LoRAAdapter` | `nn.LoRALinear` | Two low-rank projections. |
| `DoRAAdapter` | `nn.DoRALinear` | A low-rank direction update and learned magnitude. |
| `AdaLoRAAdapter` | `nn.AdaLoRALinear` | Low-rank factors with a maskable rank vector. |
| `LoHaAdapter` | `nn.LoHaLinear` | Hadamard product of two low-rank updates. |
| `LoKrAdapter` | `nn.LoKrLinear` | Kronecker-product update. |
| `VeRAAdapter` | `nn.VeRALinear` | Shared frozen projections with trainable scales. |

`VeRAAdapter` prepares projections large enough for all matched layers and
shares them across those layers. They are part of the model state even though
the projections themselves are frozen.

## Quantized bases and sharding

A PEFT wrapper can sit around a quantized `nn.Linear` base. The base keeps its
Qwix quantized weight; newly created adapter parameters remain floating-point
by default. Apply quantization to the base before wrapping it, or construct
the base with its `quant` argument. Quantizing a model later is a separate
model-conversion step, not something `LoRAAdapter` performs automatically.

Direct wrappers inherit the base kernel's logical axis names and partition
specification when no sharding arguments are supplied. Logical names are
resolved under the current mapping rules, so construct the wrapper in the
intended mesh and mapping context. To choose a different adapter layout, pass
`axis_names` or `partition_spec` to the direct wrapper. For example,
`partition_spec=PartitionSpec()` requests replicated adapter factors while
leaving the base layer's sharding unchanged. See the [SPMD guide](spmd.md) for
mesh and logical-axis setup.

## AdaLoRA rank updates

`nn.AdaLoRALinear` provides `mask_rank()` and `orthogonal_loss()`. It does not
choose a rank budget or schedule pruning automatically. Your training code
must add the orthogonal penalty to its loss if desired and apply masks at the
steps you choose:

```python
layer = nn.AdaLoRALinear(
    nn.Linear(32, 64, rngs=nn.Rngs(5)),
    rank=4,
    alpha=8.0,
    rngs=nn.Rngs(6),
)

penalty = layer.orthogonal_loss()
layer.mask_rank(jnp.array([True, True, False, False]))
```

`mask_rank()` zeros the selected rank entries; later optimizer updates can
regrow them, so reapply the mask if those entries must stay inactive.

## Save adapter weights

For the direct LoRA model above, select the adapter leaves from
`flat_state_dict()`:

```python
adapter_state = {
    name: value
    for name, value in model.flat_state_dict().items()
    if name.startswith(("dense1.lora_", "dense2.lora_"))
}
```

Save this mapping with Orbax as shown in the [checkpoint guide](checkpoint.md).
To restore it, first rebuild or load the same base model, wrap the same layers,
then call `model.load_flat_state_dict(restored_adapter_state)`. For other PEFT
methods, include every adapter value required at inference—not only values
selected by the optimizer. VeRA's frozen shared projections are one example.
