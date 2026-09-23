# Taktiny

A neural networks library for JAX. Build models from Python modules, pass them
through JAX transformations, and use the data and training tools that fit your
project.

Taktiny is experimental; APIs may change.

[Installation](docs/getting_started/installation.md) ·
[Quickstart](docs/getting_started/quickstart.md) ·
[Guides](docs/index.md) ·
[API reference](docs/api/nn.md)

## Install

Taktiny requires Python 3.12 or newer. Install the PyPI release with `uv` or
`pip`:

```bash
uv add taktiny
# or
pip install taktiny
```

The default installation uses JAX on CPU. For accelerator extras and source
installs, see the [installation guide](docs/getting_started/installation.md).
The `master` branch is the more-stable source branch; `experiment` has newer,
less-tested changes.

## Build and train a model

Modules own their parameters and are JAX PyTrees. Pass a model and optimizer
through a compiled training step just as you would pass arrays:

```python
import jax
import jax.numpy as jnp
import optax

from taktiny import nn
from taktiny.takt import Optimizer


class MLP(nn.Module):
    def __init__(self, *, rngs: nn.Rngs):
        self.hidden = nn.Linear(8, 32, rngs=rngs)
        self.output = nn.Linear(32, 1, rngs=rngs)

    def __call__(self, x):
        return self.output(jax.nn.silu(self.hidden(x)))


model = MLP(rngs=nn.Rngs(0))
optimizer = Optimizer(model, optax.adam(1e-2))


@jax.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        return jnp.mean((model(x) - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(model)
    model = optimizer.update(model, grads)
    return model, optimizer, loss


x = jax.random.normal(jax.random.key(1), (32, 8))
y = jnp.sum(x, axis=-1, keepdims=True)

for _ in range(20):
    model, optimizer, loss = train_step(model, optimizer, x, y)

print(loss)
```

`Optimizer(include=[r"hidden\..*"])` selects parameters by full regex match
when you want to update only part of a model. The
[quickstart](docs/getting_started/quickstart.md) continues with data loading,
evaluation, and inference.

## Work with weights

Use dotted paths to obtain or load a subset of parameters:

```python
hidden_state = model.flat_state_dict(include=[r"hidden\..*"])

restored = MLP(rngs=nn.Rngs(2))
restored.load_flat_state_dict(hidden_state, include=[r"hidden\..*"])
```

`state_dict()` and `load_state_dict()` offer the same `include` filter with a
nested dictionary. All four methods accept `None` for the full state or `[]`
for no parameters. See the [checkpoint guide](docs/guides/checkpoint.md) for
saving to disk and restoring sharded weights.

## Explore

- [Layers and modules](docs/api/nn.md): linear and convolutional layers,
  embeddings, normalization, recurrent layers, attention, and more. Linear and
  convolutional layers support structured feature dimensions.
- [Data](docs/guides/data.md): Grain-backed loading, transforms, batching, and
  packing for caller-provided records.
- [Sharding](docs/guides/spmd.md): logical axis names and JAX mesh placement.
- [Quantization](docs/api/utils/quantization.md): Qwix-backed quantization
  utilities and operations.
- [Trainer](docs/guides/trainer.md): an experimental configurable training loop
  with evaluation, callbacks, accumulation, and Orbax checkpoints.
- [PEFT](docs/guides/peft.md): low-rank layers and an experimental adapter
  framework.

## Development

```bash
uv sync --frozen --group dev
JAX_PLATFORMS=cpu make test-fast
```

The code lives in `src/taktiny/`, with tests in `tests/`. Taktiny is licensed
under [Apache 2.0](LICENSE.md).
