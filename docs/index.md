# Taktiny

:::{container} taktiny-intro
A deep learning library built on JAX.

Define models with Python modules, transform them with JAX, and choose how to
load data, distribute computation, and run training.
:::

[Installation](getting_started/installation.md) ·
[Quickstart](getting_started/quickstart.md) ·
[API reference](api/nn.md)

:::{container} taktiny-project-note
Taktiny is experimental. APIs may change as the library develops.
:::

## A model is a PyTree

Layers compose as ordinary Python objects. A module can be passed to a JAX
transformation alongside its inputs.

```python
import jax
import jax.numpy as jnp
from taktiny import nn


class MLP(nn.Module):
    def __init__(self, *, rngs: nn.Rngs):
        self.hidden = nn.Linear(16, 32, rngs=rngs)
        self.output = nn.Linear(32, 4, rngs=rngs)

    def __call__(self, x):
        return self.output(jax.nn.relu(self.hidden(x)))


model = MLP(rngs=nn.Rngs(0))
forward = jax.jit(model)

y = forward(jnp.ones((8, 16)))
print(y.shape)  # (8, 4)
```

The [quickstart](getting_started/quickstart.md) continues with data loading,
optimization, and evaluation.

## Find your way

::::{grid} 1 1 2 2
:gutter: 3
:class-container: taktiny-doc-index

:::{grid-item}
### Start here

Set up your environment and train a first model.

- [Install Taktiny](getting_started/installation.md)
- [Build and train a model](getting_started/quickstart.md)
:::

:::{grid-item}
### Guides

Work with the parts of a training pipeline.

- [Saving and Loading Models](guides/checkpoint.md)
- [Parameter-efficient fine-tuning](guides/peft.md)
- [Data loading and preprocessing](guides/data.md)
- [Training and checkpoints](guides/trainer.md)
- [Sharding and parallelism](guides/spmd.md)
:::

:::{grid-item}
### Tutorials

Complete examples, from simple models to image generation.

- [Linear regression](tutorial/linear_regression.md)
- [Image classification](tutorial/image_classification.md)
- [Generative adversarial networks](tutorial/gan.md)
:::

:::{grid-item}
### API reference

Signatures, arguments, and examples by package.

- [Neural network modules](api/nn.md)
- [Adapter injection](api/takt.md)
- [Data transforms and loaders](api/data.md)
- [Trainer and callbacks](api/trainer.md)
- [JAX and quantization utilities](api/utils.md)
:::
::::

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

getting_started/installation
getting_started/quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Maybe Useful

guides/checkpoint
guides/peft
guides/data
guides/trainer
guides/spmd
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Tutorials

tutorial/linear_regression
tutorial/image_classification
tutorial/gan
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

api/nn
api/takt
api/data
api/trainer
api/utils
```
