# Taktiny

A Deep Learning framework built on **JAX**, featuring object-oriented PyTree modeling, parameter-efficient fine-tuning (PEFT), SPMD sharding, post-training quantization, and training lifecycle utilities.

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started/installation
getting_started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guides

guides/spmd
guides/peft
guides/data
guides/trainer
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorial/linear_regression
tutorial/image_classification
tutorial/gan
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/nn
api/takt
api/data
api/trainer
api/utils
```

## Features

- **Object-Oriented PyTrees**: Stateful `nn.Module` and `nn.Parameter` objects registered as pure JAX PyTrees.
- **SPMD Sharding**: Direct mesh integration, logical axis name mapping, and eager partition specs.
- **PEFT Adapters via `Takt`**: Native support for LoRA, DoRA, AdaLoRA, LoHa, LoKr, and VeRA with shared projection support.
- **Quantization**: Built-in weight-only PTQ integrated with `qwix` (INT4, FP8, grouped block scaling).
- **Data Pipeline**: Fast token packing, streaming datasets, and Grain-compatible loaders.
- **Full Trainer Lifecycle**: Checkpointing with Orbax, Optax optimizer management, and Wandb/TensorBoard callbacks.
