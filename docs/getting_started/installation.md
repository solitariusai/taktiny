# Installation

Taktiny requires Python **3.12+** and **JAX 0.10.2+**.

## Installing with `uv`

The recommended way to install and develop Taktiny is using [uv](https://docs.astral.sh/uv/):

```bash
# Clone the repository
git clone https://github.com/solitariusai/taktiny.git -b experiment
cd taktiny

# Install development dependencies and virtual environment
uv sync --group dev
```

## Installing with `pip`

You can also install Taktiny into an existing virtual environment with pip:

```bash
pip install -e .
```

### Optional Extras

Taktiny provides optional extras for experiment tracking:

```bash
# TensorBoard support
pip install -e ".[tensorboard]"

# Weights & Biases support
pip install -e ".[wandb]"

# All reporting tools
pip install -e ".[reporting]"
```
