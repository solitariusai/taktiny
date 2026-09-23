# Installation

Taktiny requires Python **3.12+** and **JAX 0.10.2+**.

## Install from PyPI

With [uv](https://docs.astral.sh/uv/):

```bash
uv add taktiny
```

Or with pip in an existing virtual environment:

```bash
pip install taktiny
```

The default installation supports CPU.

## Accelerator support

To use an accelerator, choose the extra that matches your system:

| Hardware | Extra |
| --- | --- |
| NVIDIA GPU with CUDA 12 | `cuda12` |
| NVIDIA GPU with CUDA 13 | `cuda13` |
| AMD GPU with an existing ROCm 7 installation | `rocm7-local` |
| Google Cloud TPU VM | `tpu` |

For example, to install the CUDA 12 extra:

```bash
uv add "taktiny[cuda12]"
# Or, with pip:
pip install "taktiny[cuda12]"
```

Replace `cuda12` with the extra for your hardware. Choose only one accelerator
extra; `rocm7-local` does not install ROCm itself. Check the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
supported platforms and driver requirements. These extras must be present in
the Taktiny release you install; for unreleased extras, use the source version
below or install the appropriate JAX extra separately.

To confirm that JAX sees your devices, run:

```bash
python -c "import jax; print(jax.devices())"
```

## Install from source

The repository's main branch is `master`. It contains the latest more-stable
source code, while `experiment` has the newest, less-tested commits. To install
from `master` instead of PyPI:

```bash
uv add git+https://github.com/solitariusai/taktiny.git@master
# Or, with pip:
pip install git+https://github.com/solitariusai/taktiny.git@master
```

Replace `@master` with `@experiment` if you want the newest development code.
You can request an accelerator extra from either branch, for example:

```bash
uv add "taktiny[cuda12] @ git+https://github.com/solitariusai/taktiny.git@master"
```

To work on Taktiny itself, clone the repository and install its development
dependencies:

```bash
git clone --branch master https://github.com/solitariusai/taktiny.git
cd taktiny
uv sync --group dev
```

Use `--branch experiment` to develop against the newest commits instead.
