# Saving and loading models

Taktiny gives you the weights; you choose where to store them. A
{py:class}`Module<taktiny.nn.Module>` exposes its parameters through
{py:meth}`state_dict()<taktiny.nn.Module.state_dict>` and
{py:meth}`flat_state_dict()<taktiny.nn.Module.flat_state_dict>`. Both return
ordinary PyTrees, so you can save them with Orbax or convert them for another
checkpoint format.

## State Dictionaries

`state_dict()` follows the module tree. For this model:

```python
from taktiny import nn


class MLP(nn.Module):
    def __init__(self, *, rngs: nn.Rngs):
        self.input = nn.Linear(32, 64, rngs=rngs)
        self.output = nn.Linear(64, 4, rngs=rngs)


model = MLP(rngs=nn.Rngs(0))
state = model.state_dict()
```

`state` has this structure:

```
{
    "input": {
        "kernel": ...,
        "bias": ...,
    },
    "output": {
        "kernel": ...,
        "bias": ...,
    },
}
```

Only parameter values are included. Save the constructor settings needed to
rebuild the model separately; layer sizes, activation choices, and other static
configuration are not part of `state_dict()`.

## Saving with Orbax

Orbax can save the state dictionary directly. The checkpoint path is a directory
that must not already exist:

```python
from pathlib import Path
import orbax.checkpoint as ocp

checkpoint_dir = Path("checkpoints/model").resolve()
checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(checkpoint_dir, model.state_dict())
```

Use a new directory for each checkpoint, such as `checkpoints/step_1000`.
`StandardCheckpointer` writes asynchronously; leaving the `with` block waits
for the save to finish.

## Loading a Model

Build the same architecture, then restore into it. The new model's state tells
Orbax which shapes, types, and array placement to restore:

```python
restored_model = MLP(rngs=nn.Rngs(1))

with ocp.StandardCheckpointer() as checkpointer:
    restored_state = checkpointer.restore(
        checkpoint_dir,
        target=restored_model.state_dict(),
    )

restored_model.load_state_dict(restored_state)
```

The restored values replace the fresh initialization. The model and checkpoint
must have compatible parameter names, shapes, and types.

### Restore without allocating initialized weights

The example above briefly holds both the newly initialized weights and the
restored weights. For a large model, use `jax.eval_shape` to build an abstract
model instead:

```python
import jax

abstract_model = jax.eval_shape(lambda: MLP(rngs=nn.Rngs(0)))

with ocp.StandardCheckpointer() as checkpointer:
    restored_state = checkpointer.restore(
        checkpoint_dir,
        target=abstract_model.state_dict(),
    )

abstract_model.load_state_dict(restored_state)
restored_model = abstract_model
```

`jax.eval_shape` traces the constructor without allocating its parameter
arrays. Orbax then creates the concrete weights during restoration, avoiding
an extra initialized copy. Keep checkpoint I/O outside `jax.jit`; for a
sharded model, the abstract restore target must also specify the intended
sharding (see the [SPMD guide](spmd.md)).

## Partial State Loading

`load_state_dict()` updates only the parameters present in the supplied mapping.
You can also pass `include` to select parameter paths by full regex match. For
example, restore the input layer while keeping a new output layer:

```python
partial_model = MLP(rngs=nn.Rngs(2))
partial_model.load_state_dict(restored_state, include=[r"input\..*"])
```

`partial_model.output` keeps its initial values. The same approach works when
reusing an encoder and replacing a classifier head.

## Flat State Dictionaries

`flat_state_dict()` returns the same parameter values with dotted path names:

```python
state = model.flat_state_dict()
```

```
{
    "input.kernel": ...,
    "input.bias": ...,
    "output.kernel": ...,
    "output.bias": ...,
}
```

This form is useful for inspecting weights or exchanging them with formats that
use a flat name-to-array mapping:

```python
for name, value in model.flat_state_dict().items():
    print(name, value.shape, value.dtype)
```

Load a flat mapping with:

```python
model.load_flat_state_dict(state)
```

As with `load_state_dict()`, omitted parameters are left unchanged.

## Saving Selected Parameters

Select one part of the example model by its dotted parameter paths:

```python
input_state = model.flat_state_dict(include=[r"input\..*"])
```

Save the selection to its own checkpoint directory:

```python
with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(Path("checkpoints/input").resolve(), input_state)
```

To load those values into a compatible model:

```python
partial_model.load_flat_state_dict(input_state, include=[r"input\..*"])
```

The same `include` argument works with hierarchical `state_dict()`. `None`
selects all parameters, while `[]` selects none. A nonempty pattern list that
matches no model parameter raises `ValueError`.

## Converting Between Weight Formats

Flat names make weight conversion easier. Map source names to the names expected
by your model before loading:

```python
state = external_weights()

converted = {
    name.replace("dense.weight", "dense.kernel"): value
    for name, value in state.items()
}

model.load_flat_state_dict(converted)
```

Renaming is enough only when the array shapes and layouts already match. Some
formats also require transposing, splitting, or combining weights.

## Sharded Models

`state_dict()` retains the model's sharded JAX arrays. To restore their intended
placement, build the target model with the desired mesh and sharding rules
first (see the [SPMD guide](spmd.md)):

```python
import jax

with jax.set_mesh(mesh):
    model = build_sharded_model()

    with ocp.StandardCheckpointer() as checkpointer:
        state = checkpointer.restore(
            checkpoint_path,
            target=model.state_dict(),
        )

    model.load_state_dict(state)
```

The initialized state supplies the expected PyTree structure and target array
sharding. In multi-process programs, all participating processes must call the
checkpoint operation consistently.

## Quantized Parameters

Quantized parameters use the same state-dictionary API. For example, an
`nn.Linear(..., quant="int8")` layer stores a quantized weight value in its
state. Save that state as above; when restoring, construct the target layer
with the same quantization configuration and pass its `state_dict()` as the
Orbax restore target. This supplies the expected custom PyTree structure for
the quantized weight.

## Hierarchical vs. Flat State

Use `state_dict()` with `load_state_dict()` for ordinary model checkpoints. Use
`flat_state_dict()` with `load_flat_state_dict()` when you need to inspect,
filter, or rename individual weights. Both represent the same parameters; only
their keys differ.
