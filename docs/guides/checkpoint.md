# Saving and loading models

Taktiny separates **model state** from **serialization**.

A `Module` exposes its parameters through `state_dict()` and `flat_state_dict()`, while the actual storage format is left to the caller. This means model weights can be stored with Orbax or converted to another checkpoint format without tying the model API to a particular serialization backend.

## State Dictionaries

Every Taktiny `Module` provides a hierarchical state dictionary:

```python
state = model.state_dict()
```

The returned dictionary follows the structure of the module tree.

For example, a model such as:

```python
class MLP(nn.Module):
    def __init__(self, *, rngs: nn.Rngs):
        self.input = nn.Linear(32, 64, rngs=rngs)
        self.output = nn.Linear(64, 4, rngs=rngs)
```

produces a state dictionary conceptually shaped like:

```python
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

The state dictionary contains parameter values, not the Python model definition itself.

Attributes such as layer sizes, activation choices, constructor arguments, and other static model configuration should therefore be stored separately when they are required to reconstruct the architecture.

## Saving with Orbax

Orbax can serialize a Taktiny state dictionary directly:

```python
import os
import orbax.checkpoint as ocp

path = os.path.abspath("./checkpoints/model")

state = model.state_dict()

with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(
        path,
        args=ocp.args.StandardSave(state),
    )
```

The checkpoint contains the model state at the time `state_dict()` was called.

A checkpoint path should normally represent a new checkpoint directory rather than an existing checkpoint that is overwritten in place.

## Loading a Model

Model construction and state restoration are separate operations.

First construct a compatible model:

```python
model = MLP(rngs=nn.Rngs(0))
```

Then restore the checkpoint using its current state as the target structure:

```python
import os
import orbax.checkpoint as ocp

path = os.path.abspath("./checkpoints/model")

with ocp.StandardCheckpointer() as checkpointer:
    state = checkpointer.restore(
        path,
        target=model.state_dict(),
    )

model.load_state_dict(state)
```

The initialization values produced when constructing the model are replaced by the restored values.

The model should be constructed with the same parameter structure expected by the checkpoint.

## Partial State Loading

`load_state_dict()` updates parameters that are present in the supplied state dictionary.

This makes it possible to restore only part of a model:

```python
state = {
    "input": saved_state["input"],
}

model.load_state_dict(state)
```

Parameters not present in the supplied state remain unchanged.

This can be useful when reusing part of a pretrained network, replacing a prediction head, or initializing only selected submodules.

For example:

```python
pretrained = load_weights(...)

model = Classifier(
    num_classes=10,
    rngs=nn.Rngs(0),
)

model.load_state_dict({
    "encoder": pretrained["encoder"],
})
```

The encoder is restored while the classifier head keeps its newly initialized parameters.

## Flat State Dictionaries

Taktiny also provides a flattened representation:

```python
state = model.flat_state_dict()
```

Instead of nested dictionaries, parameter paths become string keys:

```python
{
    "input.kernel": ...,
    "input.bias": ...,
    "output.kernel": ...,
    "output.bias": ...,
}
```

This representation is useful when working with checkpoint formats or model repositories that represent weights as a flat mapping from parameter names to arrays.

Inspecting weights is also straightforward:

```python
for name, value in model.flat_state_dict().items():
    print(name, value.shape, value.dtype)
```

A flat state dictionary can be restored with:

```python
model.load_flat_state_dict(state)
```

As with `load_state_dict()`, parameters that are not included in the supplied mapping are left unchanged.

## Saving Selected Parameters

Because a flat state dictionary is an ordinary mapping, individual parts of a model can be selected before serialization.

For example:

```python
state = model.flat_state_dict()

encoder_state = {
    name: value
    for name, value in state.items()
    if name.startswith("encoder.")
}
```

The selected state can then be saved independently:

```python
with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(
        os.path.abspath("./encoder"),
        args=ocp.args.StandardSave(encoder_state),
    )
```

The same technique can be used for adapters, heads, embeddings, or any other subset identified by its parameter path.

To restore selected parameters:

```python
model.load_flat_state_dict(encoder_state)
```

This makes weight-only components independent from the full model checkpoint.

## Converting Between Weight Formats

`flat_state_dict()` is generally the most convenient representation when exchanging weights with another framework or checkpoint format.

For example, parameter names can be transformed before loading:

```python
state = external_weights()

converted = {
    name.replace("dense.weight", "dense.kernel"): value
    for name, value in state.items()
}

model.load_flat_state_dict(converted)
```

More complicated mappings can split, combine, transpose, or rename parameters before they are passed to the model.

The model does not need to know where the weights originated. It only consumes the resulting state mapping.

## Sharded Models

Parameters in a Taktiny module may already be distributed across devices using JAX sharding.

The state dictionary preserves the parameter values exposed by the model:

```python
state = model.state_dict()
```

When restoring a distributed model, construct the model under the intended mesh and sharding configuration before restoring the checkpoint:

```python
with mesh:
    model = DistributedModel(rngs=nn.Rngs(0))

    with ocp.StandardCheckpointer() as checkpointer:
        state = checkpointer.restore(
            checkpoint_path,
            target=model.state_dict(),
        )

    model.load_state_dict(state)
```

Using the initialized model as the restore target gives the checkpoint backend the expected PyTree structure and target array placement.

For multi-process JAX programs, checkpoint operations should be coordinated consistently across participating processes.

## Quantized Parameters

Quantized Taktiny modules expose their parameter state through the same state-dictionary interface:

```python
state = quantized_model.state_dict()
```

There is no separate model-saving API for quantized layers.

When custom PyTree parameter values are present, such as quantized weight representations, restoring against a model with the matching parameter structure is recommended:

```python
quantized_model = QuantizedModel(...)

with ocp.StandardCheckpointer() as checkpointer:
    state = checkpointer.restore(
        checkpoint_path,
        target=quantized_model.state_dict(),
    )

quantized_model.load_state_dict(state)
```

This preserves the expected structure of the quantized parameters during restoration.

## Hierarchical vs. Flat State

Use `state_dict()` when saving and restoring a Taktiny model directly. Its nested structure naturally mirrors the module hierarchy.

Use `flat_state_dict()` when parameter names need to be inspected, filtered, renamed, converted, or exchanged with an external weight format.

Both representations contain the parameter values of the model and can be loaded back through their corresponding APIs:

```python
model.load_state_dict(state)
```

or:

```python
model.load_flat_state_dict(flat_state)
```

The choice affects the representation of parameter names, not the model computation itself.
