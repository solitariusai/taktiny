# SPMD & Distributed Sharding

Sharding describes how an array is distributed across devices. You still write
operations on the whole array; JAX executes the computation using its shards
and handles the required communication.

Taktiny uses JAX's `Mesh`, `PartitionSpec`, and `NamedSharding`. Logical axis
names let a model describe its dimensions without hard-coding a device layout
into every layer.

## Mesh axes and array dimensions

| Object | Meaning | Example |
| --- | --- | --- |
| `Mesh` | An arrangement of devices with named axes. | A 2 × 2 mesh named `("data", "model")`. |
| `PartitionSpec` | Which mesh axes partition each array dimension. | `P(None, "model")` partitions the second dimension. |
| `NamedSharding` | A spec attached to a particular mesh. | `NamedSharding(mesh, P("data", None))`. |
| `axis_names` | Logical names for a parameter's dimensions. | `("input", "output")`. |

In a spec, `None` leaves that array dimension unpartitioned. Mesh axes unused
by the entire spec replicate the array. For example, `P(None, "model")`
partitions a matrix's columns along `model` and replicates it along `data`.
`P()` requests full replication.

The spec describes array dimensions, not individual devices. Each partitioned
dimension must be divisible by the product of the sizes of its assigned mesh axes.

## A complete four-device example

To try a 2 × 2 mesh without an accelerator, save the Python block below as
`sharding_example.py` and run:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=4' uv run python sharding_example.py
```

Set the environment before JAX initializes its backend. These are four logical
CPU devices for testing, not four physical accelerators.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from taktiny import nn

assert len(jax.devices()) == 4, "Run this example with four JAX devices"
mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("data", "model"))

with jax.set_mesh(mesh):
    layer = nn.Linear(
        8, 12,
        partition_spec=P(None, "model"),
        rngs=nn.Rngs(0),
    )
    x = jax.device_put(
        np.ones((4, 8), dtype=np.float32),
        NamedSharding(mesh, P("data", None)),
    )
    forward = jax.jit(
        lambda model, inputs: model(inputs),
        out_shardings=NamedSharding(mesh, P("data", "model")),
    )
    y = forward(layer, x)

assert layer.kernel.value.shape == (8, 12)
assert layer.kernel.value.sharding.spec == P(None, "model")
assert layer.bias.value.sharding.spec == P("model")
assert y.shape == (4, 12)
assert y.sharding.spec == P("data", "model")
```

The kernel's output dimension and the bias are partitioned across `model`.
Inputs are partitioned across `data`. The requested output layout uses both:
each device holds part of the batch and part of the output features.

The remaining Python blocks build on this example's imports and `mesh`.

## Name dimensions, then choose their placement

Use `map_logical_axis_names` to map model-level names to mesh axes:

```python
from taktiny.utils.spmd import map_logical_axis_names

with jax.set_mesh(mesh), map_logical_axis_names({
    "input": None,
    "output": "model",
}):
    logical_layer = nn.Linear(
        8, 12,
        axis_names=("input", "output"),
        rngs=nn.Rngs(1),
    )

assert logical_layer.kernel.axis_names == ("input", "output")
assert logical_layer.kernel.partition_spec == P(None, "model")
assert logical_layer.kernel.value.sharding.spec == P(None, "model")
```

Taktiny preserves the original `axis_names` and stores the resolved mesh spec
separately. For `Linear`, the bias uses the trailing output axis names and
spec entries.

The mapping applies when the initializer or parameter is created. Leaving the
context restores previous rules; it does not move existing arrays. Changing
rules later does not automatically reshard an existing model either.

### When both arguments are supplied

`axis_names` takes precedence over `partition_spec`. The explicit spec is not
a fallback for individual unmapped dimensions:

```python
with jax.set_mesh(mesh), map_logical_axis_names({"output": "model"}):
    overridden = nn.Linear(
        8, 12,
        axis_names=("input", "output"),
        partition_spec=P("data", None),
        rngs=nn.Rngs(2),
    )

assert overridden.kernel.partition_spec == P(None, "model")
```

Here `input` is unmapped, so it resolves to `None`, not the explicit spec's
`"data"`. If no logical names match any rules, every dimension resolves to
`None` and the parameter is replicated on the active mesh.

To use an explicit spec, omit `axis_names`. To request replication while keeping
logical names, map those names to `None` when creating the parameter.

## Rule order and scope

Inspect a mapping without creating an array using `logical_to_mesh_axes`:

```python
from taktiny.utils.spmd import logical_to_mesh_axes

assert logical_to_mesh_axes(
    ("batch", "feature"),
    rules=(("batch", "data"), ("feature", "model")),
) == P("data", "model")

assert logical_to_mesh_axes(
    ("input", "output"),
    rules=(("output", "model"), ("input", "model")),
) == P(None, "model")
```

Rules are considered in order. A mesh axis cannot partition two dimensions of
the same array, so the first available assignment wins. In the second example,
`output` claims `model`, leaving `input` unpartitioned. Repeating a non-`None`
logical name within one array raises an error.

A logical name can map to a tuple of mesh axes. For example,
`{"feature": ("data", "model")}` partitions one dimension across both axes;
its size must be divisible by four on this mesh.

Prefer `with map_logical_axis_names(...)` for temporary rules. Nested blocks
prepend their rules, giving them priority over outer rules, and restore the
previous rules on exit.

For longer-lived configuration, `set_logical_axis_rules(rules)` replaces the
current rule sequence; `get_logical_axis_rules()` reads it. The storage is
thread-local, not shared across threads or processes.

Calling `map_logical_axis_names(...)` without `with` also changes the rules
immediately, but prepends them instead of replacing them. Merely constructing
that context object has this effect; it does not wait until context entry.

## When initialization is sharded

Layers such as `Linear` wrap their initializers with `with_logical_partitioning`
when sharding arguments are supplied. With an active `jax.set_mesh(mesh)`
context and a resolved spec, the wrapper compiles the initializer with
`jax.jit(..., out_shardings=spec)`. Its returned array is already sharded;
you do not need a separate post-initialization placement step.

This describes the output layout, not every temporary allocation inside the
initializer. Invalid specs, incompatible dimensions, and other compilation
errors from this wrapper are not silently converted into replication.

The wrapper has a fallback for a missing active mesh: it calls the original
initializer. A parameter can therefore retain sharding metadata without its
array having that layout. Construct parameters inside the mesh context when
you want placement at initialization, and check the array's `.sharding`, not
only its metadata.

You can wrap a custom initializer directly:

```python
from taktiny.utils.spmd import with_logical_partitioning

initialize = with_logical_partitioning(
    jax.nn.initializers.normal(),
    axis_names=("input", "output"),
)
with jax.set_mesh(mesh), map_logical_axis_names({"output": "model"}):
    weight = initialize(jax.random.key(3), (8, 12), jnp.float32)

assert weight.sharding.spec == P(None, "model")
```

Use the usual initializer signature `(key, shape, dtype)`. Logical rules are
resolved when the wrapped initializer is called, not when it is wrapped.

## Match names to the parameter shape

A parameter needs one logical axis entry per array dimension. A feature shape
with several dimensions therefore needs several entries, not one name:

```python
with jax.set_mesh(mesh), map_logical_axis_names({"output": "model"}):
    nd_layer = nn.Linear(
        (2, 4), (3, 4),
        axis_names=("input_group", "input_feature", "output_group", "output"),
        rngs=nn.Rngs(4),
    )

assert nd_layer.kernel.value.shape == (2, 4, 3, 4)
assert nd_layer.kernel.partition_spec == P(None, None, None, "model")
assert nd_layer.bias.value.shape == (3, 4)
assert nd_layer.bias.partition_spec == P(None, "model")
```

Check each layer's parameter layout before copying a spec: convolution kernels
also have spatial dimensions, and bilinear kernels have dimensions for both
inputs before the output dimensions.

## Inspect placement and keep data loading separate

For an ordinary array-backed parameter, `parameter.value.sharding` reports its
actual placement. `array.addressable_shards` exposes the shards accessible
from the current process, including their global indices and local arrays.
Replication can mean several devices hold the same global slice.

Use `jax.device_put(array, NamedSharding(mesh, spec))` to explicitly place or
reshard an existing array. Updating parameter metadata alone does not move its
value.

Parameter specs do not automatically assign layouts to input batches. Place
batches explicitly, as in the first example, and choose output layouts at the
JIT boundary when your application requires them.

Likewise, `DataLoader(shard_index=..., shard_count=...)` partitions source
records between processes; it does not produce device-sharded JAX arrays.
These examples use one process with four devices. Multi-process execution also
requires JAX distributed initialization and correct construction of global
arrays from process-local data; logical-axis rules do not perform that setup.

See the [SPMD API reference](../api/utils/spmd.md) for mapping helpers and the
[data guide](data.md) for input-pipeline sharding.
