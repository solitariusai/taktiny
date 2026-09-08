# SPMD & Distributed Sharding

Taktiny provides native support for JAX SPMD (Single Program, Multiple Data) sharding via `NamedSharding` and `PartitionSpec`.

## Eager vs Logical Partitioning

Layers such as `nn.Linear` and `nn.Bilinear` accept `partition_spec` and `axis_names`:

```python
from jax.sharding import Mesh, PartitionSpec as P
from taktiny import nn

# When an active JAX Mesh is in context, parameters are placed
# eagerly onto the device mesh upon initialization.
layer = nn.Linear(
    in_features=128,
    out_features=256,
    partition_spec=P("fsdp", "tp"),
    rngs=nn.Rngs(0),
)
```

## Logical Axis Mapping

You can map model-level logical axis names (e.g., `embed`, `mlp`, `heads`) to mesh axis names (`data`, `fsdp`, `tp`):

```python
from taktiny.utils.spmd import set_logical_axis_rules

rules = (
    ("batch", "data"),
    ("mlp", "tp"),
    ("embed", "fsdp"),
)
set_logical_axis_rules(rules)
```
