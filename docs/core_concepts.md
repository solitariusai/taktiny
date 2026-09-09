# Core Concepts

## Modules and Parameters

`nn.Module` subclasses are registered JAX PyTrees. Array state is normally held
inside `nn.Parameter`, while ordinary Python attributes are static PyTree
metadata.

```python
import jax
import jax.numpy as jnp

from taktiny import nn


class Projection(nn.Module):
    def __init__(self, in_features, out_features, *, rngs):
        self.weight = nn.Parameter(
            jax.random.normal(rngs(), (in_features, out_features)) * 0.02
        )

    def __call__(self, x):
        return jnp.matmul(x, self.weight.value)


module = Projection(8, 4, rngs=nn.Rngs(42))
output = jax.jit(module)(jnp.ones((2, 8)))
```

Common state helpers are:

```python
flat_parameters = module.flat_parameter_dict()
flat_arrays = module.flat_state_dict()
nested_state = module.state_dict()
module.load_state_dict(nested_state)
```

The parameter dictionaries contain references to existing `Parameter` objects;
creating them does not copy every weight array.

## Random Keys

`nn.Rngs` owns one JAX PRNG key. Each call advances the stream and returns a
new subkey.

```python
from taktiny import nn

rngs = nn.Rngs(42)
initialization_key = rngs()
dropout_key = rngs()
current_key = rngs.key
```

Generation accepts an integer `seed`. The trainer maintains its own `nn.Rngs`
stream and saves it in resumable checkpoints.

## Module Containers

`nn.List` stores independent module instances and executes no transformation
by itself.

`nn.SeqStack` stacks matching module leaves along a leading layer axis. Its
call receives a scan body and executes layers sequentially with
`jax.lax.scan`. Every module must have the same PyTree structure. Per-layer
values that vary, such as Gemma3 local/full attention settings, must therefore
be array leaves rather than different static metadata.

`nn.Stack` uses `jax.vmap` to apply stacked modules independently. It is not a
decoder loop because vectorized calls do not feed one layer's result into the
next.

`nn.Sequential` calls ordinary modules in order without stacking their
parameters.

## Decoder Execution

`TransformerModel` embeds integer token IDs and runs
`config.num_hidden_layers` decoder blocks. `use_list=True` selects `nn.List`;
`use_list=False` selects `nn.SeqStack`.

`TransformerContext` carries KV caches, cached position state, and the causal
flag. Most users interact with it indirectly through `generate` or
`stream_generate`.

Packed sequences pass two-dimensional `position_ids`. A reset to zero marks a
new example, allowing attention to derive segment boundaries without storing a
quadratic block-diagonal mask in the dataloader.

## Sharding

Parameters carry logical axis names such as `vocab`, `embed`, `heads`, `mlp`,
and `batch`. Passing a JAX `Mesh` plus logical-to-mesh rules to
`Maestro.from_pretrained` places checkpoint arrays with matching
`NamedSharding` values.

```python
import jax
from jax.sharding import Mesh

from taktiny import Maestro

devices = jax.devices()
mesh = Mesh(devices, ("fsdp",))

model = Maestro.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    mesh=mesh,
    sharding_rules=[
        ("batch", "fsdp"),
        ("vocab", "fsdp"),
        ("embed", None),
        ("heads", None),
        ("kv_heads", None),
        ("head_dim", None),
        ("mlp", None),
    ],
)
```

The mesh shape must match the available device count. Sharding reduces
per-device storage only when a parameter axis is mapped across multiple
devices.

## Rematerialization

Implemented causal models expose `enable_remat()`:

```python
model.enable_remat()
```

This checkpoints decoder-layer computation during gradient evaluation,
trading additional computation for lower activation memory. It does not reduce
parameter, optimizer-state, or KV-cache storage, and it is intentionally
configured on the model rather than through `TrainingConfig`.
