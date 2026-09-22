# Data Loading

Data loader and dataset splitting utilities.

## Batch placement

Use `axis_names` for logical axes or `partition_spec` for explicit mesh axes.
Either can be shared across arrays or supplied per batch field. Logical names
override the explicit specification for that field.

```python
import jax
from jax.sharding import PartitionSpec as P
from taktiny.data import DataLoader
from taktiny.utils.spmd import map_logical_axis_names

mesh = jax.make_mesh((jax.device_count(),), ('data',))
with jax.set_mesh(mesh), map_logical_axis_names({'batch': 'data'}):
    dataloader = DataLoader(
        dataset['train'],
        batch_size=512,
        drop_remainder=True,
        axis_names={
            'img': ('batch', 'height', 'width', 'channel'),
            'label': ('batch',),
        },
        partition_spec=P(),
    )
    batch = next(iter(dataloader))
```

The example assumes numeric channels-last images and scalar labels. Logical
rules are resolved at construction; the mesh is captured when creating an
iterator. Placement occurs after batching and retains the iterator's Grain
checkpoint methods. Existing record-shard and worker arguments are unchanged.

```{eval-rst}
.. autoclass:: taktiny.data.DataLoader
   :members:

.. autoclass:: taktiny.data.RandomAccessSource

.. autofunction:: taktiny.data.train_validation_split
```
