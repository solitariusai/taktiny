# `Optimizer` API

`Optimizer` holds an Optax transformation and its state. Call `update` with
matching model and gradient trees to obtain the updated model. The optimizer
advances its own state; it does not store the model.

```{eval-rst}
.. autoclass:: taktiny.takt.optimizer.Optimizer
   :members: update
```

## Example

Pass the optimizer into a compiled training step and return it alongside the
model so its updated state is available to the next step.

```python
import jax
import jax.numpy as jnp
import optax

from taktiny import nn
from taktiny.takt import Optimizer

model = nn.Linear(2, 1, rngs=nn.Rngs(0))
optimizer = Optimizer(model, optax.adam(1e-3))


def loss_fn(model, x, y):
    return jnp.mean((model(x) - y) ** 2)


@jax.jit
def step(model, optimizer, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    model = optimizer.update(model, grads)
    return model, optimizer, loss


x = jnp.ones((4, 2))
y = jnp.zeros((4, 1))
for _ in range(10):
    model, optimizer, loss = step(model, optimizer, x, y)
```

Choose the update algorithm, learning-rate schedule, and gradient transforms
through Optax. Extra keyword arguments passed to `update` are forwarded to
transformations that support them.
