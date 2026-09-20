# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, cast

import optax

from taktiny.nn.base import Pytree


class Optimizer(Pytree):
    """Manage an Optax transformation and its evolving state.

    ``model`` and gradients must have matching parameter-tree structures.
    ``update`` changes this optimizer's state and returns updated parameters;
    it does not store or mutate the model. Extra keyword arguments are
    forwarded to transformations that support them.

    Under ``jax.jit``, pass the optimizer as an argument and return it with
    the updated model. Do not capture it as mutable global or closure state::

        @jax.jit
        def step(model, optimizer, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
            model = optimizer.update(model, grads)
            return model, optimizer, loss

        model, optimizer, loss = step(model, optimizer, x, y)
    """

    def __init__(self, model: Any, tx: optax.GradientTransformation) -> None:
        self.tx = optax.with_extra_args_support(tx)
        self.state = tx.init(model)

    def tree_flatten(
        self,
    ) -> tuple[tuple[optax.OptState], tuple[tuple[str, ...], dict[str, Any]]]:
        # Optax states may mix arrays, EmptyState, and custom registered trees.
        # Explicit children avoid heuristic classification of that structure.
        return (self.state,), (('state',), {'tx': self.tx})

    def update[M](self, model: M, grad: M, **extra_args: Any) -> M:
        """Apply gradients, advance optimizer state, and return the new model."""
        # Optax's aliases omit custom registered PyTrees such as Module.
        params = cast(optax.Params, model)
        grads = cast(optax.Updates, grad)
        updates, state = self.tx.update(
            grads, self.state, params, **extra_args
        )
        # Optax preserves the parameter tree but annotates its result broadly.
        model = cast(M, optax.apply_updates(params, updates))
        self.state = state
        return model


__all__ = ['Optimizer']
