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

import re
from collections.abc import Sequence
from typing import Any, cast

import jax
import optax

from taktiny.nn.base import Module, Parameter, Pytree


def _leaf_names(tree: Any) -> tuple[str, ...]:
    names = []
    for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]:
        current = tree
        parts = []
        for key in path:
            if isinstance(key, jax.tree_util.FlattenedIndexKey):
                children, metadata = current.tree_flatten()
                if isinstance(current, (Module, Pytree)) and not isinstance(current, Parameter):
                    parts.append(metadata[0][key.key])
                elif not isinstance(current, Parameter):
                    parts.append(str(key.key))
                current = children[key.key]
            elif isinstance(key, jax.tree_util.GetAttrKey):
                parts.append(key.name)
                current = getattr(current, key.name)
            elif isinstance(key, jax.tree_util.DictKey):
                parts.append(str(key.key))
                current = current[key.key]
            else:
                parts.append(str(key.idx))
                current = current[key.idx]
        names.append('.'.join(parts))
    return tuple(names)


class Optimizer(Pytree):
    r"""Manage an Optax transformation and its evolving state.

    ``model`` and gradients must have matching parameter-tree structures.
    ``update`` changes this optimizer's state and returns updated parameters;
    it does not store or mutate the model. Extra keyword arguments are
    forwarded to transformations that support them.

    ``include`` selects leaves by full regex match against dot-separated paths
    (e.g. ``[r'encoder\..*\.kernel']``). None keeps the original full-tree
    behavior; an empty sequence freezes all leaves. Nonempty patterns matching
    no leaves raise ValueError. Selection is fixed at initialization. With
    selection enabled, Optax receives a flat dictionary keyed by selected paths;
    optimizer masks and tree-shaped extra arguments must use that structure.
    Unselected leaves have no optimizer state and receive no updates or decay.
    This does not prevent gradient computation for unselected leaves.

    Under ``jax.jit``, pass the optimizer as an argument and return it with
    the updated model. Do not capture it as mutable global or closure state::

        @jax.jit
        def step(model, optimizer, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
            model = optimizer.update(model, grads)
            return model, optimizer, loss

        model, optimizer, loss = step(model, optimizer, x, y)
    """

    def __init__(
        self, model: Any, tx: optax.GradientTransformation,
        *, include: Sequence[str] | None = None,
    ) -> None:
        self.tx = optax.with_extra_args_support(tx)
        self._selection: tuple[tuple[int, str], ...] | None = None
        self._structure: jax.tree_util.PyTreeDef | None = None
        if include is not None:
            if isinstance(include, str) or not isinstance(include, Sequence):
                raise TypeError('include must be a sequence of regex strings or None')
            if any(not isinstance(pattern, str) for pattern in include):
                raise TypeError('include must contain regex strings')
            patterns = tuple(re.compile(pattern) for pattern in include)
            names = _leaf_names(model)
            if len(set(names)) != len(names):
                raise ValueError('Parameter paths must be unique after dot-separated conversion')
            self._selection = tuple((i, name) for i, name in enumerate(names)
                                    if any(pattern.fullmatch(name) for pattern in patterns))
            if patterns and not self._selection:
                raise ValueError('include patterns matched no parameter leaves')
            leaves, self._structure = jax.tree.flatten(model)
            model = {name: leaves[i] for i, name in self._selection}
        self.state = self.tx.init(model)

    def tree_flatten(
        self,
    ) -> tuple[tuple[optax.OptState], tuple[tuple[str, ...], dict[str, Any]]]:
        # Optax states may mix arrays, EmptyState, and custom registered trees.
        # Explicit children avoid heuristic classification of that structure.
        return (self.state,), (('state',), {
            'tx': self.tx, '_selection': self._selection, '_structure': self._structure,
        })

    def update[M](self, model: M, grad: M, **extra_args: Any) -> M:
        """Apply gradients, advance optimizer state, and return the new model."""
        if self._selection is not None:
            leaves, structure = jax.tree.flatten(model)
            gradients, grad_structure = jax.tree.flatten(grad)
            if structure != self._structure or grad_structure != structure:
                raise ValueError('Model and gradient structures must match initialization')
            if not self._selection:
                return model
            selected_params = {name: leaves[i] for i, name in self._selection}
            selected_grads = {name: gradients[i] for i, name in self._selection}
            updates, state = self.tx.update(selected_grads, self.state, selected_params, **extra_args)
            selected = cast(dict[str, Any], optax.apply_updates(selected_params, updates))
            for i, name in self._selection:
                leaves[i] = selected[name]
            result = jax.tree.unflatten(structure, leaves)
            self.state = state
            return cast(M, result)
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
