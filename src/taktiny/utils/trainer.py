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

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import qwix

from taktiny.nn.base import Module, Parameter
from taktiny.utils.typing import PyTree


def _is_trainable_value(value: Any) -> bool:
    if isinstance(value, qwix.QArray):
        return False

    if not hasattr(value, 'dtype'):
        return False

    if not jnp.issubdtype(value.dtype, jnp.inexact):
        return False

    return value.dtype != getattr(jnp, 'float8_e4m3fn', None)


def _parameter_labels(params: PyTree) -> PyTree:
    """Build Optax labels from explicit Taktiny parameter metadata."""
    if not isinstance(params, Module):
        return jax.tree.map(
            lambda value: (
                'trainable'
                if _is_trainable_value(value)
                else 'frozen'
            ),
            params,
        )

    def label_parameter(parameter: Parameter) -> PyTree:
        trainable = (
            getattr(parameter, 'trainable', True)
            and _is_trainable_value(parameter.value)
        )
        label = 'trainable' if trainable else 'frozen'
        return jax.tree.map(lambda _: label, parameter)

    return jax.tree.map(
        label_parameter,
        params,
        is_leaf=lambda value: isinstance(value, Parameter),
    )


def _partition_params(params: PyTree, labels: PyTree) -> tuple[PyTree, PyTree]:
    def label_value(label: PyTree) -> PyTree:
        return label.value if isinstance(label, Parameter) else label

    def empty_value(value: PyTree) -> PyTree:
        if isinstance(value, Parameter):
            empty = object.__new__(type(value))
            empty.__dict__.update(value.__dict__)
            empty._value = None
            return empty
        return None

    trainable = jax.tree.map(
        lambda value, label: (
            value
            if label_value(label) == 'trainable'
            else empty_value(value)
        ),
        params,
        labels,
        is_leaf=lambda value: isinstance(value, Parameter),
    )
    frozen = jax.tree.map(
        lambda value, label: (
            empty_value(value)
            if label_value(label) == 'trainable'
            else value
        ),
        params,
        labels,
        is_leaf=lambda value: isinstance(value, Parameter),
    )
    return trainable, frozen


def _combine_params(trainable: PyTree, frozen: PyTree) -> PyTree:
    def combine_value(
        trainable_value: PyTree,
        frozen_value: PyTree,
    ) -> PyTree:
        if isinstance(trainable_value, Parameter):
            if trainable_value.value is None:
                return frozen_value
            return trainable_value
        return frozen_value if trainable_value is None else trainable_value

    return jax.tree.map(
        combine_value,
        trainable,
        frozen,
        is_leaf=lambda value: value is None or isinstance(value, Parameter),
    )


def _global_grad_norm(grads: PyTree) -> jax.Array:
    """Compute the L2 norm of a gradient tree without full-size copies.

    Each leaf is contracted with itself through ``jnp.vdot``, which XLA lowers
    to a fused dot with float32 accumulation; only a per-leaf scalar is
    produced. No squared or cast copy of the gradient tree is ever allocated,
    so the peak memory of the norm is negligible even for huge models.
    Frozen leaves (``None``) are skipped.
    """
    norm_sq = jax.tree.reduce(
        lambda acc, grad: (
            acc
            if grad is None
            else acc + jnp.vdot(grad, grad).astype(jnp.float32)
        ),
        grads,
        initializer=jnp.asarray(0.0, dtype=jnp.float32),
        is_leaf=lambda value: value is None,
    )
    return jnp.sqrt(norm_sq)


def _zeros_like_grads(params: PyTree) -> PyTree:
    """Build a zero gradient accumulator matching a trainable parameter tree.

    Frozen (``None``) leaves are preserved so the accumulator mirrors the
    gradient tree returned by ``jax.value_and_grad``.
    """
    return jax.tree.map(
        lambda value: (
            None if value is None else jnp.zeros_like(value)
        ),
        params,
        is_leaf=lambda value: value is None,
    )


def _accumulate_grads(total: PyTree, value: PyTree) -> PyTree:
    """Sum two gradient trees, preserving frozen (``None``) leaves."""
    return jax.tree.map(
        lambda a, b: (
            None if a is None or b is None else a + b
        ),
        total,
        value,
        is_leaf=lambda value: value is None,
    )


def _copy_tree(tree: PyTree) -> PyTree:
    """Return a fresh structure with independent array leaves."""
    return jax.tree.map(
        lambda value: (
            value.copy()
            if hasattr(value, 'dtype')
            else copy.deepcopy(value)
        ),
        tree,
    )


def _ema_update(
    ema: PyTree,
    params: PyTree,
    decay: float,
) -> PyTree:
    """Blend the EMA tree toward the current weights, in the weights' dtype.

    Frozen (``None``) leaves are passed through unchanged.
    """
    def blend(ema_value: Any, param_value: Any) -> Any:
        if ema_value is None or param_value is None:
            return param_value
        return ema_value * decay + param_value * (1.0 - decay)

    return jax.tree.map(
        blend,
        ema,
        params,
        is_leaf=lambda value: value is None,
    )


def _tree_shardings(tree: PyTree) -> PyTree:
    return jax.tree.map(
        lambda value: (
            value.sharding if isinstance(value, jax.Array) else None
        ),
        tree,
    )


def _parameter_mesh(params: PyTree) -> jax.sharding.Mesh | None:
    for value in jax.tree.leaves(params):
        sharding = getattr(value, 'sharding', None)
        if isinstance(sharding, jax.sharding.NamedSharding):
            return sharding.mesh
    return None


def _sharding_mesh(sharding: PyTree) -> jax.sharding.Mesh | None:
    for value in jax.tree.leaves(sharding):
        if isinstance(value, jax.sharding.NamedSharding):
            return value.mesh
    return None


def _uses_multiple_devices(tree: PyTree) -> bool:
    for value in jax.tree.leaves(tree):
        sharding = getattr(value, 'sharding', None)
        if sharding is not None and len(sharding.device_set) > 1:
            return True
    return False


def _validate_parameter_placement(
    params: PyTree,
    batch_mesh: jax.sharding.Mesh | None,
) -> None:
    parameter_mesh = _parameter_mesh(params)
    if (
        parameter_mesh is not None
        and batch_mesh is not None
        and parameter_mesh != batch_mesh
    ):
        raise ValueError(
            'Model parameters and batches must use the same device mesh.'
        )
    if batch_mesh is None or batch_mesh.size <= 1:
        return
    if parameter_mesh is not None or _uses_multiple_devices(params):
        return
    raise ValueError(
        'Multi-device batch sharding requires pre-sharded model parameters. '
        'Load the model with the same mesh and parameter sharding rules; '
        'Trainer will not replicate an unsharded model automatically.'
    )


def _place_trainable_params(
    tree: PyTree,
    mesh: jax.sharding.Mesh | None,
) -> PyTree:
    if mesh is None:
        return tree

    replicated = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )

    def place(value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        if isinstance(value.sharding, jax.sharding.NamedSharding):
            return value
        return jax.device_put(value, replicated)

    return jax.tree.map(place, tree)


def _place_optimizer_state(
    tree: PyTree,
    mesh: jax.sharding.Mesh | None,
) -> PyTree:
    if mesh is None or mesh.size <= 1:
        return tree

    replicated = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )

    def place(value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        if isinstance(value.sharding, jax.sharding.NamedSharding):
            return value
        if value.ndim == 0:
            return jax.device_put(value, replicated)
        raise ValueError(
            'A non-scalar optimizer state did not inherit parameter sharding. '
            'Provide an optimizer whose parameter-shaped state preserves '
            'input shardings.'
        )

    return jax.tree.map(place, tree)


def _prefetch[T](
    iterable: Iterable[T],
    place: Callable[[T], T],
    size: int,
) -> Iterator[T]:
    """Place a bounded number of batches ahead of consumption."""
    iterator = iter(iterable)
    if size == 0:
        for value in iterator:
            yield place(value)
        return

    queue = deque()
    for _ in range(size):
        try:
            queue.append(place(next(iterator)))
        except StopIteration:
            break

    while queue:
        yield queue.popleft()
        try:
            queue.append(place(next(iterator)))
        except StopIteration:
            pass


def _format_iteration_time(seconds: float) -> str:
    if seconds < 1:
        return f'{seconds * 1000:.1f} ms/it'

    if seconds < 60:
        return f'{seconds:.1f} s/it'

    return f'{seconds / 60:.1f} min/it'


