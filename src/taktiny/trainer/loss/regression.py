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
"""Regression losses."""

from __future__ import annotations

import jax.numpy as jnp

from taktiny.utils.typing import Array, ArrayLike


def _validate_reduction(reduction: str) -> None:
    if reduction not in {'none', 'sum', 'mean'}:
        raise ValueError('reduction must be "none", "sum", or "mean"')


def _reduce(losses: Array, reduction: str) -> Array:
    if reduction == 'none':
        return losses
    if reduction == 'sum':
        return jnp.sum(losses, dtype=jnp.float32)
    return jnp.mean(losses, dtype=jnp.float32)


def mse_loss(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    reduction: str = 'mean',
) -> Array:
    """Mean squared error between ``prediction`` and ``target``."""
    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(
            'prediction and target must have equal shapes, got '
            f'{prediction.shape} and {target.shape}'
        )
    _validate_reduction(reduction)
    return _reduce((prediction - target) ** 2, reduction)


def mae_loss(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    reduction: str = 'mean',
) -> Array:
    """Mean absolute error between ``prediction`` and ``target``."""
    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(
            'prediction and target must have equal shapes, got '
            f'{prediction.shape} and {target.shape}'
        )
    _validate_reduction(reduction)
    return _reduce(jnp.abs(prediction - target), reduction)


def smooth_l1_loss(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    beta: float = 1.0,
    reduction: str = 'mean',
) -> Array:
    """Smooth L1 (Huber) loss with quadratic behaviour inside ``beta``.

    ``0.5 * x^2 / beta`` for ``|x| < beta``, otherwise ``|x| - 0.5 * beta``.
    """
    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(
            'prediction and target must have equal shapes, got '
            f'{prediction.shape} and {target.shape}'
        )
    if not isinstance(beta, (int, float)) or beta <= 0:
        raise ValueError('beta must be a positive number')
    _validate_reduction(reduction)
    diff = jnp.abs(prediction - target)
    quadratic = jnp.minimum(diff, beta)
    losses = 0.5 * quadratic ** 2 / beta + jnp.maximum(diff - beta, 0.0)
    return _reduce(losses, reduction)


__all__ = ['mse_loss', 'mae_loss', 'smooth_l1_loss']
