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
"""Preference (DPO / IPO) losses for RLHF-style fine-tuning."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from taktiny.utils.typing import Array, ArrayLike


def _preference_log_ratio(
    chosen_logps: ArrayLike,
    rejected_logps: ArrayLike,
    ref_chosen_logps: ArrayLike,
    ref_rejected_logps: ArrayLike,
) -> Array:
    """Per-pair log-ratio ``(chosen - ref_chosen) - (rejected - ref_rejected)``."""
    chosen = jnp.asarray(chosen_logps)
    rejected = jnp.asarray(rejected_logps)
    ref_chosen = jnp.asarray(ref_chosen_logps)
    ref_rejected = jnp.asarray(ref_rejected_logps)
    for name, value in (
        ('chosen_logps', chosen),
        ('rejected_logps', rejected),
        ('ref_chosen_logps', ref_chosen),
        ('ref_rejected_logps', ref_rejected),
    ):
        if value.ndim != 1:
            raise ValueError(f'{name} must be a vector, got {value.shape}')
    shapes = {value.shape for value in (chosen, rejected, ref_chosen, ref_rejected)}
    if len(shapes) != 1:
        raise ValueError('all log-probability vectors must have equal shapes')
    return (chosen - ref_chosen) - (rejected - ref_rejected)


def dpo_loss(
    chosen_logps: ArrayLike,
    rejected_logps: ArrayLike,
    ref_chosen_logps: ArrayLike,
    ref_rejected_logps: ArrayLike,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reduction: str = 'mean',
) -> Array:
    """Direct Preference Optimization loss (Rafailov et al., 2023).

    ``log_ratio = (chosen - ref_chosen) - (rejected - ref_rejected)`` and
    ``loss = -log(sigmoid(beta * log_ratio))`` with optional label smoothing.

    Args:
        chosen_logps: Policy log-probabilities of the chosen completions,
            shape ``[batch]``.
        rejected_logps: Policy log-probabilities of the rejected completions.
        ref_chosen_logps: Reference-model log-probabilities of the chosen.
        ref_rejected_logps: Reference-model log-probabilities of the rejected.
        beta: Temperature of the implicit reward.
        label_smoothing: In ``[0, 1)``; 0 is standard DPO.
        reduction: ``"none"``, ``"sum"``, or ``"mean"`` (over the batch).
    """
    log_ratio = _preference_log_ratio(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )
    if not isinstance(beta, (int, float)) or beta <= 0:
        raise ValueError('beta must be a positive number')
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError('label_smoothing must be in [0, 1)')
    if reduction not in {'none', 'sum', 'mean'}:
        raise ValueError('reduction must be "none", "sum", or "mean"')

    loss = -jax.nn.log_sigmoid(beta * log_ratio)
    if label_smoothing > 0.0:
        loss = (1.0 - label_smoothing) * loss - (
            label_smoothing * jax.nn.log_sigmoid(-beta * log_ratio)
        )

    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return jnp.sum(loss, dtype=jnp.float32)
    return jnp.mean(loss, dtype=jnp.float32)


def ipo_loss(
    chosen_logps: ArrayLike,
    rejected_logps: ArrayLike,
    ref_chosen_logps: ArrayLike,
    ref_rejected_logps: ArrayLike,
    *,
    beta: float = 0.1,
    reduction: str = 'mean',
) -> Array:
    """Identity Preference Optimization loss (Azar et al., 2023).

    ``loss = (log_ratio - 1 / (2 * beta)) ** 2`` where ``log_ratio`` is the
    same implicit-reward difference as DPO.
    """
    log_ratio = _preference_log_ratio(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )
    if not isinstance(beta, (int, float)) or beta <= 0:
        raise ValueError('beta must be a positive number')
    if reduction not in {'none', 'sum', 'mean'}:
        raise ValueError('reduction must be "none", "sum", or "mean"')

    loss = (log_ratio - 1.0 / (2.0 * beta)) ** 2
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return jnp.sum(loss, dtype=jnp.float32)
    return jnp.mean(loss, dtype=jnp.float32)


__all__ = ['dpo_loss', 'ipo_loss']
