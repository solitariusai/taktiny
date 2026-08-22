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
"""Distributional losses (KL divergence)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from taktiny.utils.typing import Array, ArrayLike


def kl_divergence(
    logits: ArrayLike,
    target_logits: ArrayLike,
    *,
    reduction: str = 'batchmean',
) -> Array:
    """KL divergence ``KL(p || q)`` over the last axis, where ``p`` is the
    softmax of ``logits`` and ``q`` is the softmax of ``target_logits``.

    Each row is ``sum_d p_d (log p_d - log q_d)``, so passing
    ``target_logits`` as the reference makes the loss an asymmetric regularizer
    toward the target distribution.

    Args:
        logits: Unnormalized scores, shape ``[..., d]``.
        target_logits: Unnormalized reference scores, shape ``[..., d]``.
        reduction: ``"none"`` (per-row), ``"sum"``, ``"mean"`` (mean over all
            elements), or ``"batchmean"`` (sum over features, mean over the
            leading dimensions; PyTorch ``KLDivLoss`` default).

    Returns:
        Per-row KL values or a scalar according to ``reduction``.
    """
    logits = jnp.asarray(logits)
    target_logits = jnp.asarray(target_logits)
    if logits.shape != target_logits.shape:
        raise ValueError(
            'logits and target_logits must have equal shapes, got '
            f'{logits.shape} and {target_logits.shape}'
        )
    if logits.ndim == 0:
        raise ValueError('logits must have at least one dimension')
    if reduction not in {'none', 'sum', 'mean', 'batchmean'}:
        raise ValueError(
            'reduction must be "none", "sum", "mean", or "batchmean"'
        )

    p = jax.nn.softmax(logits, axis=-1)
    log_p = jax.nn.log_softmax(logits, axis=-1)
    log_q = jax.nn.log_softmax(target_logits, axis=-1)
    per_row = jnp.sum(p * (log_p - log_q), axis=-1)

    if reduction == 'none':
        return per_row
    if reduction == 'sum':
        return jnp.sum(per_row, dtype=jnp.float32)
    if reduction == 'mean':
        return jnp.mean(per_row, dtype=jnp.float32)
    # batchmean: sum over features, mean over leading dims.
    return jnp.mean(per_row, dtype=jnp.float32)


__all__ = ['kl_divergence']
