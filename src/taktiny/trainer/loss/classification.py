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
"""Classification loss functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from taktiny.utils.typing import Array, ArrayLike


def cross_entropy_loss(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
) -> Array:
    """Compute stable integer-label cross entropy.

    Args:
        logits: Unnormalized scores with shape ``[..., vocabulary]``.
        labels: Integer targets with shape ``logits.shape[:-1]``.
        mask: Optional boolean mask selecting targets that contribute.
        ignore_index: Label value excluded from the loss.
        reduction: ``"none"``, ``"sum"``, or ``"mean"``. The mean is over
            selected targets rather than all tensor elements.

    Returns:
        Per-target losses or a scalar according to ``reduction``. An empty
        mean returns zero rather than NaN.
    """
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)
    if logits.ndim < 1:
        raise ValueError('logits must have at least one dimension')
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            'labels must match logits without the vocabulary dimension; '
            f'got logits {logits.shape} and labels {labels.shape}'
        )
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise TypeError('labels must have an integer dtype')
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError('ignore_index must be an integer')
    if reduction not in {'none', 'sum', 'mean'}:
        raise ValueError('reduction must be "none", "sum", or "mean"')

    selected = labels != ignore_index
    if mask is not None:
        mask = jnp.asarray(mask, dtype=jnp.bool_)
        if mask.shape != labels.shape:
            raise ValueError(
                f'mask must have shape {labels.shape}, got {mask.shape}'
            )
        selected &= mask

    safe_labels = jnp.where(selected, labels, 0)

    import optax
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32),
        safe_labels,
    )
    losses = jnp.where(selected, losses, 0.0)

    if reduction == 'none':
        return losses
    total = jnp.sum(losses, dtype=jnp.float32)
    if reduction == 'sum':
        return total
    count = jnp.sum(selected, dtype=jnp.float32)
    return total / jnp.maximum(count, 1.0)


def focal_loss(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
) -> Array:
    """Focal loss (Lin et al., 2017) for imbalanced classification.

    ``FL = -alpha_t * (1 - p_t)^gamma * log(p_t)``, where ``p_t`` is the
    predicted probability of the true class. Compared with plain cross
    entropy, the ``(1 - p_t)^gamma`` factor down-weights well-classified
    examples.

    Args:
        logits: Unnormalized scores, shape ``[..., vocab]``.
        labels: Integer targets, shape ``logits.shape[:-1]``.
        gamma: Focusing parameter (``>= 0``).
        alpha: Optional class weight; either a scalar or one weight per class.
        ignore_index: Label value excluded from the loss.
        reduction: ``"none"``, ``"sum"``, or ``"mean"`` (mean over selected
            targets).
    """
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)
    if logits.ndim < 1:
        raise ValueError('logits must have at least one dimension')
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            'labels must match logits without the vocabulary dimension; '
            f'got logits {logits.shape} and labels {labels.shape}'
        )
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise TypeError('labels must have an integer dtype')
    if not isinstance(gamma, (int, float)) or gamma < 0:
        raise ValueError('gamma must be a non-negative number')
    if reduction not in {'none', 'sum', 'mean'}:
        raise ValueError('reduction must be "none", "sum", or "mean"')

    selected = labels != ignore_index
    safe_labels = jnp.where(selected, labels, 0)
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    log_p_t = jnp.take_along_axis(
        log_probs,
        safe_labels[..., None],
        axis=-1,
    )[..., 0]
    p_t = jnp.exp(log_p_t)
    loss = -((1.0 - p_t) ** gamma) * log_p_t

    if alpha is not None:
        alpha = jnp.asarray(alpha, dtype=jnp.float32)
        weight = alpha if alpha.ndim == 0 else alpha[safe_labels]
        loss = weight * loss

    loss = jnp.where(selected, loss, 0.0)
    if reduction == 'none':
        return loss
    total = jnp.sum(loss, dtype=jnp.float32)
    if reduction == 'sum':
        return total
    count = jnp.sum(selected, dtype=jnp.float32)
    return total / jnp.maximum(count, 1.0)


__all__ = ['cross_entropy_loss', 'focal_loss']
