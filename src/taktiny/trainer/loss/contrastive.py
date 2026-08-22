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
"""Contrastive losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from taktiny.utils.typing import Array, ArrayLike


def infonce_loss(
    query: ArrayLike,
    keys: ArrayLike,
    *,
    temperature: float = 1.0,
    positive_mask: ArrayLike | None = None,
    reduction: str = 'mean',
) -> Array:
    """InfoNCE / NT-Xent contrastive loss over ``query`` vs ``keys``.

    Args:
        query: Shape ``[batch, dim]`` anchor embeddings.
        keys: Shape ``[batch, num_keys, dim]`` candidate embeddings.
        temperature: Logit scaling factor.
        positive_mask: Optional boolean mask ``[batch, num_keys]`` marking the
            positive keys. When omitted, the first key of every row is treated
            as the positive.
        reduction: ``"none"``, ``"sum"``, or ``"mean"`` (over the batch).

    Returns:
        ``-log( sum_pos exp(q.k / t) / sum_all exp(q.k / t) )`` per row,
        reduced according to ``reduction``.
    """
    query = jnp.asarray(query)
    keys = jnp.asarray(keys)
    if query.ndim != 2:
        raise ValueError(f'query must be [batch, dim], got {query.shape}')
    if keys.ndim != 3:
        raise ValueError(
            f'keys must be [batch, num_keys, dim], got {keys.shape}'
        )
    if query.shape[0] != keys.shape[0]:
        raise ValueError(
            'query and keys must have equal batch, got '
            f'{query.shape[0]} and {keys.shape[0]}'
        )
    if query.shape[-1] != keys.shape[-1]:
        raise ValueError(
            'query and keys must have equal dim, got '
            f'{query.shape[-1]} and {keys.shape[-1]}'
        )
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError('temperature must be a positive number')
    if reduction not in {'none', 'sum', 'mean'}:
        raise ValueError('reduction must be "none", "sum", or "mean"')

    logits = jnp.einsum('bd,bnd->bn', query, keys) / temperature

    if positive_mask is None:
        positive_mask = jnp.zeros_like(logits, dtype=jnp.bool_)
        positive_mask = positive_mask.at[:, 0].set(True)
    else:
        positive_mask = jnp.asarray(positive_mask, dtype=jnp.bool_)
        if positive_mask.shape != logits.shape:
            raise ValueError(
                'positive_mask must match [batch, num_keys], got '
                f'{positive_mask.shape}'
            )

    num_positives = jnp.sum(positive_mask, axis=-1, keepdims=True)
    log_positive = jax.nn.logsumexp(
        jnp.where(positive_mask, logits, -jnp.inf),
        axis=-1,
    ) - jnp.log(jnp.maximum(num_positives, 1.0)[..., 0])
    log_all = jax.nn.logsumexp(logits, axis=-1)
    losses = -log_positive + log_all

    if reduction == 'none':
        return losses
    if reduction == 'sum':
        return jnp.sum(losses, dtype=jnp.float32)
    return jnp.mean(losses, dtype=jnp.float32)


__all__ = ['infonce_loss']
