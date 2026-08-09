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
"""Loss functions for TakTiny trainers."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from taktiny.cosettes.common import TransformerContext
from taktiny.utils.typing import Array, ArrayLike, Batch


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
    log_probabilities = jax.nn.log_softmax(
        logits.astype(jnp.float32),
        axis=-1,
    )
    losses = -jnp.take_along_axis(
        log_probabilities,
        safe_labels[..., None],
        axis=-1,
    )[..., 0]
    losses = jnp.where(selected, losses, 0.0)

    if reduction == 'none':
        return losses
    total = jnp.sum(losses, dtype=jnp.float32)
    if reduction == 'sum':
        return total
    count = jnp.sum(selected, dtype=jnp.float32)
    return total / jnp.maximum(count, 1.0)


def causal_lm_loss(
    model: Callable[..., Any],
    batch: Batch,
    *,
    ignore_index: int = -100,
) -> Array:
    """Compute next-token loss for a TakTiny causal language model.

    ``batch`` must contain ``input_ids`` and ``labels``. Optional
    ``attention_mask`` and ``position_ids`` values are forwarded to the model.
    A two-dimensional attention mask is interpreted as a key-padding mask.
    Reset positions mark packed sequence boundaries and are excluded from the
    shifted targets.

    This function has the ``loss_fn(model, batch)`` signature expected by
    :class:`~taktiny.trainer.Trainer`.
    """
    if not isinstance(batch, Mapping):
        raise TypeError('batch must be a mapping')
    missing = {'input_ids', 'labels'} - set(batch)
    if missing:
        names = ', '.join(sorted(missing))
        raise KeyError(f'causal LM batch is missing: {names}')

    input_ids = jnp.asarray(batch['input_ids'])
    labels = jnp.asarray(batch['labels'])
    if input_ids.ndim != 2:
        raise ValueError(
            'input_ids must have shape [batch, sequence], '
            f'got {input_ids.shape}'
        )
    if labels.shape != input_ids.shape:
        raise ValueError(
            'labels and input_ids must have equal shapes, got '
            f'{labels.shape} and {input_ids.shape}'
        )
    if input_ids.shape[1] < 2:
        raise ValueError('causal LM loss requires at least two tokens')

    attention_mask = batch.get('attention_mask')
    token_mask = None
    if attention_mask is not None:
        attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
        if attention_mask.ndim == 2:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    'a two-dimensional attention_mask must match input_ids'
                )
            token_mask = attention_mask
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.ndim not in (3, 4):
            raise ValueError(
                'attention_mask must have two, three, or four dimensions'
            )

    position_ids = batch.get('position_ids')
    if position_ids is not None:
        position_ids = jnp.asarray(position_ids, dtype=jnp.int32)
        if position_ids.shape != input_ids.shape:
            raise ValueError('position_ids and input_ids must have equal shapes')

    ctx = TransformerContext(
        key_cache=None,
        value_cache=None,
        position_idx=None,
        is_causal=True,
    )
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        ctx=ctx,
    )
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    if hasattr(logits, 'logits'):
        logits = logits.logits
    logits = jnp.asarray(logits)
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError(
            'model logits must have shape [batch, sequence, vocabulary], '
            f'got {logits.shape}; expected batch and sequence dimensions '
            f'{input_ids.shape}'
        )

    target_mask = None
    if token_mask is not None:
        target_mask = token_mask[:, 1:]
    if position_ids is not None:
        boundaries = position_ids[:, 1:] != 0
        target_mask = (
            boundaries
            if target_mask is None
            else target_mask & boundaries
        )

    return cross_entropy_loss(
        logits[:, :-1, :],
        labels[:, 1:],
        mask=target_mask,
        ignore_index=ignore_index,
    )


__all__ = ['cross_entropy_loss', 'causal_lm_loss']
