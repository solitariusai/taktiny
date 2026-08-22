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
"""Causal loss functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
from taktiny.utils.typing import Array, Batch

from .classification import cross_entropy_loss


def causal_lm_loss(
    model: Callable[..., Any],
    batch: Batch,
    *,
    ignore_index: int = -100,
    logits_chunk_size: int | None = None,
    attention_kernel: str = 'dot_product',
) -> Array:
    """Compute next-token loss for a TakTiny causal language model.

    ``batch`` must contain ``input_ids`` and ``labels``. Optional
    ``attention_mask`` and ``position_ids`` values are forwarded to the model.
    A two-dimensional attention mask is interpreted as a key-padding mask.
    Reset positions mark packed sequence boundaries and are excluded from the
    shifted targets.

    ``logits_chunk_size`` enables the chunked vocabulary projection on models
    implementing ``compute_causal_loss``: the LM head and cross entropy run
    over sequence chunks of that size inside a rematerialized scan, so the
    ``(sequence, vocab)`` logits tensor never materializes at full size.
    ``attention_kernel`` selects the attention backend for the decoder
    (``'dot_product'``, ``'flash'``, ``'ragged'``, ...).

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

    if logits_chunk_size is not None:
        if not hasattr(model, 'compute_causal_loss'):
            raise TypeError(
                'logits_chunk_size requires a model implementing '
                'compute_causal_loss'
            )
        return model.compute_causal_loss(
            input_ids,
            labels,
            attention_mask=batch.get('attention_mask'),
            position_ids=batch.get('position_ids'),
            ignore_index=ignore_index,
            logits_chunk_size=logits_chunk_size,
            attention_kernel=attention_kernel,
        )

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        is_causal=True,
        kernel=attention_kernel,
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


__all__ = ['causal_lm_loss']
