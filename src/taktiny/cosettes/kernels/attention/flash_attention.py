#  Copyright 2026 Shinapri
#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Portable blockwise attention using boundary offsets and arbitrary masks."""
from __future__ import annotations

import typing as tp
import jax
import jax.numpy as jnp

if tp.TYPE_CHECKING:
    from taktiny.cosettes.kernels.attention.splash_attention import SegmentIds
else:
    SegmentIds = tp.Any

MaskFunction = tp.Callable[
    [jax.Array, jax.Array, jax.Array | None, jax.Array | None],
    jax.Array,
]
AttentionMask = jax.Array | MaskFunction
AttentionStatistics = dict[str, jax.Array]


def _round_up(value: int, block_size: int) -> int:
    return ((value + block_size - 1) // block_size) * block_size


def _normalize_mask(
    mask: AttentionMask,
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
) -> jax.Array:
    mask = jnp.asarray(mask)
    if mask.dtype != jnp.bool_:
        raise TypeError(f'mask must have boolean dtype, got {mask.dtype}')

    if mask.ndim == 2:
        mask = mask[None, None, :, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    elif mask.ndim != 4:
        raise ValueError(
            'mask must have shape [query, key], [batch, query, key], or '
            f'[batch, heads, query, key], got {mask.shape}'
        )

    expected = (batch_size, num_heads, query_length, key_length)
    axis_names = ('batch', 'heads', 'query', 'key')
    for actual, wanted, axis in zip(
        mask.shape,
        expected,
        axis_names,
        strict=True,
    ):
        if actual not in (1, wanted):
            raise ValueError(
                f'mask {axis} axis must have size 1 or {wanted}, '
                f'got shape {mask.shape}'
            )
    return mask


def _normalize_boundaries(
    boundary_ids: jax.Array,
    *,
    batch_size: int,
) -> jax.Array:
    boundaries = jnp.asarray(boundary_ids)
    if not jnp.issubdtype(boundaries.dtype, jnp.integer):
        raise TypeError(
            f'boundary_ids must have an integer dtype, got {boundaries.dtype}'
        )
    if boundaries.ndim == 1:
        boundaries = boundaries[None, :]
    elif boundaries.ndim != 2:
        raise ValueError(
            'boundary_ids must have shape [num_boundaries] or '
            f'[batch, num_boundaries], got {boundaries.shape}'
        )
    if boundaries.shape[-1] < 2:
        raise ValueError('boundary_ids must contain at least a start and end')
    if boundaries.shape[0] not in (1, batch_size):
        raise ValueError(
            f'boundary_ids batch size must be 1 or {batch_size}, '
            f'got {boundaries.shape[0]}'
        )
    return jnp.broadcast_to(
        boundaries,
        (batch_size, boundaries.shape[-1]),
    )


def _document_ids(
    positions: jax.Array,
    boundaries: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the interval index and validity of every absolute position."""
    positions = jnp.asarray(positions, dtype=jnp.int32)
    if positions.ndim == 1:
        positions = jnp.broadcast_to(
            positions[None, :],
            (boundaries.shape[0], positions.shape[0]),
        )
    elif positions.ndim != 2 or positions.shape[0] != boundaries.shape[0]:
        raise ValueError(
            'positions must have shape [sequence] or [batch, sequence]'
        )
    document_ids = jnp.sum(
        positions[:, :, None] >= boundaries[:, None, 1:],
        axis=-1,
        dtype=jnp.int32,
    )
    valid = (
        (positions >= boundaries[:, :1])
        & (positions < boundaries[:, -1:])
    )
    return document_ids, valid


def _broadcast_block_mask(
    mask: jax.Array,
    *,
    batch_size: int,
    num_heads: int,
    block_q: int,
    block_kv: int,
) -> jax.Array:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if mask.ndim == 2:
        mask = mask[None, None, :, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    elif mask.ndim != 4:
        raise ValueError(
            'a mask function must return a rank-2, rank-3, or rank-4 array, '
            f'got shape {mask.shape}'
        )
    try:
        return jnp.broadcast_to(
            mask,
            (batch_size, num_heads, block_q, block_kv),
        )
    except ValueError as error:
        raise ValueError(
            'mask block is not broadcastable to '
            f'{(batch_size, num_heads, block_q, block_kv)}; got {mask.shape}'
        ) from error


def materialize_attention_mask(
    mask: AttentionMask | None,
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
    boundary_ids: jax.Array | None = None,
    is_causal: bool = False,
    query_offset: int | jax.Array = 0,
    respect_boundaries: bool = True,
) -> jax.Array:
    """Materialize the visibility accepted by boundary FlashAttention.

    This is intended for backends such as ``jax.nn.dot_product_attention``
    that require a dense mask. The blockwise implementation evaluates the
    same rules per block and does not call this helper.
    """
    offset = jnp.asarray(query_offset, dtype=jnp.int32)
    if offset.ndim == 0:
        offset = jnp.broadcast_to(offset, (batch_size,))
    elif offset.shape != (batch_size,):
        raise ValueError(
            f'query_offset must be scalar or shape {(batch_size,)}, '
            f'got {offset.shape}'
        )

    query_positions = (
        offset[:, None]
        + jnp.arange(query_length, dtype=jnp.int32)[None, :]
    )
    key_positions = jnp.arange(key_length, dtype=jnp.int32)
    visibility = jnp.ones(
        (batch_size, num_heads, query_length, key_length),
        dtype=jnp.bool_,
    )

    query_documents = key_documents = None
    if boundary_ids is not None:
        boundaries = _normalize_boundaries(
            boundary_ids,
            batch_size=batch_size,
        )
        query_documents, query_valid = _document_ids(
            query_positions,
            boundaries,
        )
        key_documents, key_valid = _document_ids(
            key_positions,
            boundaries,
        )
        if respect_boundaries:
            visibility &= (
                (query_documents[:, None, :, None]
                 == key_documents[:, None, None, :])
                & query_valid[:, None, :, None]
                & key_valid[:, None, None, :]
            )

    if is_causal:
        visibility &= (
            key_positions[None, None, None, :]
            <= query_positions[:, None, :, None]
        )

    if callable(mask):
        function_mask = mask(
            query_positions[:, :, None],
            key_positions[None, None, :],
            (
                None
                if query_documents is None
                else query_documents[:, :, None]
            ),
            (
                None
                if key_documents is None
                else key_documents[:, None, :]
            ),
        )
        visibility &= _broadcast_block_mask(
            function_mask,
            batch_size=batch_size,
            num_heads=num_heads,
            block_q=query_length,
            block_kv=key_length,
        )
    elif mask is not None:
        visibility &= jnp.broadcast_to(
            _normalize_mask(
                mask,
                batch_size=batch_size,
                num_heads=num_heads,
                query_length=query_length,
                key_length=key_length,
            ),
            visibility.shape,
        )

    return visibility


def flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: AttentionMask | None = None,
    mask_value: float = -jnp.inf,
    boundary_ids: jax.Array | None = None,
    *,
    block_q: int = 128,
    block_kv: int = 128,
    scale: float | None = None,
    cap: float | None = None,
    is_causal: bool = False,
    query_offset: int | jax.Array = 0,
    respect_boundaries: bool = True,
    save_residuals: bool = False,
) -> jax.Array | tuple[jax.Array, AttentionStatistics]:
    """Compute blockwise grouped-query attention without full logits.

    ``q``, ``k``, and ``v`` use TakTiny's native
    ``[batch, sequence, heads, dimension]`` layout. ``boundary_ids`` contains
    packed-document offsets such as ``[0, 3, 10, 17]`` and can instead be
    shaped ``[batch, num_boundaries]`` for per-example boundaries.

    ``mask`` may be an arbitrary boolean array with shape ``[query, key]``,
    ``[batch, query, key]``, or ``[batch, heads, query, key]``. It may also be
    a callable accepting ``(query_positions, key_positions,
    query_documents, key_documents)``. A callable is evaluated per block and
    therefore does not need to allocate a full quadratic visibility matrix.

    Set ``respect_boundaries=False`` when a custom mask deliberately permits
    cross-document attention. ``query_offset`` gives the absolute position of
    the first query, which supports cached decoding where query and key lengths
    differ.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('q, k, and v must all be rank-4 arrays')

    batch_size, query_length, num_query_heads, head_dim = q.shape
    key_batch, key_length, num_kv_heads, key_head_dim = k.shape
    value_batch, value_length, value_heads, value_dim = v.shape
    if key_batch != batch_size or value_batch != batch_size:
        raise ValueError('q, k, and v must have the same batch size')
    if value_length != key_length:
        raise ValueError('k and v must have the same sequence length')
    if value_heads != num_kv_heads:
        raise ValueError('k and v must have the same number of heads')
    if key_head_dim != head_dim:
        raise ValueError('q and k must have the same head dimension')
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            'query heads must be divisible by key/value heads, got '
            f'{num_query_heads} and {num_kv_heads}'
        )
    if block_q <= 0 or block_kv <= 0:
        raise ValueError('block_q and block_kv must be greater than zero')
    if cap is not None and cap <= 0:
        raise ValueError('cap must be greater than zero')

    array_mask = None
    mask_function = mask if callable(mask) else None
    if mask is not None and mask_function is None:
        array_mask = _normalize_mask(
            mask,
            batch_size=batch_size,
            num_heads=num_query_heads,
            query_length=query_length,
            key_length=key_length,
        )

    padded_query_length = _round_up(query_length, block_q)
    padded_key_length = _round_up(key_length, block_kv)
    query_padding = padded_query_length - query_length
    key_padding = padded_key_length - key_length

    input_dtype = q.dtype
    q = jnp.pad(q, ((0, 0), (0, query_padding), (0, 0), (0, 0)))
    k = jnp.pad(k, ((0, 0), (0, key_padding), (0, 0), (0, 0)))
    v = jnp.pad(v, ((0, 0), (0, key_padding), (0, 0), (0, 0)))
    if array_mask is not None:
        array_mask = jnp.broadcast_to(
            array_mask,
            (batch_size, num_query_heads, query_length, key_length),
        )
        array_mask = jnp.pad(
            array_mask,
            ((0, 0), (0, 0), (0, query_padding), (0, key_padding)),
            constant_values=False,
        )

    offset = jnp.asarray(query_offset, dtype=jnp.int32)
    if offset.ndim == 0:
        offset = jnp.broadcast_to(offset, (batch_size,))
    elif offset.shape != (batch_size,):
        raise ValueError(
            f'query_offset must be scalar or shape {(batch_size,)}, '
            f'got {offset.shape}'
        )
    query_positions = (
        offset[:, None]
        + jnp.arange(padded_query_length, dtype=jnp.int32)[None, :]
    )
    key_positions = jnp.arange(padded_key_length, dtype=jnp.int32)
    query_valid = query_positions < query_length + offset[:, None]
    key_valid = key_positions < key_length

    query_documents = key_documents = None
    query_boundary_valid = key_boundary_valid = None
    if boundary_ids is not None:
        boundaries = _normalize_boundaries(
            boundary_ids,
            batch_size=batch_size,
        )
        query_documents, query_boundary_valid = _document_ids(
            query_positions,
            boundaries,
        )
        key_documents, key_boundary_valid = _document_ids(
            key_positions,
            boundaries,
        )

    groups = num_query_heads // num_kv_heads
    q = q.transpose(0, 2, 1, 3).reshape(
        batch_size,
        num_kv_heads,
        groups,
        padded_query_length,
        head_dim,
    )
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    accumulator = jnp.zeros(
        (
            batch_size,
            num_kv_heads,
            groups,
            padded_query_length,
            value_dim,
        ),
        dtype=jnp.float32,
    )
    denominator = jnp.zeros(
        (batch_size, num_kv_heads, groups, padded_query_length),
        dtype=jnp.float32,
    )
    maximum = jnp.full(denominator.shape, -jnp.inf, dtype=jnp.float32)
    dot_scale = head_dim ** -0.5 if scale is None else scale

    def key_loop(
        key_block_index: jax.Array,
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        accumulator, denominator, maximum = state
        key_start = key_block_index * block_kv
        key_block = jax.lax.dynamic_slice_in_dim(
            k, key_start, block_kv, axis=2
        )
        value_block = jax.lax.dynamic_slice_in_dim(
            v, key_start, block_kv, axis=2
        )
        block_key_positions = jax.lax.dynamic_slice_in_dim(
            key_positions, key_start, block_kv
        )[None, :]
        block_key_valid = jax.lax.dynamic_slice_in_dim(
            key_valid, key_start, block_kv
        )[None, :]

        block_key_documents = None
        block_key_boundary_valid = None
        if key_documents is not None:
            block_key_documents = jax.lax.dynamic_slice_in_dim(
                key_documents, key_start, block_kv, axis=1
            )[:, None, :]
            block_key_boundary_valid = jax.lax.dynamic_slice_in_dim(
                key_boundary_valid, key_start, block_kv, axis=1
            )[:, None, :]

        def query_loop(
            query_block_index: jax.Array,
            inner_state: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            accumulator, denominator, maximum = inner_state
            query_start = query_block_index * block_q
            query_block = jax.lax.dynamic_slice_in_dim(
                q, query_start, block_q, axis=3
            )
            block_query_positions = jax.lax.dynamic_slice_in_dim(
                query_positions, query_start, block_q, axis=1
            )[:, :, None]
            block_query_valid = jax.lax.dynamic_slice_in_dim(
                query_valid, query_start, block_q, axis=1
            )[:, :, None]

            block_query_documents = None
            block_query_boundary_valid = None
            if query_documents is not None:
                block_query_documents = jax.lax.dynamic_slice_in_dim(
                    query_documents, query_start, block_q, axis=1
                )[:, :, None]
                block_query_boundary_valid = jax.lax.dynamic_slice_in_dim(
                    query_boundary_valid, query_start, block_q, axis=1
                )[:, :, None]

            block_mask = block_query_valid & block_key_valid
            block_mask = jnp.broadcast_to(
                block_mask[:, None, :, :],
                (batch_size, num_query_heads, block_q, block_kv),
            )
            if is_causal:
                block_mask &= (
                    block_key_positions <= block_query_positions
                )[:, None, :, :]

            if respect_boundaries and block_query_documents is not None:
                boundary_mask = (
                    block_query_documents == block_key_documents
                ) & block_query_boundary_valid & block_key_boundary_valid
                block_mask &= boundary_mask[:, None, :, :]

            if array_mask is not None:
                sliced_mask = jax.lax.dynamic_slice(
                    array_mask,
                    (0, 0, query_start, key_start),
                    (
                        array_mask.shape[0],
                        array_mask.shape[1],
                        block_q,
                        block_kv,
                    ),
                )
                block_mask &= jnp.broadcast_to(
                    sliced_mask,
                    (batch_size, num_query_heads, block_q, block_kv),
                )

            if mask_function is not None:
                function_mask = mask_function(
                    block_query_positions,
                    block_key_positions,
                    block_query_documents,
                    block_key_documents,
                )
                block_mask &= _broadcast_block_mask(
                    function_mask,
                    batch_size=batch_size,
                    num_heads=num_query_heads,
                    block_q=block_q,
                    block_kv=block_kv,
                )

            def compute_visible_block(
                state: tuple[jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                accumulator, denominator, maximum = state
                grouped_mask = block_mask.reshape(
                    batch_size,
                    num_kv_heads,
                    groups,
                    block_q,
                    block_kv,
                )
                accumulator_block = jax.lax.dynamic_slice_in_dim(
                    accumulator, query_start, block_q, axis=3
                )
                denominator_block = jax.lax.dynamic_slice_in_dim(
                    denominator, query_start, block_q, axis=3
                )
                maximum_block = jax.lax.dynamic_slice_in_dim(
                    maximum, query_start, block_q, axis=3
                )

                logits = jnp.einsum(
                    'bngqd,bnkd->bngqk',
                    query_block,
                    key_block,
                    preferred_element_type=jnp.float32,
                )
                logits *= jnp.asarray(dot_scale, dtype=jnp.float32)
                if cap is not None:
                    logits = jnp.tanh(logits / cap) * cap
                masked_logits = jnp.where(grouped_mask, logits, -jnp.inf)

                block_maximum = jnp.max(masked_logits, axis=-1)
                new_maximum = jnp.maximum(maximum_block, block_maximum)
                safe_maximum = jnp.where(
                    jnp.isfinite(new_maximum),
                    new_maximum,
                    0.0,
                )
                old_factor = jnp.where(
                    jnp.isfinite(maximum_block),
                    jnp.exp(maximum_block - safe_maximum),
                    0.0,
                )
                probabilities = jnp.where(
                    grouped_mask,
                    jnp.exp(masked_logits - safe_maximum[..., None]),
                    0.0,
                )
                new_denominator = (
                    denominator_block * old_factor
                    + jnp.sum(probabilities, axis=-1)
                )
                new_accumulator = (
                    accumulator_block * old_factor[..., None]
                    + jnp.einsum(
                        'bngqk,bnkd->bngqd',
                        probabilities,
                        value_block.astype(jnp.float32),
                        preferred_element_type=jnp.float32,
                    )
                )

                accumulator = jax.lax.dynamic_update_slice_in_dim(
                    accumulator, new_accumulator, query_start, axis=3
                )
                denominator = jax.lax.dynamic_update_slice_in_dim(
                    denominator, new_denominator, query_start, axis=3
                )
                maximum = jax.lax.dynamic_update_slice_in_dim(
                    maximum, new_maximum, query_start, axis=3
                )
                return accumulator, denominator, maximum

            return jax.lax.cond(
                jnp.any(block_mask),
                compute_visible_block,
                lambda state: state,
                (accumulator, denominator, maximum),
            )

        return jax.lax.fori_loop(
            0,
            padded_query_length // block_q,
            query_loop,
            (accumulator, denominator, maximum),
        )

    accumulator, denominator, maximum = jax.lax.fori_loop(
        0,
        padded_key_length // block_kv,
        key_loop,
        (accumulator, denominator, maximum),
    )

    output = accumulator / jnp.where(
        denominator > 0, denominator, 1.0
    )[..., None]
    output = output.reshape(
        batch_size,
        num_query_heads,
        padded_query_length,
        value_dim,
    ).transpose(0, 2, 1, 3)
    output = output[:, :query_length].astype(input_dtype)
    if not save_residuals:
        return output

    maximum = maximum.reshape(
        batch_size,
        num_query_heads,
        padded_query_length,
    ).transpose(0, 2, 1)
    denominator = denominator.reshape(
        batch_size,
        num_query_heads,
        padded_query_length,
    ).transpose(0, 2, 1)
    has_visible_keys = denominator > 0
    logsumexp = jnp.where(
        has_visible_keys,
        maximum + jnp.log(denominator),
        jnp.asarray(mask_value, dtype=jnp.float32),
    )
    reported_maximum = jnp.where(
        has_visible_keys,
        maximum,
        jnp.asarray(mask_value, dtype=jnp.float32),
    )
    statistics = {
        'logsumexp': jax.lax.stop_gradient(logsumexp[:, :query_length]),
        'max_logits': jax.lax.stop_gradient(
            reported_maximum[:, :query_length]
        ),
    }
    return output, statistics


# This function computes masked flash attention using a block-sparse approach.
# This implementation keeps the full batch and number of heads dimensions
# throughout the attention computation while iterating through blocks of the
# key/value sequence and, within each, iterates through blocks of the query
# sequence. The `mask_blocked` is used to skip computations for blocks where all
# attention scores are masked out, improving efficiency for sparse masks.
def flash_attention_block_masked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    block_kv: int,
    block_q: int,
    mask: jax.Array,
    mask_value: float,
    cap: tp.Optional[float] = None,
    save_residuals: bool = False,
) -> tp.Union[jax.Array, tp.Tuple[jax.Array, tp.Tuple[jax.Array, jax.Array]]]:
    """Computes masked flash attention using block-sparse masking.

    Args:
        q: Query tensor with shape (batch_size, num_q_heads, q_seq_len,
            head_dim).
        k: Key tensor with shape (batch_size, num_kv_heads, kv_seq_len, head_dim).
        v: Value tensor with shape (batch_size, num_kv_heads, kv_seq_len,
            v_head_dim).
        segment_ids: SegmentIds are a mechanism to ensure that there is no
            cross-attention between segments (fraction of a sequence) that have been
            concatenated together into a sequence. Each array is a list of ids
            (integers). Only tokens with the same id are allowed to attend to each
            other. It stores the segment ids of the query and key/value sequences.
        block_kv: Block size for the key/value sequence dimension.
        block_q: Block size for the query sequence dimension.
        mask: The full attention mask with shape of (q_seq_len, kv_seq_len). This
            mask will be used for all batches.
        mask_value: The value to use for masked-out attention scores.
        cap: tp.Optional cap for attention logits. This helps to prevent extremely
            large logits: capped_logits = jnp.tanh(logits / attn_logits_soft_cap) *
            attn_logits_soft_cap
        save_residuals: Whether to save residuals. If True, returns a tuple of
            (output, dict=(logsumexp, max_logits)). Both `logsumexp` and `max_logits`
            are of shape (batch_size, num_q_heads, q_seq_len).

    Returns:
        If save_residuals is True, returns a tuple containing:
            - The output of the attention computation.
            - A dict of (logsumexp, max_logits)
        Otherwise, returns the output of the attention computation.
    """
    batch_size, num_q_heads, q_seq_len, qk_head_dim_size = q.shape
    _, num_kv_heads, kv_seq_len, _ = k.shape
    v_head_dim_size = v.shape[-1]
    data_type = q.dtype
    q_groups = num_q_heads // num_kv_heads
    q = q.reshape(
        (
            batch_size,
            num_kv_heads,
            q_groups,
            q_seq_len,
            qk_head_dim_size,
        )
    )

    # Calculate the number of key/value and query blocks.
    num_kv_blocks = kv_seq_len // block_kv
    num_q_blocks = q_seq_len // block_q

    if mask is None:
        mask = jnp.ones((q_seq_len, kv_seq_len), dtype=jnp.bool_)
    elif mask.ndim == 3 and mask.shape[0] == batch_size:
        mask_full = mask
    elif mask.ndim == 2:
        mask_full = jnp.broadcast_to(mask[None, :, :], (batch_size, q_seq_len, kv_seq_len))
    else:
        mask_full = jnp.broadcast_to(mask, (batch_size, q_seq_len, kv_seq_len))

    if 'mask_full' not in locals():
        mask_full = jnp.broadcast_to(mask[None, :, :], (batch_size, q_seq_len, kv_seq_len))

    if segment_ids is not None:
        segment_ids_q = segment_ids.q[:, :, None]
        segment_ids_kv = segment_ids.kv[:, None, :]
        mask_full = jnp.logical_and(mask_full, segment_ids_q == segment_ids_kv)
    mask_blocked = jax.jit(mask_blocker_, static_argnums=[1, 2])(
        mask_full,
        block_q,
        block_kv,
    )

    # Initialize `l` (logsumexp) and `m` (max_logits) for the online softmax.
    # `l` is initialized to 0 since no blocks have been processed yet and the sum
    # is 0.
    l = jnp.zeros((batch_size, num_kv_heads, q_groups, q_seq_len), dtype=data_type)
    # `m` is initialized to the mask_value so that the first block's maximum logit
    # correctly becomes the running maximum.
    m = jnp.full(
        (batch_size, num_kv_heads, q_groups, q_seq_len),
        mask_value,
        dtype=data_type,
    )

    output = jnp.zeros(
        (
            batch_size,
            num_kv_heads,
            q_groups,
            q_seq_len,
            v_head_dim_size,
        ),
        dtype=data_type,
    )

    # Outer loop over the key/value blocks.
    def outer_loop_body(j: Any, carried: Any) -> tuple[Any, ...]:
        output, l, m = carried
        k_j_slice = jax.lax.dynamic_slice_in_dim(k, j * block_kv, block_kv, axis=-2)
        v_j_slice = jax.lax.dynamic_slice_in_dim(v, j * block_kv, block_kv, axis=-2)
        # Inner loop over the query blocks.
        def inner_loop_body(i: Any, carried_inner: Any) -> tuple[Any, ...]:
            output, l, m = carried_inner

            # let's get the slice of Q in N dimension
            q_slice = jax.lax.dynamic_slice_in_dim(q, i * block_q, block_q, axis=-2)

            # Calculates the attention computation (Q@K.T)@V with online softmax for
            # the current query and key/value blocks.
            def compute_attention_block(output: Any, l: Any, m: Any) -> tuple[Any, ...]:
                output_i_slice = jax.lax.dynamic_slice_in_dim(output, i * block_q, block_q, axis=-2)
                l_i_slice = jax.lax.dynamic_slice_in_dim(l, i * block_q, block_q, axis=-1)
                m_i_slice = jax.lax.dynamic_slice_in_dim(m, i * block_q, block_q, axis=-1)
                s_i_j = jnp.einsum(
                    "bxhqc,bxkc->bxhqk",
                    q_slice,
                    k_j_slice,
                    preferred_element_type=data_type,
                )
                full_mask_i_j_slice = jax.lax.dynamic_slice(
                    mask_full,
                    (0, i * block_q, j * block_kv),
                    (batch_size, block_q, block_kv),
                )
                broadcasted_mask = jnp.broadcast_to(
                    full_mask_i_j_slice[:, None, None, :, :],
                    (batch_size, num_kv_heads, q_groups, block_q, block_kv),
                )
        
                if cap is not None:
                    s_i_j = jnp.tanh(s_i_j / cap)
                    s_i_j = s_i_j * cap

                s_i_j = jnp.where(broadcasted_mask, s_i_j, mask_value)
                m_i_j = s_i_j.max(axis=-1)
                p_i_j = jnp.exp(s_i_j - m_i_j[..., None])
                l_i_j = p_i_j.sum(axis=-1)
                assert m_i_j.shape == m_i_slice.shape
                m_i_new = jnp.maximum(m_i_slice, m_i_j)
                m_i_difference = jnp.exp(m_i_slice - m_i_new)
                m_i_j_difference = jnp.exp(m_i_j - m_i_new)
                l_i_new = m_i_difference * l_i_slice + m_i_j_difference * l_i_j
        
                divider = l_i_new[..., None]
                numerator = l_i_slice[..., None] * m_i_difference[..., None] * output_i_slice + m_i_j_difference[
                    ..., None
                ] * jnp.einsum(
                    "bxhqk,bxkc->bxhqc",
                    p_i_j,
                    v_j_slice,
                    preferred_element_type=data_type,
                )
        
                output_i_slice_new = numerator / divider
                output = jax.lax.dynamic_update_index_in_dim(output, output_i_slice_new, i * block_q, axis=-2)
                l = jax.lax.dynamic_update_index_in_dim(l, l_i_new, i * block_q, axis=-1)
                m = jax.lax.dynamic_update_index_in_dim(m, m_i_new, i * block_q, axis=-1)
                return output, l, m

            def identity(output: Any, l: Any, m: Any) -> tuple[Any, ...]:
                """A no-op identity function."""
        
                return output, l, m

            batch_size = mask_blocked.shape[0]
            mask_i_j_slice = jax.lax.dynamic_slice(mask_blocked, (0, i, j), (batch_size, 1, 1))
            # The compute_attention_block should be executed if at least one element
            # in the slice is non-zero, meaning at least one batch requires work for
            # this block.
            output, l, m = jax.lax.cond(
                jnp.any(jnp.not_equal(mask_i_j_slice, 0)),
                compute_attention_block,
                identity,
                output,
                l,
                m,
            )
    
            return output, l, m

        output, l, m = jax.lax.fori_loop(0, num_q_blocks, inner_loop_body, (output, l, m), unroll=True)
        
        return (output, l, m)

    output, l, m = jax.lax.fori_loop(0, num_kv_blocks, outer_loop_body, (output, l, m), unroll=True)
    
    output = output.reshape(
        batch_size,
        num_q_heads,
        q_seq_len,
        v_head_dim_size,
    )
    if not save_residuals:
        # To avoid remat of the output, we can use context=hbm remat policy as in
        # maxtext/configs/types.py
        return output

    l = l.reshape(batch_size, num_q_heads, q_seq_len)
    m = m.reshape(batch_size, num_q_heads, q_seq_len)
    stats = {"logsumexp": m + jnp.log(l), "max_logits": m}
    stats = jax.tree.map(jax.lax.stop_gradient, stats)
    return output, stats

def mask_blocker_(mask: jax.Array, block_q: int, block_kv: int) -> jax.Array:
    """Creates a blocked mask from a full mask.

    Args:
        mask: The attention mask with shape of (batch_size, q_seq_len, kv_seq_len).
        block_q: Block size for the query sequence dimension.
        block_kv: Block size for the key/value sequence dimension.

    Returns:
        A blocked mask where each element indicates the number of non-zero
        elements in the corresponding block of the original mask.
    """
    batch_size, q_seq_len, kv_seq_len = mask.shape

    if q_seq_len % block_q != 0:
        raise ValueError(f"q_seq_len {q_seq_len} must be divisible by block_q {block_q}")
    if kv_seq_len % block_kv != 0:
        raise ValueError(f"kv_seq_len {kv_seq_len} must be divisible by block_kv {block_kv}")
    q_blocks = q_seq_len // block_q
    kv_blocks = kv_seq_len // block_kv

    blocked_mask = mask.reshape(batch_size, q_blocks, block_q, kv_blocks, block_kv)
    return jnp.count_nonzero(blocked_mask, axis=(2, 4)).astype(jnp.int32)
