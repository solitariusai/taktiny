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
"""PEFT modules"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from taktiny.nn.base import Module, Parameter
from taktiny.nn.modules.linear import Linear, default_kernel_initializer
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import _constrain
from taktiny.utils.typing import AxisNames, DType, Initializer


class LoRALinear(Module):
    """Low-Rank Adaptation (LoRA) wrapper for Linear."""

    def __init__(
        self,
        base: Linear,
        rank: int,
        alpha: float,
        *,
        dtype: DType | None = None,
        rngs: Rngs,
        initializer: Initializer = default_kernel_initializer,
        dot_general: Any = None,
        axis_names: tuple[AxisNames | None, AxisNames | None] | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"`rank` must be > 0, got {rank}")

        self.base = base.eval()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        a_axis_names = None
        b_axis_names = None

        if axis_names is not None:
            a_axis_names, b_axis_names = axis_names

        self.lora_A = Linear(
            base.in_features,
            rank,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            dot_general=dot_general,
            axis_names=a_axis_names,
        )

        self.lora_B = Linear(
            rank,
            base.out_features,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            initializer=jax.nn.initializers.zeros,
            dot_general=dot_general,
            axis_names=b_axis_names,
        )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        out = self.base(
            x,
            out_sharding=out_sharding,
        )

        lora = self.lora_A(x)
        lora = self.lora_B(
            lora,
            out_sharding=out_sharding,
        )

        return out + lora * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.scaling * self.rank}"

class DoRALinear(Module):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA) for Linear layers.
    """

    def __init__(
        self,
        base: Linear,
        rank: int = 8,
        alpha: float = 16.0,
        *,
        dtype: DType | None = None,
        rngs: Rngs,
        initializer: Initializer = default_kernel_initializer,
        dot_general: Any = None,
        axis_names: tuple[
            AxisNames | None,
            AxisNames | None,
        ] | None = None,
        shard_mode: ShardMode | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"`rank` must be > 0, got {rank}")

        self.base = base.eval()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if dtype is None:
            dtype = base.weight.value.dtype

        if shard_mode is None:
            shard_mode = base.shard_mode

        a_axis_names = None
        b_axis_names = None

        if axis_names is not None:
            a_axis_names, b_axis_names = axis_names

        self.lora_A = Linear(
            base.in_features,
            rank,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            dot_general=dot_general,
            axis_names=a_axis_names,
        )

        self.lora_B = Linear(
            rank,
            base.out_features,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            initializer=jax.nn.initializers.zeros,
            dot_general=dot_general,
            axis_names=b_axis_names,
        )

        # m = ||W|| over all input-feature axes
        weight = base.weight.value
        input_axes = tuple(range(len(base.in_features)))

        magnitude = jnp.sqrt(
            jnp.sum(
                jnp.square(weight.astype(jnp.float32)),
                axis=input_axes,
            )
        )

        self.magnitude = Parameter(
            magnitude.astype(dtype)
        )

        if b_axis_names is not None:
            self.magnitude.axis_names = b_axis_names[-len(base.out_features):]

    def _direction_norm(self) -> jax.Array:
        """
        Compute

            ||W + sAB||

        without explicitly constructing AB.

        W: [I, O]
        A: [I, R]
        B: [R, O]

        where I/O are flattened input/output feature groups.
        """
        weight = self.base.weight.value.astype(jnp.float32)

        A = self.lora_A.weight.value.astype(jnp.float32)
        B = self.lora_B.weight.value.astype(jnp.float32)

        out_shape = self.base.out_features

        W = weight.reshape(
            (-1, math.prod(out_shape))
        )
        A = A.reshape(
            (-1, self.rank)
        )
        B = B.reshape(
            (self.rank, -1)
        )

        # ||W||²
        w_norm_sq = jnp.sum(
            jnp.square(W),
            axis=0,
        )

        # <W, AB>
        at_w = A.T @ W
        cross = jnp.sum(
            at_w * B,
            axis=0,
        )

        # ||AB||²
        gram = A.T @ A
        ab_norm_sq = jnp.sum(
            B * (gram @ B),
            axis=0,
        )

        s = self.scaling

        norm_sq = (
            w_norm_sq
            + 2.0 * s * cross
            + (s * s) * ab_norm_sq
        )

        norm = jnp.sqrt(
            jnp.maximum(norm_sq, 0.0)
        )

        return norm.reshape(out_shape)

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        out = self.base(
            x,
            out_sharding=out_sharding,
        )

        # DoRA decomposes the weight, not the bias.
        if self.base.has_bias:
            bias = self.base.bias.value
            out = out - bias
        else:
            bias = None

        lora = self.lora_A(x)
        lora = self.lora_B(
            lora,
            out_sharding=out_sharding,
        )

        direction = out + self.scaling * lora

        direction_norm = self._direction_norm()

        magnitude_scale = (
            self.magnitude.value.astype(jnp.float32)
            / direction_norm
        ).astype(direction.dtype)

        out = direction * magnitude_scale

        if bias is not None:
            out = out + bias

        return out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"

class AdaLoRALinear(Module):
    """
    Adaptive Low-Rank Adaptation (AdaLoRA) for Linear layers.

    The weight update is parameterized as:

        ΔW = A diag(E) B

    where E controls the importance / effective rank of each
    singular-value triplet.
    """

    def __init__(
        self,
        base: Linear,
        rank: int = 8,
        alpha: float = 16.0,
        *,
        dtype: DType | None = 'float32',
        rngs: Rngs,
        initializer: Initializer = default_kernel_initializer,
        dot_general: Any = None,
        axis_names: tuple[
            AxisNames | None,
            AxisNames | None,
        ] | None = None,
        shard_mode: ShardMode | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError(
                f"`rank` must be > 0, got {rank}"
            )

        self.base = base.eval()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if shard_mode is None:
            shard_mode = base.shard_mode

        if dot_general is None:
            dot_general = base.dot_general

        a_axis_names = None
        b_axis_names = None

        if axis_names is not None:
            a_axis_names, b_axis_names = axis_names

        self.lora_A = Linear(
            base.in_features,
            rank,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            dot_general=dot_general,
            axis_names=a_axis_names,
        )

        self.lora_B = Linear(
            rank,
            base.out_features,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            dot_general=dot_general,
            axis_names=b_axis_names,
        )

        # Singular-value-like parameters.
        # Zero initialization makes ΔW = 0 initially.
        self.lora_E = Parameter(
            jnp.zeros(
                (rank,),
                dtype=dtype,
            )
        )

        if axis_names is not None:
            rank_axis = None

            if a_axis_names is not None:
                rank_axis = a_axis_names[-1]

            if b_axis_names is not None:
                if (
                    rank_axis is not None
                    and b_axis_names[0] != rank_axis
                ):
                    raise ValueError(
                        "Rank axis names of lora_A and "
                        "lora_B must match."
                    )

                rank_axis = b_axis_names[0]

            if rank_axis is not None:
                self.lora_E.axis_names = (rank_axis,)

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        base = self.base(
            x,
            out_sharding=out_sharding,
        )

        delta = self.lora_A(x)

        delta = delta * self.lora_E.value.astype(delta.dtype)

        delta = self.lora_B(
            delta,
            out_sharding=out_sharding,
        )

        return base + delta * self.scaling

    def mask_rank(
        self,
        mask: jax.Array,
    ) -> None:
        """
        Zero singular values selected for pruning.
        """
        if mask.shape != (self.rank,):
            raise ValueError(
                f"`mask` must have shape {(self.rank,)}, "
                f"got {mask.shape}"
            )

        self.lora_E._value = jnp.where(
            mask,
            self.lora_E.value,
            jnp.zeros_like(self.lora_E.value),
        )

    def orthogonal_loss(self) -> jax.Array:
        """
        Orthogonal regularization for A and B.
        """
        rank = self.rank

        A = self.lora_A.weight.value.reshape(
            (-1, rank)
        ).astype(jnp.float32)

        B = self.lora_B.weight.value.reshape(
            (rank, -1)
        ).astype(jnp.float32)

        eye = jnp.eye(rank, dtype=jnp.float32)

        a_loss = jnp.linalg.norm(
            A.T @ A - eye,
            ord='fro',
        )

        b_loss = jnp.linalg.norm(
            B @ B.T - eye,
            ord='fro',
        )

        return 0.5 * (a_loss + b_loss)

    def extra_repr(self) -> str:
        return (
            f"init_rank={self.rank}, "
            f"alpha={self.alpha}"
        )

class LoHaLinear(Module):
    """
    Low-Rank Hadamard Product (LoHa) adaptation for Linear layers.
    """

    def __init__(
        self,
        base: Linear,
        rank: int = 8,
        alpha: float = 16.0,
        *,
        dtype: DType | None = 'float32',
        rngs: Rngs | None,
        initializer: Initializer = default_kernel_initializer,
        dot_general: Any = None,
        axis_names: tuple[
            AxisNames | None,
            AxisNames | None,
            AxisNames | None,
            AxisNames | None,
        ] | None = None,
        shard_mode: ShardMode | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"`rank` must be > 0, got {rank}")

        if rngs is None:
            raise ValueError(
                "A rngs must be provided to initialize LoHa layer"
            )

        self.base = base.eval()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.dot_general = (
            base.dot_general
            if dot_general is None
            else dot_general
        )
        self.shard_mode = (
            base.shard_mode
            if shard_mode is None
            else shard_mode
        )

        in_shape = base.in_features
        out_shape = base.out_features

        self.loha_A1 = Parameter(
            initializer(
                rngs(),
                in_shape + (rank,),
                dtype,
            )
        )
        self.loha_B1 = Parameter(
            initializer(
                rngs(),
                (rank,) + out_shape,
                dtype,
            )
        )

        self.loha_A2 = Parameter(
            initializer(
                rngs(),
                in_shape + (rank,),
                dtype,
            )
        )
        self.loha_B2 = Parameter(
            jnp.zeros(
                (rank,) + out_shape,
                dtype=dtype,
            )
        )

        if axis_names is not None:
            (
                self.loha_A1.axis_names,
                self.loha_B1.axis_names,
                self.loha_A2.axis_names,
                self.loha_B2.axis_names,
            ) = axis_names

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        rank = self.rank

        A1 = self.loha_A1.value
        B1 = self.loha_B1.value
        A2 = self.loha_A2.value
        B2 = self.loha_B2.value

        # (A1 B1) ⊙ (A2 B2)
        #
        # = A_eff B_eff
        #
        # where the effective latent dimension is rank².

        A = (
            A1[..., :, None]
            * A2[..., None, :]
        ).reshape(
            self.base.in_features + (rank * rank,)
        )

        B = (
            B1[:, None, ...]
            * B2[None, :, ...]
        ).reshape(
            (rank * rank,) + self.base.out_features
        )

        in_dims = len(self.base.in_features)

        dimension_numbers_A = (
            (
                tuple(range(x.ndim - in_dims, x.ndim)),
                tuple(range(in_dims)),
            ),
            ((), ()),
        )

        if self.dot_general is not None:
            delta = self.dot_general(
                x,
                A,
                dimension_numbers_A,
            )
        else:
            delta = jax.lax.dot_general(
                x,
                A,
                dimension_numbers_A,
            )

        dimension_numbers_B = (
            ((delta.ndim - 1,), (0,)),
            ((), ()),
        )


        if self.dot_general is not None:
            delta = self.dot_general(
                delta,
                B,
                dimension_numbers_B,
            )
        else:
            delta = jax.lax.dot_general(
                delta,
                B,
                dimension_numbers_B,
                out_sharding=out_sharding,
            )

        delta = _constrain(
            delta,
            out_sharding,
            self.shard_mode,
        )

        return (
            self.base(
                x,
                out_sharding=out_sharding,
            )
            + delta * self.scaling
        )

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"

def _factorization(
    dimension: int,
    factor: int = -1,
) -> tuple[int, int]:
    if factor > 0 and dimension % factor == 0:
        return factor, dimension // factor

    if factor == -1:
        factor = dimension

    m, n = 1, dimension
    length = m + n

    while m < n:
        new_m = m + 1

        while dimension % new_m != 0:
            new_m += 1

        new_n = dimension // new_m

        if new_m + new_n > length or new_m > factor:
            break

        m, n = new_m, new_n

    if m > n:
        m, n = n, m

    return m, n

class LoKrLinear(Module):
    """
    Low-Rank Kronecker Product (LoKr) adaptation for Linear layers.
    """

    def __init__(
        self,
        base: Linear,
        rank: int = 8,
        alpha: float = 16.0,
        *,
        dtype: DType | None = None,
        rngs: Rngs | None,
        initializer: Initializer = default_kernel_initializer,
        decompose_both: bool = False,
        decompose_factor: int = -1,
        dot_general: Any = None,
        shard_mode: ShardMode | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"`rank` must be > 0, got {rank}")

        if rngs is None:
            raise ValueError(
                "A rngs must be provided to initialize LoKr layer"
            )

        self.base = base.eval()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.decompose_both = decompose_both
        self.decompose_factor = decompose_factor

        self.dot_general = (
            base.dot_general
            if dot_general is None
            else dot_general
        )
        self.shard_mode = (
            base.shard_mode
            if shard_mode is None
            else shard_mode
        )

        if dtype is None:
            dtype = jnp.float32

        in_dim = math.prod(base.in_features)
        out_dim = math.prod(base.out_features)

        in_m, in_n = _factorization(
            in_dim,
            decompose_factor,
        )
        out_l, out_k = _factorization(
            out_dim,
            decompose_factor,
        )

        self.in_m = in_m
        self.in_n = in_n
        self.out_l = out_l
        self.out_k = out_k

        # W1: [in_m, out_l]
        # W2: [in_n, out_k]
        #
        # kron(W1, W2):
        # [in_m * in_n, out_l * out_k]
        #
        # == [in_dim, out_dim]

        self.decompose_w1 = (
            decompose_both
            and rank < max(in_m, out_l) / 2
        )

        self.decompose_w2 = (
            rank < max(in_n, out_k) / 2
        )

        if self.decompose_w1:
            self.lokr_w1_A = Parameter(
                jnp.zeros(
                    (in_m, rank),
                    dtype=dtype,
                )
            )

            self.lokr_w1_B = Parameter(
                initializer(
                    rngs(),
                    (rank, out_l),
                    dtype,
                )
            )
        else:
            self.lokr_w1 = Parameter(
                jnp.zeros(
                    (in_m, out_l),
                    dtype=dtype,
                )
            )

        if self.decompose_w2:
            self.lokr_w2_A = Parameter(
                initializer(
                    rngs(),
                    (in_n, rank),
                    dtype,
                )
            )

            self.lokr_w2_B = Parameter(
                initializer(
                    rngs(),
                    (rank, out_k),
                    dtype,
                )
            )
        else:
            self.lokr_w2 = Parameter(
                initializer(
                    rngs(),
                    (in_n, out_k),
                    dtype,
                )
            )

    def _weights(
        self,
    ) -> tuple[jax.Array, jax.Array]:
        if self.decompose_w1:
            w1 = (
                self.lokr_w1_A.value
                @ self.lokr_w1_B.value
            )
        else:
            w1 = self.lokr_w1.value

        if self.decompose_w2:
            w2 = (
                self.lokr_w2_A.value
                @ self.lokr_w2_B.value
            )
        else:
            w2 = self.lokr_w2.value

        return w1, w2

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        w1, w2 = self._weights()

        in_dims = len(self.base.in_features)

        x = x.reshape(
            x.shape[:-in_dims]
            + (self.in_m, self.in_n)
        )

        # Equivalent to:
        #
        #   x @ kron(w1, w2)
        #
        # without constructing the Kronecker matrix.

        dimension_numbers = (
            ((x.ndim - 1,), (0,)),
            ((), ()),
        )

        if self.dot_general is not None:
            delta = self.dot_general(
                x,
                w2,
                dimension_numbers,
            )
        else:
            delta = jax.lax.dot_general(
                x,
                w2,
                dimension_numbers,
            )

        # delta:
        # [..., in_m, out_k]

        dimension_numbers = (
            ((delta.ndim - 2,), (0,)),
            ((), ()),
        )

        if self.dot_general is not None:
            delta = self.dot_general(
                delta,
                w1,
                dimension_numbers,
            )
        else:
            delta = jax.lax.dot_general(
                delta,
                w1,
                dimension_numbers,
            )

        # dot_general gives:
        #
        # [..., out_k, out_l]
        #
        # but kron ordering is:
        #
        # [..., out_l, out_k]

        delta = jnp.swapaxes(delta, -1, -2)

        delta = delta.reshape(
            delta.shape[:-2]
            + self.base.out_features
        )

        delta = _constrain(
            delta,
            out_sharding,
            self.shard_mode,
        )

        return (
            self.base(
                x.reshape(
                    x.shape[:-2] + self.base.in_features
                ),
                out_sharding=out_sharding,
            )
            + delta * self.scaling
        )

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"

class VeRALinear(Module):
    """
    Vector-based Random Matrix Adaptation (VeRA) for Linear layers.

    The weight update is parameterized as:

        ΔW = diag(lambda_d) A B diag(lambda_b)

    where A and B are frozen random projections, while lambda_d
    and lambda_b are trainable.
    """

    def __init__(
        self,
        base: Linear,
        rank: int = 8,
        *,
        dtype: DType | None = None,
        d_initial: float = 0.1,
        vera_A: Parameter,
        vera_B: Parameter,
        dot_general: Any = None,
        shard_mode: ShardMode | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"`rank` must be > 0, got {rank}")

        self.base = base.eval()
        self.rank = rank

        self.dot_general = (
            base.dot_general
            if dot_general is None
            else dot_general
        )
        self.shard_mode = (
            base.shard_mode
            if shard_mode is None
            else shard_mode
        )

        if dtype is None:
            dtype = jnp.float32

        in_features = math.prod(base.in_features)
        out_features = math.prod(base.out_features)

        # A/B may be shared across multiple VeRA layers.
        #
        # Shared projections can be larger than this layer;
        # the required slices are selected during forward.

        if vera_A.value.shape[0] < in_features:
            raise ValueError(
                "`vera_A` does not have enough input features"
            )

        if vera_A.value.shape[1] < rank:
            raise ValueError(
                "`vera_A` does not have enough rank dimensions"
            )

        if vera_B.value.shape[0] < rank:
            raise ValueError(
                "`vera_B` does not have enough rank dimensions"
            )

        if vera_B.value.shape[1] < out_features:
            raise ValueError(
                "`vera_B` does not have enough output features"
            )

        self.vera_A = vera_A
        self.vera_B = vera_B

        # Trainable vectors.
        #
        # lambda_d scales the rank dimension.
        self.vera_lambda_d = Parameter(
            jnp.full(
                (rank,),
                d_initial,
                dtype=dtype,
            )
        )

        # lambda_b scales the output dimension.
        #
        # Zero-init makes the adapter a no-op initially.
        self.vera_lambda_b = Parameter(
            jnp.zeros(
                base.out_features,
                dtype=dtype,
            )
        )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        base = self.base(
            x,
            out_sharding=out_sharding,
        )

        in_dims = len(self.base.in_features)
        in_features = math.prod(self.base.in_features)
        out_features = math.prod(self.base.out_features)

        x_flat = x.reshape(
            x.shape[:-in_dims] + (in_features,)
        )

        A = self.vera_A.value[
            :in_features,
            :self.rank,
        ].astype(x.dtype)

        B = self.vera_B.value[
            :self.rank,
            :out_features,
        ].astype(x.dtype)

        # x A
        dimension_numbers = (
            ((x_flat.ndim - 1,), (0,)),
            ((), ()),
        )

        if self.dot_general is not None:
            delta = self.dot_general(
                x_flat,
                A,
                dimension_numbers,
            )
        else:
            delta = jax.lax.dot_general(
                x_flat,
                A,
                dimension_numbers,
            )

        # Scale the random rank basis.
        delta *= self.vera_lambda_d.value.astype(delta.dtype)

        # (...) B
        dimension_numbers = (
            ((delta.ndim - 1,), (0,)),
            ((), ()),
        )

        if self.dot_general is not None:
            delta = self.dot_general(
                delta,
                B,
                dimension_numbers,
            )
        else:
            delta = jax.lax.dot_general(
                delta,
                B,
                dimension_numbers,
            )

        delta = delta.reshape(
            delta.shape[:-1] + self.base.out_features
        )

        delta *= self.vera_lambda_b.value.astype(delta.dtype)

        delta = _constrain(
            delta,
            out_sharding,
            self.shard_mode,
        )

        return base + delta

    def extra_repr(self) -> str:
        return f"rank={self.rank}"


__all__ = [
    'AdaLoRALinear',
    'DoRALinear',
    'LoHaLinear',
    'LoKrLinear',
    'LoRALinear',
    'VeRALinear'
]
