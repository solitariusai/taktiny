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
import qwix
from jax.lax import PrecisionLike
from jax.sharding import NamedSharding, PartitionSpec
from jax.typing import DTypeLike

from taktiny.nn.base import Module, Parameter
from taktiny.nn.modules.linear import (
    Linear,
    default_bias_initializer,
    default_kernel_initializer,
)
from taktiny.nn.rng import Rngs
from taktiny.nn.utils import _constrain, _validate_integer
from taktiny.utils.spmd import logical_to_mesh_axes, with_logical_partitioning
from taktiny.utils.typing import (
    AxisNames,
    DotGeneral,
    DType,
    Initializer,
    MetaData,
    QuantConfig,
)


class LoRALinear(Module):
    """Add a low-rank update to an existing Linear module.

    Computes ``base(x) + (alpha / rank) * lora_B(lora_A(x))``.
    A contracts all input-feature axes into the rank dimension; B projects
    that dimension into all output-feature axes. B's kernel starts at zero,
    so the default zero bias makes the initial result equal to ``base(x)``.
    The base is put in evaluation mode; this does not freeze its gradients.
    Select adapter parameters in the optimizer when training only LoRA.

    Omitted (or None) dtype, bias, metadata, and dot settings inherit from
    the base. When both sharding arguments are None, the base's logical
    labels and partition specification are inherited. Linear resolves those
    labels using the current mapping rules, overriding the inherited spec.
    Outside the mapping context, unmapped labels therefore become replicated.
    A base without logical names supplies its physical specification instead.
    Providing either sharding argument selects an adapter-specific layout.
    Use ``partition_spec=PartitionSpec()`` for replicated adapters.
    Initializers and quantization keep their adapter defaults because Linear
    does not retain their original constructor configurations.

    Args:
        base: Existing Linear, including one with N-D feature shapes.
        rank: Positive adapter rank.
        alpha: Finite scaling numerator; the update is scaled by alpha/rank.
        rngs: Random stream used to initialize adapter parameters.
        bias: Create an output bias in B. A is always bias-free. The base
            bias is unaffected. None inherits whether the base has a bias.
            Set False for the usual bias-free LoRA update.
        dtype: Adapter initializer dtype. None inherits the base kernel dtype
            (the scale dtype for a quantized base).
        kernel_initializer: Initializer for A; B always uses zeros.
        bias_initializer: Initializer for B's optional bias. A nonzero bias
            makes the initial adapter update nonzero when alpha is nonzero.
        quant: Optional Qwix configuration for adapter kernels only. Defaults
            to dense adapters even when the base is quantized.
        dot_general: Custom dot operation for non-quantized adapter kernels;
            None inherits the base implementation.
        axis_names: Logical names in base-kernel order: input axes followed
            by output axes. A and B receive their respective feature names
            and an unnamed rank axis.
        partition_spec: Explicit base-kernel specification, split between
            A and B with a replicated rank axis. Omitted trailing entries
            are replicated. Logical mappings override explicit specs.
        kernel_metadata: Metadata attached to both adapter kernels. None
            copies the base kernel metadata; an empty dict clears it.
        bias_metadata: Metadata attached to B's bias. None copies the base
            bias metadata, when present; an empty dict clears it.
        precision: Dot precision for both adapter projections; None inherits
            the base precision.
        preferred_element_type: Preferred adapter accumulation/result dtype;
            None inherits the base setting.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> base = nn.Linear((2, 3), (4, 5), rngs=nn.Rngs(0))
        >>> layer = nn.LoRALinear(base, 2, 4.0, rngs=nn.Rngs(1))
        >>> layer(jnp.ones((7, 2, 3))).shape
        (7, 4, 5)
    """

    def __init__(
        self,
        base: Linear,
        rank: int,
        alpha: float,
        *,
        bias: bool | None = None,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_kernel_initializer,
        bias_initializer: Initializer = default_bias_initializer,
        quant: QuantConfig = None,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        bias_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        if not isinstance(base, Linear):
            raise TypeError('base must be a Linear module')

        self.rank = _validate_integer(rank, 'rank')
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError('alpha must be a finite real number')

        if not math.isfinite(alpha):
            raise ValueError('alpha must be finite')

        if bias is None:
            bias = base.bias is not None

        if not isinstance(bias, bool):
            raise TypeError('bias must be a boolean')

        self.alpha = alpha
        self.scaling = alpha / rank

        if dtype is None:
            dtype = base.kernel.value.dtype

        if dot_general is None:
            dot_general = base.dot_general

        if precision is None:
            precision = base.precision

        if preferred_element_type is None:
            preferred_element_type = base.preferred_element_type

        if kernel_metadata is None:
            kernel_metadata = base.kernel.metadata
            
        if bias_metadata is None and base.bias is not None:
            bias_metadata = base.bias.metadata

        inherit_sharding = axis_names is None and partition_spec is None
        if inherit_sharding:
            axis_names = base.kernel.axis_names
            partition_spec = base.kernel.partition_spec
            # Also recognize kernels sharded after construction, without
            # logical/partition metadata on the Parameter.
            if partition_spec is None:
                sharding = getattr(base.kernel.value, 'sharding', None)
                if isinstance(sharding, NamedSharding):
                    partition_spec = sharding.spec

        input_dims = len(base.in_features)
        kernel_dims = input_dims + len(base.out_features)
        a_axis_names = None
        b_axis_names = None
        if axis_names is not None:
            if isinstance(axis_names, PartitionSpec):
                raise TypeError('use partition_spec for a PartitionSpec')

            if len(axis_names) != kernel_dims:
                raise ValueError('axis_names must match the base kernel rank')

            if any(name is not None and not isinstance(name, str)
                   for name in axis_names):
                raise TypeError('axis_names entries must be strings or None')

            a_axis_names = tuple(axis_names[:input_dims]) + (None,)
            b_axis_names = (None,) + tuple(axis_names[input_dims:])

        a_partition_spec = None
        b_partition_spec = None
        if partition_spec is not None:
            if len(partition_spec) > kernel_dims:
                raise ValueError('partition_spec exceeds the base kernel rank')

            entries = tuple(partition_spec) + (None,) * (
                kernel_dims - len(partition_spec)
            )
            a_partition_spec = PartitionSpec(*entries[:input_dims], None)
            b_partition_spec = PartitionSpec(None, *entries[input_dims:])

        self.base = base.eval()

        self.lora_A = Linear(
            base.in_features,
            rank,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=kernel_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=a_axis_names,
            partition_spec=a_partition_spec,
            kernel_metadata=kernel_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

        self.lora_B = Linear(
            rank,
            base.out_features,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            kernel_initializer=jax.nn.initializers.zeros,
            bias_initializer=bias_initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=b_axis_names,
            partition_spec=b_partition_spec,
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
            precision=precision,
            preferred_element_type=preferred_element_type,
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

        return _constrain(out + lora * self.scaling, out_sharding)

    def extra_repr(self) -> str:
        in_shape = '×'.join(map(str, self.base.in_features))
        out_shape = '×'.join(map(str, self.base.out_features))
        return (
            f'{in_shape} ➤ {out_shape}, rank={self.rank}, alpha={self.alpha:g}'
        )


def _dense(value: Any) -> jax.Array:
    """Read a floating kernel, dequantizing a Qwix base when necessary."""
    return qwix.dequantize(value) if isinstance(value, qwix.QArray) else value


def _shard_adapter(
    value: jax.Array,
    sharding: jax.sharding.Sharding | PartitionSpec | None,
) -> jax.Array:
    """Move explicit layouts; constrain layouts managed by the compiler."""
    if sharding is None:
        return value
    mesh = (sharding.mesh if isinstance(sharding, NamedSharding)
            else jax.sharding.get_abstract_mesh())
    if mesh.are_all_axes_explicit and isinstance(sharding, (NamedSharding, PartitionSpec)):
        return jax.sharding.reshard(value, sharding)
    return jax.lax.with_sharding_constraint(value, sharding)


class _WeightAdapter(Module):
    """Common configuration for bias-free weight updates.

    Missing settings inherit from the base. Logical names are always resolved
    in the current context. Initializers create new adapter weights; they do
    not copy base values. Quantization is supported on the base, while the
    trainable adapter parameters remain floating point.
    """

    def _configure(
        self,
        base: Linear,
        rank: int,
        alpha: float,
        dtype: DType | None,
        dot_general: DotGeneral | None,
        axis_names: AxisNames | None,
        partition_spec: PartitionSpec | None,
        metadata: MetaData | None,
        precision: PrecisionLike,
        preferred_element_type: DTypeLike | None,
    ) -> None:
        if not isinstance(base, Linear):
            raise TypeError('base must be a Linear module')
        self.rank = _validate_integer(rank, 'rank')
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError('alpha must be a finite real number')
        if not math.isfinite(alpha):
            raise ValueError('alpha must be finite')
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dtype = base.kernel.value.dtype if dtype is None else dtype
        if not jnp.issubdtype(jnp.dtype(self.dtype), jnp.floating):
            raise TypeError('adapter dtype must be floating point')
        self.dot_general = base.dot_general if dot_general is None else dot_general
        self.precision = base.precision if precision is None else precision
        self.preferred_element_type = (
            base.preferred_element_type
            if preferred_element_type is None else preferred_element_type
        )
        self.kernel_metadata = (
            base.kernel.metadata if metadata is None else dict(metadata)
        )
        if axis_names is None and partition_spec is None:
            axis_names = base.kernel.axis_names
            partition_spec = base.kernel.partition_spec
            if partition_spec is None:
                sharding = getattr(base.kernel.value, 'sharding', None)
                if isinstance(sharding, NamedSharding):
                    partition_spec = sharding.spec
        ndim = len(base.in_features) + len(base.out_features)
        if axis_names is not None:
            if isinstance(axis_names, PartitionSpec):
                raise TypeError('use partition_spec for a PartitionSpec')
            if len(axis_names) != ndim:
                raise ValueError('axis_names must match the base kernel rank')
            if any(name is not None and not isinstance(name, str)
                   for name in axis_names):
                raise TypeError('axis_names entries must be strings or None')
            axis_names = tuple(axis_names)
        if partition_spec is not None:
            if len(partition_spec) > ndim:
                raise ValueError('partition_spec exceeds the base kernel rank')
            partition_spec = PartitionSpec(
                *partition_spec, *((None,) * (ndim - len(partition_spec)))
            )
        self.axis_names = axis_names
        self.partition_spec = partition_spec
        self.base = base.eval()

    def _axes(
        self, indices: tuple[int | None, ...],
    ) -> tuple[AxisNames | None, PartitionSpec | None]:
        names = None if self.axis_names is None else tuple(
            None if index is None else self.axis_names[index]
            for index in indices
        )
        spec = None if self.partition_spec is None else PartitionSpec(*(
            None if index is None else self.partition_spec[index]
            for index in indices
        ))
        return names, spec

    def _parameter(
        self, initializer: Initializer, rngs: Rngs,
        shape: tuple[int, ...], indices: tuple[int | None, ...],
    ) -> Parameter:
        names, spec = self._axes(indices)
        init = with_logical_partitioning(initializer, names, spec)
        return Parameter(
            init(rngs(), shape, self.dtype), axis_names=names,
            partition_spec=spec, metadata=self.kernel_metadata,
        )

    def _linear(
        self, output: bool, initializer: Initializer, rngs: Rngs,
    ) -> Linear:
        n = len(self.base.in_features)
        m = len(self.base.out_features)
        indices = (
            (None,) + tuple(range(n, n + m))
            if output else tuple(range(n)) + (None,)
        )
        names, spec = self._axes(indices)
        return Linear(
            self.rank if output else self.base.in_features,
            self.base.out_features if output else self.rank,
            bias=False, dtype=self.dtype, rngs=rngs,
            kernel_initializer=initializer,
            dot_general=self.dot_general, axis_names=names,
            partition_spec=spec, kernel_metadata=self.kernel_metadata,
            precision=self.precision,
            preferred_element_type=self.preferred_element_type,
        )

    def _dot(
        self, lhs: jax.Array, rhs: jax.Array,
        axes: tuple[tuple[int, ...], tuple[int, ...]],
    ) -> jax.Array:
        dimensions = (axes, ((), ()))
        if self.dot_general is not None:
            return self.dot_general(
                lhs, rhs, dimensions, precision=self.precision,
                preferred_element_type=self.preferred_element_type,
                out_sharding=None,
            )
        return jax.lax.dot_general(
            lhs, rhs, dimensions, precision=self.precision,
            preferred_element_type=self.preferred_element_type,
        )

    def extra_repr(self) -> str:
        source = '×'.join(map(str, self.base.in_features))
        target = '×'.join(map(str, self.base.out_features))
        return f'{source} ➤ {target}, rank={self.rank}, alpha={self.alpha:g}'


class DoRALinear(_WeightAdapter):
    """Adapt weight direction with LoRA and learn a separate output magnitude.

    Computes a column-normalized update to the base kernel. The base bias is
    added after magnitude scaling. A uses kernel_initializer and B starts at
    zero. Magnitude starts at the base column norm, so initialization preserves
    the base output. Zero columns are handled with a finite norm floor.

    Dtype, metadata, dot settings and sharding inherit as in LoRALinear.
    axis_names/partition_spec describe the base kernel's input then output
    axes; the adapter rank is replicated and magnitude uses output axes.
    Adapter factors are bias-free and dense, including for a quantized base.
    Base gradients are not frozen automatically; select adapter parameters
    in the optimizer. The direction norm is detached during differentiation.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> base = nn.Linear((2, 3), 4, rngs=nn.Rngs(0))
        >>> nn.DoRALinear(base, 2, 4.0, rngs=nn.Rngs(1))(jnp.ones((5, 2, 3))).shape
        (5, 4)

    Reference:
        Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024).
        https://arxiv.org/abs/2402.09353
    """

    def __init__(
        self,
        base: Linear,
        rank: int,
        alpha: float,
        *,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_kernel_initializer,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self._configure(
            base, rank, alpha, dtype, dot_general, axis_names, partition_spec,
            kernel_metadata, precision, preferred_element_type,
        )
        self.lora_A = self._linear(False, kernel_initializer, rngs)
        self.lora_B = self._linear(True, jax.nn.initializers.zeros, rngs)
        n = len(base.in_features)
        magnitude = jnp.sqrt(jnp.sum(
            jnp.square(_dense(base.kernel.value).astype(jnp.float32)),
            axis=tuple(range(n)),
        )).astype(self.dtype)
        names, spec = self._axes(tuple(range(n, n + len(base.out_features))))
        self.magnitude = Parameter(
            magnitude, axis_names=names, partition_spec=spec,
            metadata=self.kernel_metadata,
        )
        # Parameter metadata alone does not shard on every mesh backend.
        resolved = logical_to_mesh_axes(names) if names is not None else spec
        if resolved is not None:
            self.magnitude._value = jax.jit(
                lambda value: value, out_shardings=resolved,
            )(magnitude) if not jax.sharding.get_abstract_mesh().empty else magnitude

    def _direction_norm(self) -> jax.Array:
        """Column norms without materializing the full low-rank update."""
        # Keep feature axes separate so explicit meshes need no ambiguous
        # reshape of multiple sharded dimensions.
        inputs = tuple(range(len(self.base.in_features)))
        w = _dense(self.base.kernel.value).astype(jnp.float32)
        a = self.lora_A.kernel.value.astype(jnp.float32)
        b = self.lora_B.kernel.value.astype(jnp.float32)
        cross = jnp.sum(self._dot(a, w, (inputs, inputs)) * b, axis=0)
        gram = self._dot(a, a, (inputs, inputs))
        norm_sq = (
            jnp.sum(w * w, axis=inputs) + 2 * self.scaling * cross
            + self.scaling**2 * jnp.sum(
                b * self._dot(gram, b, ((1,), (0,))), axis=0,
            )
        )
        return jnp.sqrt(jnp.maximum(norm_sq, 1e-12))

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        output = self.base(x)
        bias = None if self.base.bias is None else self.base.bias.value
        if bias is not None:
            output = output - bias
        delta = self.lora_B(self.lora_A(x))
        norm = jax.lax.stop_gradient(self._direction_norm())
        scale = (self.magnitude.value.astype(jnp.float32) / norm).astype(output.dtype)
        output = (output + self.scaling * delta) * scale
        if bias is not None:
            output = output + bias
        return _shard_adapter(output, out_sharding)


class AdaLoRALinear(_WeightAdapter):
    """An SVD-style adapter: base(x) + alpha/rank * B(E * A(x)).

    A and B use kernel_initializer; the replicated rank vector E starts at
    zero. mask_rank performs one pruning update. Budget scheduling, importance
    estimation and adding orthogonal_loss to a training objective are the
    caller's responsibility.

    Dtype, kernel metadata, dot settings and sharding inherit as in LoRALinear.
    Logical axes are resolved in the current context; the rank stays replicated.
    No adapter biases are created. A quantized base may be wrapped, while the
    new factors remain floating point.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> layer = nn.AdaLoRALinear(nn.Linear(3, 4, rngs=nn.Rngs(0)), 2, 4.0, rngs=nn.Rngs(1))
        >>> layer(jnp.ones((5, 3))).shape
        (5, 4)

    Reference:
        Zhang et al., "AdaLoRA: Adaptive Budget Allocation for
        Parameter-Efficient Fine-Tuning" (2023).
        https://arxiv.org/abs/2303.10512
    """

    def __init__(
        self,
        base: Linear,
        rank: int,
        alpha: float,
        *,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_kernel_initializer,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self._configure(
            base, rank, alpha, dtype, dot_general, axis_names, partition_spec,
            kernel_metadata, precision, preferred_element_type,
        )
        self.lora_A = self._linear(False, kernel_initializer, rngs)
        self.lora_B = self._linear(True, kernel_initializer, rngs)
        self.lora_E = self._parameter(
            jax.nn.initializers.zeros, rngs, (rank,), (None,),
        )

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        delta = self.lora_A(x)
        delta = self.lora_B(delta * self.lora_E.value.astype(delta.dtype))
        return _shard_adapter(self.base(x) + self.scaling * delta, out_sharding)

    def mask_rank(self, mask: jax.Array) -> None:
        """Zero E entries where the boolean mask is False, without resizing.

        Later optimizer steps can regrow entries; reapply the mask when
        permanent pruning is required.
        """
        mask = jnp.asarray(mask)
        if mask.shape != (self.rank,):
            raise ValueError(f'mask must have shape {(self.rank,)}')
        if mask.dtype != jnp.bool_:
            raise TypeError('mask must be boolean')
        self.lora_E._value = jnp.where(mask, self.lora_E.value, 0)

    def orthogonal_loss(self) -> jax.Array:
        """Mean Frobenius norm of A.T A - I and B B.T - I."""
        a = self.lora_A.kernel.value.astype(jnp.float32)
        b = self.lora_B.kernel.value.astype(jnp.float32)
        inputs = tuple(range(len(self.base.in_features)))
        outputs = tuple(range(1, 1 + len(self.base.out_features)))
        eye = jnp.eye(self.rank, dtype=jnp.float32)
        return 0.5 * (
            jnp.linalg.norm(self._dot(a, a, (inputs, inputs)) - eye)
            + jnp.linalg.norm(self._dot(b, b, (outputs, outputs)) - eye)
        )


class LoHaLinear(_WeightAdapter):
    """Hadamard-product adapter with delta W = (A1 B1) * (A2 B2).

    The forward pass uses an effective rank of rank squared, avoiding a full
    input-by-output update kernel. B2 starts at zero; the other factors use
    kernel_initializer. The base bias is unchanged.

    All factors support N-D feature shapes, inherited dtype and metadata, and
    base-kernel logical/physical specifications with a replicated rank axis.
    Current logical rules take precedence, as in Linear. Quantized bases are
    supported with dense adapter factors.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> layer = nn.LoHaLinear(nn.Linear(3, 4, rngs=nn.Rngs(0)), 2, 4.0, rngs=nn.Rngs(1))
        >>> layer(jnp.ones((5, 3))).shape
        (5, 4)

    Reference:
        Yeh et al., "Navigating Text-To-Image Customization:
        From LyCORIS Fine-Tuning to Model Evaluation" (2024).
        https://arxiv.org/abs/2309.14859
    """

    def __init__(
        self,
        base: Linear,
        rank: int,
        alpha: float,
        *,
        dtype: DType | None = None,
        rngs: Rngs,
        kernel_initializer: Initializer = default_kernel_initializer,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self._configure(
            base, rank, alpha, dtype, dot_general, axis_names, partition_spec,
            kernel_metadata, precision, preferred_element_type,
        )
        n = len(base.in_features)
        a_indices = tuple(range(n)) + (None,)
        b_indices = (None,) + tuple(range(n, n + len(base.out_features)))
        self.loha_A1 = self._parameter(
            kernel_initializer, rngs, base.in_features + (rank,), a_indices,
        )
        self.loha_B1 = self._parameter(
            kernel_initializer, rngs, (rank,) + base.out_features, b_indices,
        )
        self.loha_A2 = self._parameter(
            kernel_initializer, rngs, base.in_features + (rank,), a_indices,
        )
        self.loha_B2 = self._parameter(
            jax.nn.initializers.zeros, rngs, (rank,) + base.out_features, b_indices,
        )

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        a = (self.loha_A1.value[..., :, None] *
             self.loha_A2.value[..., None, :]).reshape(
                 self.base.in_features + (self.rank**2,)
             )
        b = (self.loha_B1.value[:, None, ...] *
             self.loha_B2.value[None, :, ...]).reshape(
                 (self.rank**2,) + self.base.out_features
             )
        n = len(self.base.in_features)
        delta = self._dot(
            x, a, (tuple(range(x.ndim - n, x.ndim)), tuple(range(n))),
        )
        delta = self._dot(delta, b, ((delta.ndim - 1,), (0,)))
        return _shard_adapter(self.base(x) + self.scaling * delta, out_sharding)


def _factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Find factors, using an exact requested divisor when possible."""
    if factor > 0 and dimension % factor == 0:
        return factor, dimension // factor
    limit = math.isqrt(dimension) if factor == -1 else min(
        factor, math.isqrt(dimension),
    )
    for divisor in range(limit, 0, -1):
        if dimension % divisor == 0:
            return divisor, dimension // divisor
    return 1, dimension


class LoKrLinear(_WeightAdapter):
    """Kronecker-product weight adapter, delta W = kron(W1, W2).

    decompose_factor controls the split of flattened input/output widths;
    -1 selects balanced factors. Small ranks factorize W2, and
    decompose_both=True also permits factorizing W1. W1 (or its A factor)
    starts at zero; other factors use kernel_initializer.

    Dtype, metadata, dot settings, and the effective kernel's sharding inherit
    from the base. Kronecker factors are replicated because flattened factor
    dimensions do not correspond to the original N-D feature axes. A sharded
    effective kernel is materialized when a non-replicated layout is requested.
    Otherwise the forward pass contracts the two small factors directly.
    Base biases are unchanged; adapter factors stay dense for quantized bases.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> layer = nn.LoKrLinear(nn.Linear((2, 3), 4, rngs=nn.Rngs(0)), 2, 4.0, rngs=nn.Rngs(1))
        >>> layer(jnp.ones((5, 2, 3))).shape
        (5, 4)

    Reference:
        Yeh et al., "Navigating Text-To-Image Customization:
        From LyCORIS Fine-Tuning to Model Evaluation" (2024).
        https://arxiv.org/abs/2309.14859
    """

    def __init__(
        self,
        base: Linear,
        rank: int,
        alpha: float,
        *,
        dtype: DType | None = None,
        rngs: Rngs,
        decompose_both: bool = False,
        decompose_factor: int = -1,
        kernel_initializer: Initializer = default_kernel_initializer,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        kernel_metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self._configure(
            base, rank, alpha, dtype, dot_general, axis_names, partition_spec,
            kernel_metadata, precision, preferred_element_type,
        )
        if not isinstance(decompose_both, bool):
            raise TypeError('decompose_both must be a boolean')
        if (isinstance(decompose_factor, bool)
                or not isinstance(decompose_factor, int)
                or (decompose_factor != -1 and decompose_factor <= 0)):
            raise ValueError('decompose_factor must be -1 or a positive integer')
        self.decompose_both = decompose_both
        self.decompose_factor = decompose_factor
        self.in_m, self.in_n = _factorization(
            math.prod(base.in_features), decompose_factor,
        )
        self.out_l, self.out_k = _factorization(
            math.prod(base.out_features), decompose_factor,
        )
        self.decompose_w1 = decompose_both and rank < max(self.in_m, self.out_l) / 2
        self.decompose_w2 = rank < max(self.in_n, self.out_k) / 2
        self.lokr_w1 = self.lokr_w1_A = self.lokr_w1_B = None
        self.lokr_w2 = self.lokr_w2_A = self.lokr_w2_B = None
        if self.decompose_w1:
            self.lokr_w1_A = self._parameter(
                jax.nn.initializers.zeros, rngs, (self.in_m, rank), (None, None),
            )
            self.lokr_w1_B = self._parameter(
                kernel_initializer, rngs, (rank, self.out_l), (None, None),
            )
        else:
            self.lokr_w1 = self._parameter(
                jax.nn.initializers.zeros, rngs, (self.in_m, self.out_l),
                (None, None),
            )
        if self.decompose_w2:
            self.lokr_w2_A = self._parameter(
                kernel_initializer, rngs, (self.in_n, rank), (None, None),
            )
            self.lokr_w2_B = self._parameter(
                kernel_initializer, rngs, (rank, self.out_k), (None, None),
            )
        else:
            self.lokr_w2 = self._parameter(
                kernel_initializer, rngs, (self.in_n, self.out_k), (None, None),
            )

    def _weights(self) -> tuple[jax.Array, jax.Array]:
        if self.lokr_w1 is not None:
            w1 = self.lokr_w1.value
        else:
            assert self.lokr_w1_A is not None and self.lokr_w1_B is not None
            w1 = self._dot(self.lokr_w1_A.value, self.lokr_w1_B.value, ((1,), (0,)))
        if self.lokr_w2 is not None:
            w2 = self.lokr_w2.value
        else:
            assert self.lokr_w2_A is not None and self.lokr_w2_B is not None
            w2 = self._dot(self.lokr_w2_A.value, self.lokr_w2_B.value, ((1,), (0,)))
        return w1, w2

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        w1, w2 = self._weights()
        n = len(self.base.in_features)
        spec = (
            logical_to_mesh_axes(self.axis_names)
            if self.axis_names is not None else self.partition_spec
        )
        if spec is not None and any(axis is not None for axis in spec):
            kernel = jnp.kron(w1, w2).reshape(
                self.base.in_features + self.base.out_features
            )
            kernel = _shard_adapter(kernel, spec)
            delta = self._dot(
                x, kernel, (tuple(range(x.ndim - n, x.ndim)), tuple(range(n))),
            )
        else:
            flat = x.reshape(x.shape[:-n] + (self.in_m, self.in_n))
            delta = self._dot(flat, w2, ((flat.ndim - 1,), (0,)))
            delta = self._dot(delta, w1, ((delta.ndim - 2,), (0,)))
            delta = jnp.swapaxes(delta, -1, -2).reshape(
                x.shape[:-n] + self.base.out_features
            )
        return _shard_adapter(self.base(x) + self.scaling * delta, out_sharding)


class VeRALinear(_WeightAdapter):
    """Learn rank/output scales around shared, fixed random projections.

    Computes base(x) + ((x A) * lambda_d) B * lambda_b. Supplied A and B are
    two-dimensional Parameters and may be larger than this layer: prefixes
    matching flattened feature widths and rank are selected. Their values,
    metadata, and sharding are not modified. stop_gradient makes them fixed
    in differentiation; exclude them from optimizer weight decay as well.

    lambda_d is replicated and initialized to d_initial (default 0.1).
    lambda_b starts at zero and uses the base output feature shape and axes.
    Dtype, metadata, dot settings and logical/physical axes inherit from base.
    Logical names are resolved in the current context. No RNG is needed:
    randomness is entirely in the supplied projections. Quantized bases and
    supplied Qwix projections are read in floating point for the adapter.

    Example:
        >>> from taktiny import nn
        >>> import jax.numpy as jnp
        >>> base = nn.Linear(3, 4, rngs=nn.Rngs(0))
        >>> a = nn.Parameter(jnp.ones((3, 2)), trainable=False)
        >>> b = nn.Parameter(jnp.ones((2, 4)), trainable=False)
        >>> layer = nn.VeRALinear(base, 2, vera_A=a, vera_B=b)
        >>> layer(jnp.ones((5, 3))).shape
        (5, 4)

    Reference:
        Kopiczko et al., "VeRA: Vector-based Random Matrix Adaptation" (2024).
        https://arxiv.org/abs/2310.11454
    """

    def __init__(
        self,
        base: Linear,
        rank: int,
        *,
        vera_A: Parameter,
        vera_B: Parameter,
        dtype: DType | None = None,
        d_initial: float = 0.1,
        dot_general: DotGeneral | None = None,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        metadata: MetaData | None = None,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self._configure(
            base, rank, float(rank), dtype, dot_general, axis_names,
            partition_spec, metadata, precision, preferred_element_type,
        )
        if (isinstance(d_initial, bool) or not isinstance(d_initial, (int, float))
                or not math.isfinite(d_initial)):
            raise ValueError('d_initial must be finite')
        for name, projection, shape in (
            ('vera_A', vera_A, (math.prod(base.in_features), rank)),
            ('vera_B', vera_B, (rank, math.prod(base.out_features))),
        ):
            if not isinstance(projection, Parameter):
                raise TypeError(f'{name} must be a Parameter')
            if projection.value.ndim != 2:
                raise ValueError(f'{name} must be a matrix')
            if any(actual < required for actual, required in zip(
                projection.value.shape, shape,
            )):
                raise ValueError(f'{name} must have at least shape {shape}')
            if not jnp.issubdtype(projection.value.dtype, jnp.floating):
                raise TypeError(f'{name} must have a floating-point dtype')
        self.vera_A = vera_A
        self.vera_B = vera_B
        rngs = Rngs(0)  # Constant initializers ignore the key.
        self.vera_lambda_d = self._parameter(
            jax.nn.initializers.constant(d_initial), rngs, (rank,), (None,),
        )
        n = len(base.in_features)
        self.vera_lambda_b = self._parameter(
            jax.nn.initializers.zeros, rngs, base.out_features,
            tuple(range(n, n + len(base.out_features))),
        )

    def __call__(
        self, x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        n = len(self.base.in_features)
        in_width, out_width = math.prod(self.base.in_features), math.prod(self.base.out_features)
        a = jax.lax.stop_gradient(_dense(self.vera_A.value))[
            :in_width, :self.rank,
        ].astype(self.dtype)
        b = jax.lax.stop_gradient(_dense(self.vera_B.value))[
            :self.rank, :out_width,
        ].astype(self.dtype)
        flat = x.reshape(x.shape[:-n] + (in_width,))
        delta = self._dot(flat, a, ((flat.ndim - 1,), (0,)))
        delta = delta * self.vera_lambda_d.value.astype(delta.dtype)
        delta = self._dot(delta, b, ((delta.ndim - 1,), (0,)))
        delta = delta.reshape(x.shape[:-n] + self.base.out_features)
        delta = delta * self.vera_lambda_b.value.astype(delta.dtype)
        return _shard_adapter(self.base(x) + delta, out_sharding)

    def extra_repr(self) -> str:
        source = '×'.join(map(str, self.base.in_features))
        target = '×'.join(map(str, self.base.out_features))
        return f'{source} ➤ {target}, rank={self.rank}'


__all__ = [
    'AdaLoRALinear', 
    'DoRALinear', 
    'LoHaLinear', 
    'LoKrLinear',
    'LoRALinear', 
    'VeRALinear',
]
