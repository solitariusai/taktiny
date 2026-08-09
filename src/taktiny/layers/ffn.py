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
"""
Feed Forward Network modules
TODO: rewrite this file
"""
from __future__ import annotations
from typing import Any
import jax, jax.numpy as jnp

from taktiny.utils.typing import AxisNames, ShardMode
from taktiny import nn
from taktiny.nn.utils import _resolve_activation
from taktiny.utils.typing import Activation, DType

class GateMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Activation = jax.nn.silu,
        bias: bool = False,
        dtype: DType | None = None,
        rngs: nn.Rngs | None = None,
        gate_axis_names: AxisNames | None = None,
        up_axis_names: AxisNames | None = None,
        down_axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
    ) -> None:
        self.activation = _resolve_activation(activation)

        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size,
            bias=bias, dtype=dtype, rngs=rngs,
            axis_names=gate_axis_names,
            shard_mode=shard_mode,
            quant=quant, dot_general=dot_general
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size,
            bias=bias, dtype=dtype, rngs=rngs,
            axis_names=up_axis_names,
            shard_mode=shard_mode,
            quant=quant, dot_general=dot_general
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size,
            bias=bias, dtype=dtype, rngs=rngs,
            axis_names=down_axis_names,
            shard_mode=shard_mode,
            quant=quant, dot_general=dot_general
        )

    def __call__(self, x: jax.Array, out_sharding: Any=None) -> jax.Array:
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up, out_sharding=out_sharding)

class FusedGateMLP(nn.Module):
    """
    GateMLP where the gate and up projections are fused into a single linear layer.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Activation = jax.nn.silu,
        bias: bool = False,
        dtype: str | None = None,
        seed: nn.Rngs | None = None,
        linear_in_axis_names: AxisNames | None = None,
        linear_out_axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any=None,
        dot_general: Any=None,
    ) -> None:
        self.activation = _resolve_activation(activation)

        self.linear_in = nn.Linear(hidden_size, intermediate_size * 2, bias=bias, dtype=dtype, seed=seed, axis_names=linear_in_axis_names, shard_mode=shard_mode, quant=quant, dot_general=dot_general)
        self.linear_out = nn.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype, seed=seed, axis_names=linear_out_axis_names, shard_mode=shard_mode, quant=quant, dot_general=dot_general)
    def __call__(self, x: jax.Array, out_sharding: Any=None) -> jax.Array:
        h = self.linear_in(x)
        h, gate = jnp.split(h, 2, axis=-1)
        return self.linear_out(h * self.activation(gate), out_sharding=out_sharding)

    @classmethod
    def apply_gmm(
        cls,
        lhs: jax.Array,
        rhs: jax.Array,
        group_sizes: jax.Array,
        transpose_rhs: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply Megablox Grouped Matrix Multiply (GMM) kernel."""
        from taktiny.kernels.megablox import gmm
        return gmm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, **kwargs)

    @classmethod
    def apply_route(
        cls,
        x: jax.Array,
        indices: jax.Array,
        num_groups: int | None = None,
        use_gather_mosaic_kernel: bool = False,
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply MoE activation sorting/routing kernel."""
        from taktiny.kernels.sort_activations import route
        indices_2d = indices[:, None] if indices.ndim == 1 else indices
        sorted_x = route(x, indices_2d, use_gather_mosaic_kernel)
        if num_groups is not None:
            group_sizes = jnp.bincount(indices.reshape(-1), length=num_groups)
        else:
            group_sizes = jnp.bincount(indices.reshape(-1))
        return sorted_x, group_sizes

    @classmethod
    def apply_unroute(
        cls,
        x: jax.Array,
        indices: jax.Array,
        use_gather_mosaic_kernel: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply MoE activation un-sorting/un-routing kernel."""
        from taktiny.kernels.sort_activations import unroute
        indices_2d = indices[:, None] if indices.ndim == 1 else indices
        return unroute(x, indices_2d, use_gather_mosaic_kernel)

    @classmethod
    def apply(
        cls,
        lhs: jax.Array,
        rhs: jax.Array,
        group_sizes: jax.Array,
        kernel: str = "gmm",
        **kwargs: Any,
    ) -> jax.Array:
        """Unified entry point for MoE GMM kernel application."""
        if not isinstance(kernel, str):
            raise TypeError(
                f'kernel must be a string, got {type(kernel).__name__}'
            )
        kernel = kernel.lower()
        if kernel == "gmm":
            return cls.apply_gmm(lhs, rhs, group_sizes, **kwargs)
        else:
            raise ValueError(f"Unknown MoE kernel method: '{kernel}'")

class MoeFFN(nn.Module):
    """
    Mixture-of-Experts (MoE) FFN module utilizing Megablox Grouped Matrix Multiply (GMM)
    and activation sorting kernels.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        bias: bool = False,
        dtype: str | None = None,
        rngs: nn.Rngs | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

    @classmethod
    def apply_gmm(
        cls,
        lhs: jax.Array,
        rhs: jax.Array,
        group_sizes: jax.Array,
        transpose_rhs: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply Megablox Grouped Matrix Multiply (GMM) kernel."""
        from taktiny.kernels.megablox import gmm
        return gmm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, **kwargs)

    @classmethod
    def apply_route(
        cls,
        x: jax.Array,
        indices: jax.Array,
        num_groups: int | None = None,
        use_gather_mosaic_kernel: bool = False,
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply MoE activation sorting/routing kernel."""
        from taktiny.kernels.sort_activations import route
        indices_2d = indices[:, None] if indices.ndim == 1 else indices
        sorted_x = route(x, indices_2d, use_gather_mosaic_kernel)
        if num_groups is not None:
            group_sizes = jnp.bincount(indices.reshape(-1), length=num_groups)
        else:
            group_sizes = jnp.bincount(indices.reshape(-1))
        return sorted_x, group_sizes

    @classmethod
    def apply_unroute(
        cls,
        x: jax.Array,
        indices: jax.Array,
        use_gather_mosaic_kernel: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply MoE activation un-sorting/un-routing kernel."""
        from taktiny.kernels.sort_activations import unroute
        indices_2d = indices[:, None] if indices.ndim == 1 else indices
        return unroute(x, indices_2d, use_gather_mosaic_kernel)

    @classmethod
    def apply(
        cls,
        lhs: jax.Array,
        rhs: jax.Array,
        group_sizes: jax.Array,
        kernel: str = "gmm",
        **kwargs: Any,
    ) -> jax.Array:
        """Unified entry point for MoE GMM kernel application."""
        if not isinstance(kernel, str):
            raise TypeError(
                f'kernel must be a string, got {type(kernel).__name__}'
            )
        kernel = kernel.lower()
        if kernel == "gmm":
            return cls.apply_gmm(lhs, rhs, group_sizes, **kwargs)
        else:
            raise ValueError(f"Unknown MoE kernel method: '{kernel}'")

class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Activation = jax.nn.gelu,
        bias: bool = True,
        dtype: str | None = None,
        rngs: nn.Rngs | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.activation = _resolve_activation(activation)
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype, rngs=rngs, shard_mode=shard_mode)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype, rngs=rngs, shard_mode=shard_mode)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(self.activation(self.fc1(x)))
