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
import typing as tp
import jax, jax.numpy as jnp

from taktiny.utils.typing import AxisNames, ShardMode
from taktiny import nn
from taktiny.nn.continuo import _constrain, _resolve_activation
from taktiny.utils.typing import Activation, DType


class FeedForward(nn.Module):
    """Apply a two-projection transformer feed-forward network.

    Dropout receives ``rngs`` during construction, so callers do not pass PRNG
    keys through the forward path.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Activation = jax.nn.gelu,
        dropout: float = 0.0,
        dtype: DType | None = None,
        bias: bool = True,
        rngs: nn.Rngs,
        input_axis_names: AxisNames | None = None,
        output_axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
    ) -> None:
        self.activation = _resolve_activation(activation)
        self.activation_name = getattr(
            self.activation,
            '__name__',
            type(self.activation).__name__,
        )
        self.input = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=input_axis_names,
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.output = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=output_axis_names,
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.dropout = nn.Dropout(
            dropout,
            rngs=rngs,
            shard_mode=shard_mode,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = self.activation(self.input(x))
        x = self.dropout(x)
        return self.output(x, out_sharding=out_sharding)

    def extra_repr(self) -> str:
        return f'activation={self.activation_name}, dropout={self.dropout.p:g}'


class GLUMBConv(nn.Module):
    """Sana-style gated depthwise-convolution feed-forward network.

    Inputs may be image grids in ``[batch, height, width, channels]`` layout
    or flattened image tokens in ``[batch, height * width, channels]`` layout.
    Token inputs require their spatial ``height`` and ``width``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        bias: bool = True,
        norm_type: str | None = None,
        residual_connection: bool = False,
        norm_eps: float = 1e-5,
        norm_bias: bool = False,
        dtype: DType | None = None,
        rngs: nn.Rngs,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError('hidden_size must be a positive integer')
        if not isinstance(intermediate_size, int) or intermediate_size <= 0:
            raise ValueError('intermediate_size must be a positive integer')
        if norm_type not in {None, 'rms_norm', 'rmsnorm'}:
            raise ValueError(
                "norm_type must be None, 'rms_norm', or 'rmsnorm'"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.norm_type = norm_type
        self.residual_connection = bool(residual_connection)
        self.shard_mode = shard_mode

        expanded_size = intermediate_size * 2
        self.conv_inverted = nn.Conv(
            hidden_size,
            expanded_size,
            (1, 1),
            dtype=dtype,
            bias=bias,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.conv_depth = nn.Conv(
            expanded_size,
            expanded_size,
            (3, 3),
            padding=1,
            groups=expanded_size,
            dtype=dtype,
            bias=bias,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.conv_point = nn.Conv(
            intermediate_size,
            hidden_size,
            (1, 1),
            dtype=dtype,
            bias=False,
            rngs=rngs,
            shard_mode=shard_mode,
        )
        self.norm = (
            nn.RMSNorm(
                hidden_size,
                epsilon=norm_eps,
                dtype=jnp.float32,
                bias=norm_bias,
                axis_names=('embed',),
                shard_mode=shard_mode,
            )
            if norm_type is not None
            else None
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        height: int | None = None,
        width: int | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        x = jnp.asarray(x)
        if x.ndim not in {3, 4}:
            raise ValueError(
                'GLUMBConv expects [batch, tokens, channels] or '
                '[batch, height, width, channels] input'
            )
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f'expected {self.hidden_size} channels, got {x.shape[-1]}'
            )

        tokens = x.ndim == 3
        if tokens:
            if not isinstance(height, int) or not isinstance(width, int):
                raise ValueError(
                    'height and width are required for token-sequence input'
                )
            if height <= 0 or width <= 0 or height * width != x.shape[1]:
                raise ValueError(
                    'height * width must equal the token sequence length'
                )
            x = x.reshape(x.shape[0], height, width, x.shape[-1])

        residual = x
        x = jax.nn.silu(self.conv_inverted(x))
        x = self.conv_depth(x)
        x, gate = jnp.split(x, 2, axis=-1)
        x = x * jax.nn.silu(gate)
        x = self.conv_point(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.residual_connection:
            x = x + residual
        if tokens:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        return _constrain(x, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'{self.hidden_size} -> {self.intermediate_size} -> '
            f'{self.hidden_size}'
        )


class GateMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
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
        *,
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
        from taktiny.cosettes.kernels.megablox import gmm
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
        from taktiny.cosettes.kernels.sort_activations import route
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
        from taktiny.cosettes.kernels.sort_activations import unroute
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


class MoERouter(nn.Module):
    """Route tokens to their highest-scoring experts.

    The router projects each token to ``num_experts`` logits, computes the
    routing probabilities in float32, and selects ``top_k`` experts. Returned
    scores optionally sum to one across the selected experts.

    Inputs may have any leading dimensions; they are flattened into tokens in
    the returned arrays. The return shapes are ``[tokens, num_experts]`` for
    logits and ``[tokens, top_k]`` for both scores and indices.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        top_k: int,
        num_experts: int,
        norm_topk: bool = True,
        rngs: nn.Rngs,
        dtype: DType | None = None,
        dot_general: Any = None,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError('hidden_size must be greater than zero')
        if num_experts <= 0:
            raise ValueError('num_experts must be greater than zero')
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                'top_k must be between 1 and num_experts, got '
                f'{top_k} and {num_experts}'
            )

        self.top_k = top_k
        self.num_experts = num_experts
        self.norm_topk = norm_topk
        self.hidden_size = hidden_size
        self.proj = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
            dtype=dtype,
            rngs=rngs,
            dot_general=dot_general,
            axis_names=axis_names,
            shard_mode=shard_mode,
        )

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if x.ndim == 0 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                'x must have a trailing hidden dimension of size '
                f'{self.hidden_size}, got shape {x.shape}'
            )

        hidden_states = x.reshape(-1, self.hidden_size)
        router_logits = self.proj(hidden_states, out_sharding=out_sharding)
        router_probs = jax.nn.softmax(
            router_logits.astype(jnp.float32),
            axis=-1,
        )
        router_scores, router_indices = jax.lax.top_k(
            router_probs,
            self.top_k,
        )
        if self.norm_topk:
            router_scores = router_scores / jnp.sum(
                router_scores,
                axis=-1,
                keepdims=True,
            )

        return (
            router_logits,
            router_scores.astype(router_logits.dtype),
            router_indices,
        )


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts (MoE) FFN module utilizing Megablox Grouped Matrix Multiply (GMM)
    and activation sorting kernels.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        num_experts: int,
        num_experts_per_tok: int = 1,
        activation: tp.Callable[[jax.Array], jax.Array] | str = jax.nn.silu,
        bias: bool = False,
        dtype: DType | None = None,
        rngs: nn.Rngs | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: Any = None,
        dot_general: Any = None,
        router: nn.Module | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok if num_experts_per_tok is not None else 1
        self.activation = _resolve_activation(activation)
        self.quant = quant
        self.dot_general = dot_general
        self.shard_mode = shard_mode
        
        if router is None:
            self.gate = nn.Linear(
                hidden_size,
                num_experts,
                bias=False,
                dtype=jnp.float32,
                rngs=rngs,
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
            )
        else:
            self.router = router
        
        # Expert weights for GMM: [num_experts, in_features, out_features]
        rngs = rngs if rngs is not None else nn.Rngs(0)
        self.w1 = nn.Parameter(jax.random.normal(rngs(), (num_experts, hidden_size, intermediate_size), dtype=dtype))
        self.w1.quantization = quant
        self.w1.input_axis_count = 1
        self.w1.quantization_batch_axis_count = 1
        self.w1.axis_names = ('experts', 'embed', 'mlp')

        self.w3 = nn.Parameter(jax.random.normal(rngs(), (num_experts, hidden_size, intermediate_size), dtype=dtype))
        self.w3.quantization = quant
        self.w3.input_axis_count = 1
        self.w3.quantization_batch_axis_count = 1
        self.w3.axis_names = ('experts', 'embed', 'mlp')

        self.w2 = nn.Parameter(jax.random.normal(rngs(), (num_experts, intermediate_size, hidden_size), dtype=dtype))
        self.w2.quantization = quant
        self.w2.input_axis_count = 1
        self.w2.quantization_batch_axis_count = 1
        self.w2.axis_names = ('experts', 'mlp', 'embed')

    def route(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        if hasattr(self, 'router'):
            routed = self.router(x)
            if not isinstance(routed, tuple) or len(routed) != 3:
                raise TypeError(
                    'router must return probabilities, weights, and indices'
                )
            _, routing_weights, selected_experts = routed
            return routing_weights, selected_experts

        router_logits = self.gate(x)
        routing_weights = jax.nn.softmax(router_logits, axis=-1)
        routing_weights, selected_experts = jax.lax.top_k(
            routing_weights,
            self.num_experts_per_tok,
        )
        routing_weights /= jnp.sum(
            routing_weights,
            axis=-1,
            keepdims=True,
        )
        return routing_weights, selected_experts

    def __call__(self, x: jax.Array, out_sharding: Any = None) -> jax.Array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        
        # 1. Routing
        routing_weights, selected_experts = self.route(x_flat)
        
        # 2. Sort activations
        sorted_x, group_sizes = self.apply_route(x_flat, selected_experts, num_groups=self.num_experts)
        
        # 3. Apply experts using GMM
        h1 = self.apply_gmm(sorted_x, self.w1.value, group_sizes)
        h3 = self.apply_gmm(sorted_x, self.w3.value, group_sizes)
        h = self.activation(h1) * h3
        
        expert_out = self.apply_gmm(h, self.w2.value, group_sizes)
        
        # 4. Multiply by routing weights
        flat_indices = jnp.ravel(selected_experts)
        sort_inds = jnp.argsort(flat_indices)
        sorted_weights = jnp.ravel(routing_weights)[sort_inds].astype(expert_out.dtype)
        expert_out = expert_out * sorted_weights[:, None]
        
        # 5. Unroute
        out_flat = self.apply_unroute(expert_out, selected_experts)
        out = out_flat.reshape(orig_shape).astype(x.dtype)
        if out_sharding is not None:
            out = jax.lax.with_sharding_constraint(out, out_sharding)
        return out

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
        from taktiny.cosettes.kernels.megablox import gmm
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
        from taktiny.cosettes.kernels.sort_activations import route
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
        from taktiny.cosettes.kernels.sort_activations import unroute
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
        *,
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
