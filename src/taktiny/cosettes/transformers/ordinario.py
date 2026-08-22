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
# WITHOUT WARRANTIES OR CONDITIONS OF tp.Any KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common base modules for transformer architectures"""

from __future__ import annotations
from collections.abc import Callable, Iterator, Mapping, Sequence
import typing as tp
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import qwix
from dataclasses import replace
from functools import partial

from taktiny import nn
from taktiny.cosettes.continuo import _approximate_gelu
from taktiny.cosettes.overture import ModelOutput, PretrainedModel
from taktiny.maestro.config import ModelConfig
from taktiny.utils.typing import (
    ArrayLike,
    DType,
    LogicalRules,
    PathLike,
    ShardMode,
)
from taktiny.utils.sharding import create_sharding
from taktiny.cosettes.layers import (
    AdaXNorm,
    AttentionLegacy,
    ConditionalTransformerLayer as _ConditionalTransformerLayer,
    FeedForward,
    GLUMBConv,
    GateMLP,
    JointAttention,
    JointTransformerLayer as _JointTransformerLayer,
    GatedParallelTransformerLayer as _GatedParallelTransformerLayer,
    RotaryEmbedding,
    MoEFFN,
    Attention,
    _RotaryEmbedding
)
from taktiny.cosettes.continuo import (
    _activation,
    _config_value,
    _hidden_size,
    _model_dtype,
    _positive_int,
    _shard_mode,
)
from taktiny.nn.continuo import _constrain
from taktiny.maestro.config import _validate_dtype_config, _verify_required_config_attributes

KVCache = tuple[jax.Array, jax.Array]
DecodeCarry = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
GenerationSettings = tuple[int, jax.Array, int, str]
PositionEmbedding = tuple[jax.Array, jax.Array]
PositionEmbeddings = PositionEmbedding | tp.Mapping[str, PositionEmbedding]

@partial(
    jax.tree_util.register_dataclass,
    data_fields=['key_cache', 'value_cache', 'position_idx'],
    meta_fields=['is_causal', 'attention_kernel'],
)
@dataclass(frozen=True)
class TransformerContext:
    key_cache: tp.Optional[jax.Array] = None
    value_cache: tp.Optional[jax.Array] = None
    position_idx: tp.Optional[jax.Array] = None
    is_causal: tp.Optional[bool] = None
    attention_kernel: str = 'dot_product'


class ConditionalTransformerLayer(_ConditionalTransformerLayer):
    """Config-driven single-stream conditional transformer layer.

    The hidden stream is updated by modulated self-attention, read-only
    cross-attention context, and a modulated spatial feed-forward branch.
    Architectures may replace each compatible component while preserving this
    topology.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
        activation: str | Callable[[jax.Array], jax.Array] | None = None,
        ffn_dropout: float | None = None,
        attention_bias: bool | None = None,
        attention_out_bias: bool | None = None,
        cross_attention_bias: bool | None = None,
        mlp_bias: bool | None = None,
        pos_emb: nn.Module | None = None,
        cross_pos_emb: nn.Module | None = None,
        input_layernorm: nn.Module | type[nn.Module] = nn.LayerNorm,
        self_attention: nn.Module | type[nn.Module] = AttentionLegacy,
        cross_attention: nn.Module | type[nn.Module] | None = AttentionLegacy,
        cross_attention_layernorm: (
            nn.Module | type[nn.Module] | None
        ) = None,
        post_attention_layernorm: (
            nn.Module | type[nn.Module]
        ) = nn.LayerNorm,
        mlp: nn.Module | type[nn.Module] = GLUMBConv,
    ) -> None:
        hidden_size = _hidden_size(config)
        num_heads = _config_value(config, 'num_attention_heads')
        head_dim = _config_value(
            config,
            'head_dim',
            'attention_head_dim',
        )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError('config must define num_attention_heads')

        context_size = _config_value(
            config,
            'cross_attention_dim',
            'context_dim',
            default=hidden_size,
        )
        cross_num_heads = _config_value(
            config,
            'num_cross_attention_heads',
            default=num_heads,
        )
        cross_head_dim = _config_value(
            config,
            'cross_attention_head_dim',
            default=head_dim,
        )
        intermediate_size = _config_value(config, 'intermediate_size')
        if intermediate_size is None:
            ratio = _config_value(
                config,
                'mlp_ratio',
                'expand_ratio',
                default=4.0,
            )
            intermediate_size = int(hidden_size * ratio)

        qk_norm = _config_value(config, 'qk_norm')
        dropout = _config_value(config, 'dropout', default=0.0)
        if ffn_dropout is None:
            ffn_dropout = _config_value(
                config,
                'ffn_dropout',
                default=0.0,
            )
        if attention_bias is None:
            attention_bias = bool(
                _config_value(config, 'attention_bias', default=False)
            )
        if attention_out_bias is None:
            attention_out_bias = bool(
                _config_value(config, 'attention_out_bias', default=True)
            )
        if cross_attention_bias is None:
            cross_attention_bias = bool(
                _config_value(config, 'cross_attention_bias', default=True)
            )
        if mlp_bias is None:
            mlp_bias = bool(
                _config_value(
                    config,
                    'glumbconv_bias',
                    'mlp_bias',
                    default=True,
                )
            )
        super().__init__(
            hidden_size=hidden_size,
            context_size=context_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            cross_num_heads=cross_num_heads,
            cross_head_dim=cross_head_dim,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            activation=(
                _activation(config, default='gelu')
                if activation is None
                else activation
            ),
            norm_eps=_config_value(config, 'norm_eps', default=1e-6),
            norm_elementwise_affine=bool(
                _config_value(
                    config,
                    'norm_elementwise_affine',
                    default=False,
                )
            ),
            bias=attention_bias,
            attention_out_bias=attention_out_bias,
            cross_attention_bias=cross_attention_bias,
            mlp_bias=mlp_bias,
            use_qkv_norm=bool(
                _config_value(
                    config,
                    'use_qkv_norm',
                    default=qk_norm is not None,
                )
            ),
            qkv_norm_across_heads=(qk_norm == 'rms_norm_across_heads'),
            qkv_norm_eps=_config_value(
                config,
                'qkv_norm_eps',
                'norm_eps',
                default=1e-5,
            ),
            pos_emb=pos_emb,
            cross_pos_emb=cross_pos_emb,
            dtype=_model_dtype(config),
            rngs=rngs,
            shard_mode=_shard_mode(config),
            quant=_config_value(config, 'quant'),
            dot_general=_config_value(config, 'dot_general'),
            input_layernorm=input_layernorm,
            self_attention=self_attention,
            cross_attention=cross_attention,
            cross_attention_layernorm=cross_attention_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            mlp=mlp,
        )
        self.layer_idx = layer_idx


class JointTransformerLayer(_JointTransformerLayer):
    """A config-driven, composable two-stream transformer layer.

    This is the joint-attention counterpart to
    :class:`TransformerDecoderLayer`. It translates architecture config values
    into the general two-stream implementation in :mod:`taktiny.layers`, while
    allowing architectures to replace each compatible component with a module
    subclass or initialized instance.

    ``layer_idx`` selects layer-dependent topology. The final layer defaults to
    context-pre-only behavior, and indices listed in
    ``config.dual_attention_layers`` receive a second hidden-stream attention.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
        conditioning_size: int | None = None,
        context_pre_only: bool | None = None,
        dual_attention: bool | None = None,
        project_conditioning: bool = True,
        context_project_conditioning: bool | None = None,
        use_qkv_norm: bool | None = None,
        qkv_norm_eps: float | None = None,
        context_first: bool = False,
        bias: bool | None = None,
        pos_emb: nn.Module | None = None,
        activation: str | Callable[[jax.Array], jax.Array] | None = None,
        input_layernorm: nn.Module | type[nn.Module] = AdaXNorm,
        context_input_layernorm: nn.Module | type[nn.Module] = AdaXNorm,
        joint_attention: nn.Module | type[nn.Module] = JointAttention,
        second_attention: nn.Module | type[nn.Module] = AttentionLegacy,
        post_attention_layernorm: nn.Module | type[nn.Module] | None = None,
        context_post_attention_layernorm: (
            nn.Module | type[nn.Module] | None
        ) = None,
        mlp: nn.Module | type[nn.Module] = FeedForward,
        context_mlp: nn.Module | type[nn.Module] = FeedForward,
    ) -> None:
        num_heads = _config_value(config, 'num_attention_heads')
        head_dim = _config_value(config, 'head_dim', 'attention_head_dim')
        hidden_size = _config_value(config, 'hidden_size', 'inner_dim', 'dim')
        if hidden_size is None and num_heads is not None and head_dim is not None:
            hidden_size = num_heads * head_dim

        required = {
            'hidden size': hidden_size,
            'num_attention_heads': num_heads,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                'Missing required joint transformer config values: '
                + ', '.join(missing)
            )

        context_size = _config_value(
            config,
            'context_size',
            'context_dim',
            'caption_projection_dim',
            default=hidden_size,
        )
        intermediate_size = _config_value(config, 'intermediate_size')
        if intermediate_size is None:
            mlp_ratio = _config_value(config, 'mlp_ratio', default=4.0)
            intermediate_size = int(hidden_size * mlp_ratio)
        context_intermediate_size = _config_value(
            config,
            'context_intermediate_size',
            default=intermediate_size,
        )
        if conditioning_size is None:
            conditioning_size = _config_value(
                config,
                'conditioning_size',
                'time_embed_dim',
                default=hidden_size,
            )

        num_layers = _config_value(
            config,
            'num_hidden_layers',
            'num_layers',
        )
        if context_pre_only is None:
            configured_context_pre_only = _config_value(
                config,
                'context_pre_only',
            )
            context_pre_only = (
                bool(configured_context_pre_only)
                if configured_context_pre_only is not None
                else (
                    layer_idx is not None
                    and num_layers is not None
                    and layer_idx == num_layers - 1
                )
            )
        if dual_attention is None:
            dual_layers = _config_value(
                config,
                'dual_attention_layers',
                default=(),
            )
            dual_attention = layer_idx is not None and layer_idx in dual_layers

        qk_norm = _config_value(config, 'qk_norm')
        if use_qkv_norm is None:
            use_qkv_norm = bool(
                _config_value(
                    config,
                    'use_qkv_norm',
                    default=qk_norm is not None,
                )
            )
        if qkv_norm_eps is None:
            qkv_norm_eps = _config_value(
                config,
                'qkv_norm_eps',
                'norm_eps',
                default=1e-6,
            )
        if pos_emb is None:
            pos_emb = _config_value(
                config,
                'pos_emb',
                'position_embedding',
            )
        norm_type = _config_value(config, 'norm_type', default='layernorm')
        norm_type = str(norm_type).lower().replace('_', '')
        if norm_type in {'layer', 'layernormalization'}:
            norm_type = 'layernorm'
        elif norm_type in {'rms', 'rmsnormalization'}:
            norm_type = 'rmsnorm'

        super().__init__(
            hidden_size=hidden_size,
            context_size=context_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            context_intermediate_size=context_intermediate_size,
            conditioning_size=conditioning_size,
            head_dim=head_dim,
            dropout=_config_value(
                config,
                'dropout',
                'attention_dropout',
                default=0.0,
            ),
            activation=(
                activation
                if activation is not None
                else _config_value(
                    config,
                    'hidden_act',
                    'hidden_activation',
                    'activation',
                    default='gelu',
                )
            ),
            norm=norm_type,
            norm_eps=_config_value(
                config,
                'norm_eps',
                'layer_norm_eps',
                'rms_norm_eps',
                default=1e-6,
            ),
            context_pre_only=context_pre_only,
            dual_attention=dual_attention,
            bias=(
                bool(
                    _config_value(
                        config,
                        'projection_bias',
                        'attention_bias',
                        default=True,
                    )
                )
                if bias is None
                else bias
            ),
            use_qkv_norm=use_qkv_norm,
            qkv_norm_eps=qkv_norm_eps,
            context_first=context_first,
            scaling=_config_value(config, 'attention_scaling'),
            pos_emb=pos_emb,
            second_pos_emb=_config_value(config, 'second_pos_emb'),
            dtype=_model_dtype(config),
            rngs=rngs,
            shard_mode=_shard_mode(config),
            quant=_config_value(config, 'quant'),
            dot_general=_config_value(config, 'dot_general'),
            project_conditioning=project_conditioning,
            context_project_conditioning=context_project_conditioning,
            input_layernorm=input_layernorm,
            context_input_layernorm=context_input_layernorm,
            joint_attention=joint_attention,
            second_attention=second_attention,
            post_attention_layernorm=post_attention_layernorm,
            context_post_attention_layernorm=(
                context_post_attention_layernorm
            ),
            mlp=mlp,
            context_mlp=context_mlp,
        )
        self.layer_idx = layer_idx


class GatedParallelTransformerLayer(_GatedParallelTransformerLayer):
    """A config-driven transformer layer with parallel attention and FFN."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        parallel_path: nn.Module | type[nn.Module],
        layer_idx: int | None = None,
        conditioning_size: int | None = None,
        project_conditioning: bool = True,
        pos_emb: nn.Module | None = None,
        activation: str | Callable[[jax.Array], jax.Array] | None = None,
        use_qkv_norm: bool | None = None,
        bias: bool | None = None,
        input_layernorm: nn.Module | type[nn.Module] = AdaXNorm,
    ) -> None:
        num_heads = _config_value(config, 'num_attention_heads')
        head_dim = _config_value(config, 'head_dim', 'attention_head_dim')
        hidden_size = _config_value(config, 'hidden_size', 'inner_dim', 'dim')
        if hidden_size is None and num_heads is not None and head_dim is not None:
            hidden_size = num_heads * head_dim
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError('config must define a positive hidden size')
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError('config must define num_attention_heads')

        intermediate_size = _config_value(config, 'intermediate_size')
        if intermediate_size is None:
            intermediate_size = int(
                hidden_size
                * _config_value(config, 'mlp_ratio', default=4.0)
            )
        if conditioning_size is None:
            conditioning_size = _config_value(
                config,
                'conditioning_size',
                'time_embed_dim',
                default=hidden_size,
            )
        qk_norm = _config_value(config, 'qk_norm')
        if use_qkv_norm is None:
            use_qkv_norm = bool(
                _config_value(
                    config,
                    'use_qkv_norm',
                    default=qk_norm is not None,
                )
            )
        if pos_emb is None:
            pos_emb = _config_value(
                config,
                'pos_emb',
                'position_embedding',
            )
        norm_type = _config_value(config, 'norm_type', default='layernorm')
        norm_type = str(norm_type).lower().replace('_', '')
        if norm_type in {'layer', 'layernormalization'}:
            norm_type = 'layernorm'
        elif norm_type in {'rms', 'rmsnormalization'}:
            norm_type = 'rmsnorm'

        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            conditioning_size=conditioning_size,
            parallel_path=parallel_path,
            head_dim=head_dim,
            dropout=_config_value(
                config,
                'dropout',
                'attention_dropout',
                default=0.0,
            ),
            activation=(
                activation
                if activation is not None
                else _activation(config, default='gelu')
            ),
            norm=norm_type,
            norm_eps=_config_value(
                config,
                'eps',
                'norm_eps',
                'layer_norm_eps',
                default=1e-6,
            ),
            bias=(
                bool(
                    _config_value(
                        config,
                        'projection_bias',
                        'attention_bias',
                        default=False,
                    )
                )
                if bias is None
                else bias
            ),
            pos_emb=pos_emb,
            dtype=_model_dtype(config),
            rngs=rngs,
            shard_mode=_shard_mode(config),
            quant=_config_value(config, 'quant'),
            dot_general=_config_value(config, 'dot_general'),
            project_conditioning=project_conditioning,
            use_qkv_norm=use_qkv_norm,
            scaling=_config_value(config, 'attention_scaling'),
            input_layernorm=input_layernorm,
        )
        self.use_qkv_norm = use_qkv_norm
        self.layer_idx = layer_idx


DiffusionComponent: tp.TypeAlias = nn.Module | type[nn.Module]


class DiffusionTransformerModel(PretrainedModel):
    """Composable patch-based diffusion transformer backbone.

    The class owns the model-level mechanics shared by denoising transformers:
    component construction, repeated transformer layers, rematerialization,
    optional layer skipping, ControlNet residual routing, and patch
    reconstruction. Concrete Maestro architectures select compatible
    component types while Cosette layer classes define their mathematics.

    Component types follow role-specific constructor contracts. Initialized
    module instances may be supplied when an architecture requires a different
    constructor. Subclasses can override the preparation and finalization
    hooks without reimplementing layer iteration. ``use_list=False`` stores
    layers in an ``nn.SeqStack``. Depth-dependent topologies are partitioned
    into maximal contiguous stack-compatible groups while preserving one
    carry and the original execution order.
    """

    default_sharding_rules = (
        ('batch', 'fsdp'),
        ('height', None),
        ('width', None),
        ('sequence', None),
        ('embed', None),
        ('context_embed', None),
        ('heads', 'tp'),
        ('head_dim', None),
        ('mlp', 'tp'),
        ('channel', None),
    )
    _component_names = frozenset(
        {
            'patch_embedding',
            'condition_embedding',
            'context_embedding',
            'output_norm',
            'output_projection',
        }
    )

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        transformer_layer: type[nn.Module],
        patch_embedding: DiffusionComponent,
        condition_embedding: DiffusionComponent,
        context_embedding: DiffusionComponent | None,
        output_norm: DiffusionComponent | None,
        output_projection: DiffusionComponent,
        component_kwargs: Mapping[str, Mapping[str, tp.Any]] | None = None,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        use_list: bool = True,
        stack_type: tp.Literal['list', 'stack'] | None = None,
    ) -> None:
        if (
            not isinstance(transformer_layer, type)
            or not issubclass(transformer_layer, nn.Module)
        ):
            raise TypeError('transformer_layer must be an nn.Module subclass')
        if not isinstance(use_list, bool):
            raise TypeError('use_list must be a boolean')
        if stack_type is not None:
            if stack_type not in {'list', 'stack'}:
                raise ValueError("stack_type must be 'list' or 'stack'")
            use_list = stack_type == 'list'

        self.config = config
        self.num_layers = _positive_int(
            _config_value(config, 'num_layers', 'num_hidden_layers'),
            'num_layers',
        )
        self.num_attention_heads = _positive_int(
            _config_value(config, 'num_attention_heads'),
            'num_attention_heads',
        )
        self.attention_head_dim = _positive_int(
            _config_value(config, 'attention_head_dim', 'head_dim'),
            'attention_head_dim',
        )
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.in_channels = _positive_int(config.in_channels, 'in_channels')
        self.out_channels = _positive_int(
            _config_value(config, 'out_channels', default=self.in_channels),
            'out_channels',
        )
        self.patch_size = self._spatial_pair(
            _config_value(config, 'patch_size', default=2),
            'patch_size',
        )
        self.shard_mode = _shard_mode(config)
        self.dtype = _model_dtype(config)
        self.quant = _config_value(config, 'quant')
        self.dot_general = _config_value(config, 'dot_general')

        options = self._component_options(component_kwargs)
        sample_size = _config_value(config, 'sample_size', default=128)
        self.patch_embedding = self._instantiate_component(
            patch_embedding,
            name='patch_embedding',
            options={
                'sample_size': sample_size,
                'patch_size': self.patch_size,
                'in_channels': self.in_channels,
                'embedding_dim': self.inner_dim,
                'dtype': self.dtype,
                'rngs': rngs,
                'shard_mode': self.shard_mode,
                **options['patch_embedding'],
            },
        )

        if isinstance(condition_embedding, nn.Module):
            self.condition_embedding = condition_embedding
        else:
            pooled_projection_dim = _positive_int(
                config.pooled_projection_dim,
                'pooled_projection_dim',
            )
            self.condition_embedding = self._instantiate_component(
                condition_embedding,
                name='condition_embedding',
                options={
                    'embedding_dim': self.inner_dim,
                    'pooled_projection_dim': pooled_projection_dim,
                    'dtype': self.dtype,
                    'rngs': rngs,
                    'quant': self.quant,
                    'dot_general': self.dot_general,
                    'shard_mode': self.shard_mode,
                    **options['condition_embedding'],
                },
            )

        if context_embedding is None:
            self.context_embedding = None
        elif isinstance(context_embedding, nn.Module):
            self.context_embedding = context_embedding
        else:
            joint_attention_dim = _positive_int(
                config.joint_attention_dim,
                'joint_attention_dim',
            )
            caption_projection_dim = _positive_int(
                _config_value(
                    config,
                    'caption_projection_dim',
                    default=self.inner_dim,
                ),
                'caption_projection_dim',
            )
            self.context_embedding = self._instantiate_component(
                context_embedding,
                name='context_embedding',
                options={
                    'in_features': joint_attention_dim,
                    'out_features': caption_projection_dim,
                    'bias': True,
                    'dtype': self.dtype,
                    'rngs': rngs,
                    'quant': self.quant,
                    'dot_general': self.dot_general,
                    'axis_names': ('joint_embed', 'context_embed'),
                    'shard_mode': self.shard_mode,
                    **options['context_embedding'],
                },
            )

        layers = [
            transformer_layer(config, rngs=rngs, layer_idx=index)
            for index in range(self.num_layers)
        ]
        self.requested_use_list = use_list
        if use_list:
            self.layers = nn.List(layers)
        else:
            for layer in layers:
                if hasattr(layer, 'layer_idx'):
                    layer.layer_idx = None
            self.layers = nn.SeqStack(layers)
        self.use_list = isinstance(self.layers, nn.List)

        if output_norm is None:
            self.output_norm = None
        else:
            self.output_norm = self._instantiate_component(
                output_norm,
                name='output_norm',
                options={
                    'embedding_dim': self.inner_dim,
                    'out_dim': 2 * self.inner_dim,
                    'norm': 'layernorm',
                    'eps': 1e-6,
                    'activation': 'silu',
                    'bias': True,
                    'dtype': self.dtype,
                    'rngs': rngs,
                    'quant': self.quant,
                    'dot_general': self.dot_general,
                    'axis_names': ('conditioning', 'output_modulation'),
                    'shard_mode': self.shard_mode,
                    **options['output_norm'],
                },
            )

        self.output_projection = self._instantiate_component(
            output_projection,
            name='output_projection',
            options={
                'in_features': self.inner_dim,
                'out_features': (
                    self.patch_size[0]
                    * self.patch_size[1]
                    * self.out_channels
                ),
                'bias': True,
                'dtype': self.dtype,
                'rngs': rngs,
                'quant': self.quant,
                'dot_general': self.dot_general,
                'axis_names': ('embed', 'patch'),
                'shard_mode': self.shard_mode,
                **options['output_projection'],
            },
        )

        if sharding_rules is None:
            sharding_rules = self.default_sharding_rules
        self.output_sharding = None
        if mesh is not None and self.shard_mode == ShardMode.EXPLICIT:
            self.output_sharding = create_sharding(
                mesh,
                ('batch', 'height', 'width', 'channel'),
                rules=sharding_rules,
            )
        self.remat = False

    @classmethod
    def _instantiate_component(
        cls,
        component: DiffusionComponent,
        *,
        name: str,
        options: Mapping[str, tp.Any],
    ) -> nn.Module:
        if isinstance(component, nn.Module):
            return component
        if not isinstance(component, type) or not issubclass(component, nn.Module):
            raise TypeError(f'{name} must be an nn.Module subclass or instance')
        return component(**options)

    @classmethod
    def _component_options(
        cls,
        component_kwargs: Mapping[str, Mapping[str, tp.Any]] | None,
    ) -> dict[str, dict[str, tp.Any]]:
        supplied = dict(component_kwargs or {})
        unknown = supplied.keys() - cls._component_names
        if unknown:
            raise ValueError(
                'unknown diffusion component options: '
                + ', '.join(sorted(unknown))
            )
        result = {name: {} for name in cls._component_names}
        for name, values in supplied.items():
            if not isinstance(values, Mapping):
                raise TypeError(f'component_kwargs[{name!r}] must be a mapping')
            result[name] = dict(values)
        return result

    @staticmethod
    def _spatial_pair(
        value: int | Sequence[int],
        name: str,
    ) -> tuple[int, int]:
        values = (value, value) if isinstance(value, int) else tuple(value)
        if len(values) != 2:
            raise ValueError(f'{name} must contain exactly two dimensions')
        for index, size in enumerate(values):
            _positive_int(size, f'{name}[{index}]')
        return tp.cast(tuple[int, int], values)

    def enable_remat(self) -> None:
        """Rematerialize transformer blocks during differentiation."""
        self.remat = True

    def disable_remat(self) -> None:
        """Disable transformer-block rematerialization."""
        self.remat = False

    def _prepare_conditioning(
        self,
        timestep: jax.Array,
        pooled_projection: jax.Array,
    ) -> jax.Array:
        return self.condition_embedding(timestep, pooled_projection)

    def _prepare_context(self, encoder_hidden_states: jax.Array) -> jax.Array:
        if self.context_embedding is None:
            return encoder_hidden_states
        return self.context_embedding(encoder_hidden_states)

    def _call_transformer_layer(
        self,
        layer: nn.Module,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        conditioning: jax.Array,
        **layer_kwargs: tp.Any,
    ) -> tuple[jax.Array, jax.Array]:
        result = layer(
            hidden_states,
            encoder_hidden_states,
            conditioning,
            **layer_kwargs,
        )
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError(
                    'a diffusion transformer layer tuple must contain '
                    '(context, hidden_states)'
                )
            next_context, next_hidden = result
            if next_context is None:
                next_context = encoder_hidden_states
            return next_context, next_hidden
        return encoder_hidden_states, result

    @staticmethod
    def _validate_skip_layers(
        skip_layers: Sequence[int] | None,
        num_layers: int,
    ) -> frozenset[int]:
        if skip_layers is None:
            return frozenset()
        result = frozenset(skip_layers)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= num_layers
            for index in result
        ):
            raise ValueError('skip_layers contains an invalid layer index')
        return result

    def _apply_transformer_layers(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        conditioning: jax.Array,
        *,
        control_residuals: Sequence[jax.Array] | None,
        skip_layers: Sequence[int] | None,
        layer_kwargs: Mapping[str, tp.Any],
    ) -> tuple[jax.Array, jax.Array]:
        skipped = self._validate_skip_layers(skip_layers, self.num_layers)
        controls = () if control_residuals is None else tuple(control_residuals)
        if control_residuals is not None and not controls:
            raise ValueError('control_residuals must not be empty')

        call_layer = self._call_transformer_layer
        if self.remat:
            call_layer = jax.checkpoint(
                call_layer,
                prevent_cse=self.use_list,
            )

        if not self.use_list:
            control_stack = None
            if controls:
                for control in controls:
                    if control.shape != hidden_states.shape:
                        raise ValueError(
                            'each control residual must match the hidden token '
                            f'shape {hidden_states.shape}; got {control.shape}'
                        )
                control_stack = jnp.stack(
                    tuple(jnp.asarray(control) for control in controls)
                )

            skipped_indices = None
            if skipped:
                skipped_indices = jnp.asarray(
                    tuple(sorted(skipped)),
                    dtype=jnp.int32,
                )

            def apply_layer(
                layer: nn.Module,
                carry: tuple[jax.Array, jax.Array, jax.Array],
            ) -> tuple[
                tuple[jax.Array, jax.Array, jax.Array],
                None,
            ]:
                context, hidden, layer_index = carry

                def apply(
                    operands: tuple[jax.Array, jax.Array],
                ) -> tuple[jax.Array, jax.Array]:
                    current_context, current_hidden = operands
                    return call_layer(
                        layer,
                        current_hidden,
                        current_context,
                        conditioning,
                        **layer_kwargs,
                    )

                if skipped_indices is None:
                    context, hidden = apply((context, hidden))
                else:
                    should_skip = jnp.any(layer_index == skipped_indices)
                    context, hidden = jax.lax.cond(
                        should_skip,
                        lambda operands: operands,
                        apply,
                        (context, hidden),
                    )

                if (
                    control_stack is not None
                    and not getattr(layer, 'context_pre_only', False)
                ):
                    control_index = jnp.minimum(
                        layer_index * len(controls) // self.num_layers,
                        len(controls) - 1,
                    )
                    hidden = hidden + jax.lax.dynamic_index_in_dim(
                        control_stack,
                        control_index,
                        axis=0,
                        keepdims=False,
                    )

                return (context, hidden, layer_index + 1), None

            (encoder_hidden_states, hidden_states, _), _ = self.layers(
                apply_layer,
                (
                    encoder_hidden_states,
                    hidden_states,
                    jnp.asarray(0, dtype=jnp.int32),
                ),
            )
            return encoder_hidden_states, hidden_states

        for index, layer in enumerate(self.layers):
            if index not in skipped:
                encoder_hidden_states, hidden_states = call_layer(
                    layer,
                    hidden_states,
                    encoder_hidden_states,
                    conditioning,
                    **layer_kwargs,
                )

            if controls and not getattr(layer, 'context_pre_only', False):
                control_index = min(
                    int(index * len(controls) / self.num_layers),
                    len(controls) - 1,
                )
                control = jnp.asarray(controls[control_index])
                if control.shape != hidden_states.shape:
                    raise ValueError(
                        'each control residual must match the hidden token '
                        f'shape {hidden_states.shape}; got {control.shape}'
                    )
                hidden_states = hidden_states + control
        return encoder_hidden_states, hidden_states

    def _finalize_tokens(
        self,
        hidden_states: jax.Array,
        conditioning: jax.Array,
    ) -> jax.Array:
        if self.output_norm is not None:
            normalized = self.output_norm(hidden_states, conditioning)
            if isinstance(normalized, tuple):
                if len(normalized) != 2:
                    raise ValueError(
                        'output_norm tuple must contain normalized activations '
                        'and modulation'
                    )
                hidden_states, modulation = normalized
                if modulation.shape[-1] != 2 * hidden_states.shape[-1]:
                    raise ValueError(
                        'output modulation must contain one scale and shift '
                        'value per hidden feature'
                    )
                scale, shift = jnp.split(modulation, 2, axis=-1)
                hidden_states = hidden_states * (1.0 + scale[:, None, :])
                hidden_states = hidden_states + shift[:, None, :]
            else:
                hidden_states = normalized
        return self.output_projection(hidden_states)

    @staticmethod
    def _unpatchify(
        tokens: jax.Array,
        grid_size: tuple[int, int],
        patch_size: tuple[int, int],
        out_channels: int,
    ) -> jax.Array:
        batch = tokens.shape[0]
        grid_height, grid_width = grid_size
        patch_height, patch_width = patch_size
        tokens = tokens.reshape(
            batch,
            grid_height,
            grid_width,
            patch_height,
            patch_width,
            out_channels,
        )
        tokens = jnp.transpose(tokens, (0, 1, 3, 2, 4, 5))
        return tokens.reshape(
            batch,
            grid_height * patch_height,
            grid_width * patch_width,
            out_channels,
        )

    def __call__(
        self,
        x: jax.Array,
        encoder_hidden_states: jax.Array,
        pooled_projection: jax.Array,
        timestep: jax.Array,
        *,
        control_residuals: Sequence[jax.Array] | None = None,
        layer_kwargs: Mapping[str, tp.Any] | None = None,
        skip_layers: Sequence[int] | None = None,
        return_dict: bool = True,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array | tuple[jax.Array]:
        x = jnp.asarray(x)
        encoder_hidden_states = jnp.asarray(encoder_hidden_states)
        pooled_projection = jnp.asarray(pooled_projection)
        timestep = jnp.asarray(timestep)
        if x.ndim != 4 or x.shape[-1] != self.in_channels:
            raise ValueError(
                'x must have shape [batch, height, width, in_channels]'
            )
        if not jnp.issubdtype(x.dtype, jnp.floating):
            raise TypeError('x must have a floating-point dtype')
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                'encoder_hidden_states must have shape '
                '[batch, sequence, hidden]'
            )
        if pooled_projection.ndim != 2:
            raise ValueError(
                'pooled_projection must have shape [batch, hidden]'
            )
        batch = x.shape[0]
        if (
            encoder_hidden_states.shape[0] != batch
            or pooled_projection.shape[0] != batch
        ):
            raise ValueError('all inputs must share the same batch size')
        if timestep.ndim == 0:
            timestep = jnp.broadcast_to(timestep, (batch,))
        if timestep.shape != (batch,):
            raise ValueError('timestep must be a scalar or have shape [batch]')

        height, width = x.shape[1:3]
        patch_height, patch_width = self.patch_size
        if height % patch_height or width % patch_width:
            raise ValueError('latent dimensions must be divisible by patch_size')
        grid_size = (height // patch_height, width // patch_width)

        hidden_states = self.patch_embedding(x)
        conditioning = self._prepare_conditioning(
            timestep,
            pooled_projection,
        )
        encoder_hidden_states = self._prepare_context(
            encoder_hidden_states,
        )
        _, hidden_states = self._apply_transformer_layers(
            hidden_states,
            encoder_hidden_states,
            conditioning,
            control_residuals=control_residuals,
            skip_layers=skip_layers,
            layer_kwargs=dict(layer_kwargs or {}),
        )
        output = self._unpatchify(
            self._finalize_tokens(hidden_states, conditioning),
            grid_size,
            self.patch_size,
            self.out_channels,
        )
        target_sharding = (
            self.output_sharding if out_sharding is None else out_sharding
        )
        output = _constrain(output, target_sharding, self.shard_mode)
        return output if return_dict else (output,)


class TransformerMultimodalLM(PretrainedModel):
    """Unified base class for Multimodal Language Models (Conditional Generation).

    Coordinates a text language backbone (e.g., ``TransformerCausalLM``), an optional
    vision encoder/tower, an optional audio encoder/tower, and multimodal projectors.
    Supports text generation conditioned on text, image, video, and audio inputs.

    Args:
        config: Model configuration containing text, vision, and projector settings.
        rngs: Random number generator for weight initialization.
        language_model: Language model instance or module class.
        decoder: Decoder layer module class used to build a ``TransformerCausalLM``
            if ``language_model`` is not directly provided.
        vision_tower: Vision encoder instance or module class.
        multi_modal_projector: Multimodal projection layer instance or module class.
        audio_tower: Audio encoder instance or module class.
        audio_projector: Audio projection layer instance or module class.
        image_token_id: Token ID representing image placeholders in ``input_ids``.
        video_token_id: Token ID representing video placeholders in ``input_ids``.
        audio_token_id: Token ID representing audio placeholders in ``input_ids``.
        mesh: JAX device mesh for explicit sharding.
        sharding_rules: Logical-to-mesh axis mapping rules.
    """

    default_sharding_rules = [
        ('vocab', 'tp'),
        ('embed', None),
        ('heads', 'tp'),
        ('kv_heads', 'tp'),
        ('head_dim', None),
        ('mlp', 'tp'),
        ('batch', 'fsdp'),
        ('sequence', None),
    ]

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        rngs: nn.Rngs | None = None,
        language_model: nn.Module | type[nn.Module] | None = None,
        vision_tower: nn.Module | type[nn.Module] | None = None,
        multi_modal_projector: nn.Module | type[nn.Module] | None = None,
        audio_tower: nn.Module | type[nn.Module] | None = None,
        audio_projector: nn.Module | type[nn.Module] | None = None,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        **kwargs: tp.Any,
    ) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)

        self.config = config
        self.dtype = (
            getattr(config, 'torch_dtype', None)
            or getattr(config, 'dtype', None)
            or 'bfloat16'
        )
        self.shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)

        # 1. Text Language Model Backbone
        if language_model is not None:
            if isinstance(language_model, type) and issubclass(language_model, nn.Module):
                self.language_model = language_model(
                    config=config,
                    rngs=rngs,
                    mesh=mesh,
                    sharding_rules=sharding_rules,
                    **kwargs,
                )
            else:
                self.language_model = language_model
        else:
            self.language_model = TransformerCausalLM(
                config=config,
                rngs=rngs,
                mesh=mesh,
                sharding_rules=sharding_rules,
                **kwargs,
            )

        # 2. Vision Encoder / Tower
        if isinstance(vision_tower, type) and issubclass(vision_tower, nn.Module):
            self.vision_tower = vision_tower(config=config, rngs=rngs)
        else:
            self.vision_tower = vision_tower

        # 3. Multimodal Vision Projector
        if isinstance(multi_modal_projector, type) and issubclass(multi_modal_projector, nn.Module):
            self.multi_modal_projector = multi_modal_projector(config=config, rngs=rngs)
        else:
            self.multi_modal_projector = multi_modal_projector

        # 4. Audio Tower & Audio Projector
        if isinstance(audio_tower, type) and issubclass(audio_tower, nn.Module):
            self.audio_tower = audio_tower(config=config, rngs=rngs)
        else:
            self.audio_tower = audio_tower

        if isinstance(audio_projector, type) and issubclass(audio_projector, nn.Module):
            self.audio_projector = audio_projector(config=config, rngs=rngs)
        else:
            self.audio_projector = audio_projector

        # 5. Media Token IDs
        self.image_token_id = (
            kwargs.get('image_token_id')
            or getattr(config, 'image_token_id', None)
            or getattr(getattr(config, 'vision_config', None), 'image_token_id', None)
        )
        self.video_token_id = kwargs.get('video_token_id') or getattr(config, 'video_token_id', None)
        self.audio_token_id = kwargs.get('audio_token_id') or getattr(config, 'audio_token_id', None)

    def get_input_embeddings(self) -> nn.Module | None:
        if self.language_model is not None and hasattr(self.language_model, 'get_input_embeddings'):
            return self.language_model.get_input_embeddings()
        elif self.language_model is not None and hasattr(self.language_model, 'model'):
            return self.language_model.model.embed_tokens
        elif hasattr(self, 'embed_tokens'):
            return self.embed_tokens
        return None

    def get_output_embeddings(self) -> nn.Module | None:
        if self.language_model is not None and hasattr(self.language_model, 'get_output_embeddings'):
            return self.language_model.get_output_embeddings()
        elif self.language_model is not None and hasattr(self.language_model, 'lm_head'):
            return self.language_model.lm_head
        elif hasattr(self, 'lm_head'):
            return self.lm_head
        return None

    def get_language_model(self) -> nn.Module | None:
        return self.language_model

    def get_vision_tower(self) -> nn.Module | None:
        return self.vision_tower

    def get_multi_modal_projector(self) -> nn.Module | None:
        return self.multi_modal_projector

    def enable_remat(self) -> None:
        if self.language_model is not None and hasattr(self.language_model, 'enable_remat'):
            self.language_model.enable_remat()
        if self.vision_tower is not None and hasattr(self.vision_tower, 'enable_remat'):
            self.vision_tower.enable_remat()

    def encode_vision(self, pixel_values: jax.Array, **kwargs: tp.Any) -> jax.Array:
        """Encode vision inputs and project features into hidden dimension."""
        if self.vision_tower is None:
            raise ValueError("vision_tower is not configured for this model")
        vision_outputs = self.vision_tower(pixel_values, **kwargs)
        if isinstance(vision_outputs, tuple):
            vision_features = vision_outputs[0]
        else:
            vision_features = vision_outputs

        if self.multi_modal_projector is not None:
            vision_features = self.multi_modal_projector(vision_features)
        return vision_features

    def encode_audio(self, input_features: jax.Array, **kwargs: tp.Any) -> jax.Array:
        """Encode audio inputs and project features into hidden dimension."""
        if self.audio_tower is None:
            raise ValueError("audio_tower is not configured for this model")
        audio_outputs = self.audio_tower(input_features, **kwargs)
        if isinstance(audio_outputs, tuple):
            audio_features = audio_outputs[0]
        else:
            audio_features = audio_outputs

        if self.audio_projector is not None:
            audio_features = self.audio_projector(audio_features)
        return audio_features

    def merge_multimodal_embeddings(
        self,
        input_ids: jax.Array,
        inputs_embeds: jax.Array,
        multimodal_features: jax.Array,
        media_token_id: int,
    ) -> jax.Array:
        """Merge/splice multimodal feature vectors into inputs_embeds at media_token_id positions."""
        if media_token_id is None or multimodal_features is None:
            return inputs_embeds

        mask = (input_ids == media_token_id)
        flat_features = multimodal_features.reshape(-1, multimodal_features.shape[-1])
        target_shape = inputs_embeds.shape
        reshaped_features = flat_features[:target_shape[0] * target_shape[1]].reshape(target_shape)
        return jnp.where(mask[..., None], reshaped_features, inputs_embeds)

    def __call__(
        self,
        input_ids: jax.Array | None = None,
        pixel_values: jax.Array | None = None,
        input_features: jax.Array | None = None,
        pixel_attention_mask: jax.Array | None = None,
        image_sizes: jax.Array | None = None,
        inputs_embeds: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        ctx: TransformerContext | None = None,
        logits_to_keep: int | jax.Array = 0,
        image_token_id: int | None = None,
        audio_token_id: int | None = None,
        **kwargs: tp.Any,
    ) -> tuple[jax.Array, TransformerContext | None]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You should specify either input_ids or inputs_embeds")

            embed_fn = self.get_input_embeddings()
            if embed_fn is not None:
                inputs_embeds = embed_fn(input_ids)
            else:
                inputs_embeds = input_ids

            # Process vision features
            if pixel_values is not None and self.vision_tower is not None:
                vision_features = self.encode_vision(pixel_values, **kwargs)
                img_tok_id = image_token_id or self.image_token_id
                if img_tok_id is not None:
                    inputs_embeds = self.merge_multimodal_embeddings(
                        input_ids, inputs_embeds, vision_features, img_tok_id
                    )

            # Process audio features
            if input_features is not None and self.audio_tower is not None:
                audio_features = self.encode_audio(input_features, **kwargs)
                aud_tok_id = audio_token_id or self.audio_token_id
                if aud_tok_id is not None:
                    inputs_embeds = self.merge_multimodal_embeddings(
                        input_ids, inputs_embeds, audio_features, aud_tok_id
                    )

        # Delegate to language model backbone
        if self.language_model is not None:
            return self.language_model(
                inputs_embeds,
                attention_mask=attention_mask,
                ctx=ctx,
                logits_to_keep=logits_to_keep,
            )
        elif hasattr(self, 'model'):
            x, new_cache = self.model(
                inputs_embeds,
                attention_mask=attention_mask,
                kv_cache=(ctx.key_cache, ctx.value_cache) if ctx and ctx.key_cache is not None else None,
                position_idx=ctx.position_idx if ctx else None,
                is_causal=ctx.is_causal if ctx else False,
            )
            out_embed = self.get_output_embeddings()
            logits = out_embed(x) if out_embed is not None else x
            if ctx is not None and new_cache is not None:
                ctx = replace(ctx, key_cache=new_cache[0], value_cache=new_cache[1])
            return logits, ctx
        else:
            raise NotImplementedError("Subclass should implement forward pass or provide language_model / model")

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: tp.Any,
        mesh: tp.Any = None,
        sharding_rules: tp.Any = None,
        local: bool = False,
        module_map: tp.List | None = None,
        **kwargs: tp.Any,
    ) -> tp.Any:
        kwargs = dict(kwargs)
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )

        rules = [
            ("model.language_model.", "language_model.model."),
            ("embed_tokens.weight", "embed_tokens.embedding"),
        ]

        if getattr(config, 'tie_word_embeddings', False):
            rules.append(
                ('lm_head.weight', 'language_model.model.embed_tokens.embedding')
            )

        if module_map is not None:
            rules.extend(module_map)

        return super().from_pretrained(
            path_or_repo,
            config=config,
            module_map=rules,
            local=local,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )

    def generate(
        self,
        input_ids: jax.Array,
        max_new_tokens: int = 20,
        pixel_values: jax.Array | None = None,
        input_features: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | list[int] | tuple[int, ...] | None = None,
        pad_token_id: int | None = None,
        seed: int = 42,
        streamer: tp.Any = None,
        attention_kernel: str | Mapping[str, str] = 'auto',
    ) -> jax.Array:
        """Autoregressively generate tokens conditioned on text and optional multimodal inputs."""
        if self.language_model is not None and hasattr(self.language_model, 'generate'):
            if pixel_values is not None or input_features is not None:
                logits, ctx = self(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    input_features=input_features,
                    attention_mask=attention_mask,
                )
            return self.language_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                seed=seed,
                streamer=streamer,
                attention_kernel=attention_kernel,
            )
        else:
            raise NotImplementedError("Generation requires a configured language_model")

class TransformerDecoderLayer(nn.Module):
    _norm1 = nn.RMSNorm
    _norm2 = nn.RMSNorm
    _attention = Attention
    _ffn = GateMLP
    _attention_kwargs = {}

    def __init__(self, config: ModelConfig, *, rngs: nn.Rngs, layer_idx: int | None = None, **kwargs: tp.Any) -> None:
        layer_types = config.layer_types
        if layer_types is not None and layer_idx is None:
            raise ValueError(f'{self.__class__.__name__} requires layer_idx')

        self.use_sliding_window = False
        self.sliding_pattern = None
        window_size = None
        if layer_types is not None and layer_idx is not None:
            if len(layer_types) != config.num_hidden_layers:
                raise ValueError(
                    'config.layer_types must contain one entry per layer'
                )
            sliding_pattern = tuple(
                layer_type in {'sliding_attention', 'sliding'}
                for layer_type in layer_types
            )
            self.sliding_pattern = sliding_pattern
            window_size = config.sliding_window
            self.use_sliding_window = jnp.asarray(
                sliding_pattern,
                dtype=jnp.bool_,
            )[layer_idx]

        _default_attention_kwargs = {
            'bias': config.attention_bias,
            'qk_norm': False,
            'window_size': window_size,
            'scaling': None,
            'softcap': None,
        }
        _default_attention_kwargs.update(self._attention_kwargs)
        apply_position_fn = kwargs.get('apply_position_fn', None)
        self.norm1 = self._norm1(
            config.hidden_size,
            config.rms_norm_eps,
            axis_names=config.rmsnorm_weight_axis_names,
            shard_mode=config.shard_mode
        )
        self.attention = self._attention(
            config.hidden_size,
            config.num_attention_heads,
            config.head_dim or config.hidden_size // config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            apply_position_fn=apply_position_fn,
            epsilon=config.rms_norm_eps,
            dropout=config.attention_dropout or 0.0,
            dtype=config.dtype,
            rngs=rngs,
            quant=config.quant,
            q_axis_names=config.attention_q_proj_axis_names,
            k_axis_names=config.attention_k_proj_axis_names,
            v_axis_names=config.attention_v_proj_axis_names,
            o_axis_names=config.attention_o_proj_axis_names,
            dot_general=config.dot_general,
            shard_mode=config.shard_mode,
            **_default_attention_kwargs,
        )
        self.norm2 = self._norm2(
            config.hidden_size,
            config.rms_norm_eps,
            axis_names=config.rmsnorm_weight_axis_names,
            shard_mode=config.shard_mode
        )
        self.ffn = self._ffn(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act or config.hidden_activation,
            bias=bool(config.mlp_bias),
            dtype=config.dtype,
            rngs=rngs,
            gate_axis_names=config.gatemlp_gate_proj_axis_names,
            up_axis_names=config.gatemlp_up_proj_axis_names,
            down_axis_names=config.gatemlp_down_proj_axis_names,
            shard_mode=config.shard_mode,
            quant=config.quant,
            dot_general=config.dot_general
        )

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        is_causal: bool = False,
        kv_cache: tp.Tuple[jax.Array, jax.Array] | None = None,
        cache_position: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        position_embedding: jax.Array | None = None,
        boundary_ids: jax.Array | None = None,
        kernel: str = 'dot_product',
        out_sharding: tp.Any = None,
        **kwargs: tp.Any,
    ) -> tp.Tuple[jax.Array, tp.Tuple[jax.Array, jax.Array]]:
        z = x
        x, updated_cache = self.attention(
            self.norm1(x, out_sharding=out_sharding),
            attention_mask=attention_mask,
            is_causal=is_causal,
            kv_cache=kv_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embedding=position_embedding,
            boundary_ids=boundary_ids,
            use_sliding_window=self.use_sliding_window,
            kernel=kernel,
            out_sharding=out_sharding
        )
        x = x + z
        x = self.ffn(
            self.norm2(x, out_sharding=out_sharding),
            out_sharding=out_sharding
        ) + x
        return x, updated_cache

class TransformerModel(nn.Module):
    _layer_type = None
    _token_embedding = nn.Embedding
    _norm = nn.RMSNorm

    def __init__(self, config: ModelConfig, *, rngs: nn.Rngs, **kwargs) -> None:
        if self._layer_type is None:
            raise ValueError('_layer_type cannot be None')
        self.config = config
        head_dim = (
            config.head_dim
            or config.hidden_size // config.num_attention_heads
        )
        rope_options = config.rope_parameters or config.rope_scaling
        rope_theta = config.rope_theta
        if rope_theta is None and rope_options is not None:
            rope_theta = rope_options.get('rope_theta')

        self.rotary_embedding = _RotaryEmbedding(
            head_dim,
            config.max_position_embeddings,
            base=rope_theta or 10_000.0,
            rope_scaling=rope_options,
        )
        self.token_embedding = self._token_embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
            rngs=rngs,
            quant=config.quant,
            axis_names=config.token_embedding_embedding_axis_names,
            shard_mode=config.shard_mode
        )
        self.layers = StackLayer.init_stack(
            self._layer_type,
            config,
            num_stacks=config.num_hidden_layers,
            stack_type=config.stack_type,
            rngs=rngs,
            apply_position_fn=self.rotary_embedding.apply_rope,
            **kwargs
        )
        self.norm = self._norm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype='float32',
            axis_names=config.rmsnorm_weight_axis_names,
            shard_mode=config.shard_mode,
        )
        self.remat = False

    def enable_remat(self) -> None:
        self.remat = True

    def disable_remat(self) -> None:
        self.remat = False

    def _position_embeddings(
        self,
        x: jax.Array,
        position_ids: jax.Array | None,
    ) -> PositionEmbedding:
        return self.rotary_embedding(x, position_ids)

    def _position_embedding_for_layer(
        self,
        position_embeddings: PositionEmbeddings,
        layer_idx: jax.Array,
    ) -> PositionEmbedding:
        del layer_idx
        if isinstance(position_embeddings, tp.Mapping):
            raise TypeError(f'{self.__class__.__name__} expects one rotary position embedding')
        cosine, sine = position_embeddings
        return cosine, sine

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_cache: tp.Tuple[jax.Array, jax.Array] | None = None,
        position_ids: jax.Array | None = None,
        position_embedding: tp.Tuple[jax.Array, jax.Array] | None = None,
        is_causal: bool = False,
        kernel: str = 'dot_product',
        out_sharding: tp.Any = None,
        **kwargs: tp.Any
    ) -> ModelOutput:
        x = self.token_embedding(input_ids, out_sharding=out_sharding)
        if position_embedding is None:
            position_embedding = self._position_embeddings(x, position_ids)

        def forward(
            layer: nn.Module,
            hidden_states: jax.Array,
            layer_cache: tp.Any,
            layer_idx: jax.Array,
        ) -> tuple[jax.Array, tp.Any]:
            return layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                position_ids=position_ids,
                position_embedding=self._position_embedding_for_layer(
                    position_embedding, layer_idx
                ),
                layer_idx=layer_idx,
                is_causal=is_causal,
                kernel=kernel,
                out_sharding=out_sharding,
                **kwargs,
            )

        apply_layer = forward
        if self.remat:
            apply_layer = jax.checkpoint(
                forward,
                prevent_cse=isinstance(self.layers, nn.List),
            )

        x, cache = StackLayer.call_stack(
            self.layers,
            apply_layer,
            x,
            per_layer=kv_cache,
            with_layer_index=True,
        )
        x = self.norm(x, out_sharding=out_sharding)
        return ModelOutput(x=x, kv_cache=cache)

class TransformerCausalLM(PretrainedModel):
    _model_type = None
    _default_sharding_rules = (
        ('vocab', 'tp'),
        ('embed', None),
        ('heads', 'tp'),
        ('kv_heads', 'tp'),
        ('head_dim', None),
        ('mlp', 'tp'),
        ('batch', 'fsdp'),
        ('sequence', None),
    )
    _default_module_map = [
        ('model.embed_tokens.weight', 'model.token_embedding.embedding'),
        ('input_layernorm', 'norm1'),
        ('self_attn', 'attention'),
        ('post_attention_layernorm', 'norm2'),
        ('mlp', 'ffn'),
    ]
    _axis_names = {
        'token_embedding_embedding_axis_names': ('vocab', 'embed'),
        'rmsnorm_weight_axis_names': ('embed',),
        'attention_q_proj_axis_names': ('embed', 'heads', 'head_dim'),
        'attention_k_proj_axis_names': ('embed', 'kv_heads', 'head_dim'),
        'attention_v_proj_axis_names': ('embed', 'kv_heads', 'head_dim'),
        'attention_o_proj_axis_names': ('heads', 'head_dim', 'embed'),
        'gatemlp_gate_proj_axis_names': ('embed', 'mlp'),
        'gatemlp_up_proj_axis_names': ('embed', 'mlp'),
        'gatemlp_down_proj_axis_names': ('mlp', 'embed'),
        'lm_head_proj_axis_names': ('embed', 'vocab'),
    }
    _default_config = ModelConfig()
    def __init__(self, config: ModelConfig, *, rngs: nn.Rngs, **kwargs) -> None:
        if self._model_type is None:
            raise ValueError('_model_type cannot be None')

        kwargs = kwargs or {}
        config = self._default_config.with_overrides(config)
        quant = kwargs.pop('quant', config.quant)
        shard_mode = kwargs.pop(
            'shard_mode',
            config.shard_mode or ShardMode.AUTO,
        )
        dot_general = kwargs.pop('dot_general', config.dot_general)
        mesh = kwargs.pop('mesh', config.mesh)
        sharding_rules = kwargs.pop(
            'sharding_rules',
            config.sharding_rules,
        )
        stack_type = kwargs.pop('stack_type', config.stack_type)
        dtype = _validate_dtype_config(config)


        library_config_attributes = {
            **self._axis_names,
            'quant': quant,
            'shard_mode': shard_mode,
            'dot_general': dot_general,
            'mesh': mesh,
            'sharding_rules': sharding_rules,
            'stack_type': stack_type,
            'dtype': dtype,
        }

        config.__dict__.update(library_config_attributes)
        self.config = config
        self.logits_out_sharding = None
        if mesh is not None and shard_mode == ShardMode.EXPLICIT:
            self.logits_out_sharding = create_sharding(
                mesh,
                ('batch', 'sequence', 'vocab'),
                rules=(
                    sharding_rules or \
                        self._default_sharding_rules
                ),
            )

        self.model = self._model_type(config, rngs=rngs, **kwargs)
        self.tied_word_embeddings = bool(config.tie_word_embeddings)
        self.lm_head = None
        if not self.tied_word_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=config.dtype,
                rngs=rngs,
                quant=config.quant,
                dot_general=config.dot_general,
                axis_names=config.lm_head_proj_axis_names,
                shard_mode=config.shard_mode,
            )

    def enable_remat(self) -> None:
        self.model.enable_remat()

    def disable_remat(self) -> None:
        self.model.disable_remat()

    def _lm_weight(self) -> tp.Any:
        if self.tied_word_embeddings:
            return self.model.token_embedding.embedding.value.T
        if not isinstance(self.lm_head, nn.Linear):
            raise TypeError('untied language models require an nn.Linear head')
        return self.lm_head.weight

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_cache: tp.Tuple[jax.Array, jax.Array] | None = None,
        position_ids: jax.Array | None = None,
        position_embedding: jax.Array | None = None,
        out_sharding: tp.Any = None,
        logits_to_keep: int | jax.Array = 0,
        **kwargs: tp.Any
    ) -> ModelOutput:
        output: ModelOutput = self.model(
            input_ids,
            attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids,
            position_embedding=position_embedding,
            out_sharding=out_sharding,
            **kwargs,
        )
        x = output.pop('x')
        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError('logits_to_keep should be non-negative')
            if logits_to_keep:
                x = x[:, -logits_to_keep:, :]
        else:
            indices = jnp.asarray(logits_to_keep, dtype=jnp.int32)
            if indices.ndim != 1 or indices.shape[0] != x.shape[0]:
                raise ValueError(
                    'logits_to_keep should contain one index per batch row'
                )
            indices = jnp.where(indices < 0, indices + x.shape[1], indices)
            x = jnp.take_along_axis(
                x,
                indices[:, None, None],
                axis=1,
            )
        logits = self.compute_logits(
            x,
            self._lm_weight(),
            out_sharding=(
                self.logits_out_sharding
                if out_sharding is None
                else out_sharding
            ),
        )
        logits = self._process_logits(logits)
        return ModelOutput(logits=logits, **output)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        *,
        config: ModelConfig | None = None,
        module_map: tp.List | None = None,
        local: bool = False,
        **kwargs: tp.Any,
    ) -> tp.Self:
        if config is None:
            config = ModelConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )
        rules = list(cls._default_module_map)
        if config.tie_word_embeddings:
            rules.append(
                ('lm_head.weight', 'model.token_embedding.embedding')
            )
        if module_map:
            rules.extend(module_map)
        return PretrainedModel.from_pretrained.__func__(
            cls,
            path_or_repo,
            config,
            module_map=rules,
            local=local,
            **kwargs,
        )

    @staticmethod
    def compute_logits(
        lhs: jax.Array,
        rhs: tp.Any,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Contract the last ``lhs`` axis with the first ``rhs`` axis."""
        if isinstance(rhs, nn.Parameter):
            rhs = rhs.value
        dimension_numbers = (
            (((lhs.ndim - 1,), (0,))),
            ((), ()),
        )
        if isinstance(rhs, qwix.QArray):
            logits = qwix.dot_general(lhs, rhs, dimension_numbers)
        else:
            logits = jax.lax.dot_general(lhs, rhs, dimension_numbers)
        if out_sharding is not None:
            logits = jax.lax.with_sharding_constraint(
                logits,
                out_sharding,
            )
        return logits

    def _process_logits(self, logits: jax.Array) -> jax.Array:
        """Apply architecture-specific post-processing to vocabulary logits."""
        return logits

    def compute_causal_loss(
        self,
        input_ids: jax.Array,
        labels: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        ignore_index: int = -100,
        logits_chunk_size: int = 256,
        attention_kernel: str = 'dot_product',
        reduction: str = 'mean',
    ) -> jax.Array:
        """Compute causal loss without materializing full-sequence logits.

        The transformer runs once for the complete sequence. Vocabulary
        projection and cross entropy then run over rematerialized sequence
        chunks, bounding the live logits tensor to
        ``batch * logits_chunk_size * vocab_size``.
        """
        if reduction not in {'sum', 'mean'}:
            raise ValueError(
                'chunked causal loss supports only "sum" and "mean" '
                'reductions'
            )
        if not isinstance(logits_chunk_size, int) or isinstance(
            logits_chunk_size,
            bool,
        ) or logits_chunk_size <= 0:
            raise ValueError('logits_chunk_size must be a positive integer')

        input_ids = jnp.asarray(input_ids)
        labels = jnp.asarray(labels)
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

        token_mask = None
        model_attention_mask = None
        if attention_mask is not None:
            attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
            if attention_mask.ndim == 2:
                if attention_mask.shape != input_ids.shape:
                    raise ValueError(
                        'a two-dimensional attention_mask must match input_ids'
                    )
                token_mask = attention_mask
                model_attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.ndim in (3, 4):
                model_attention_mask = attention_mask
            else:
                raise ValueError(
                    'attention_mask must have two, three, or four dimensions'
                )

        if position_ids is not None:
            position_ids = jnp.asarray(position_ids, dtype=jnp.int32)
            if position_ids.shape != input_ids.shape:
                raise ValueError(
                    'position_ids and input_ids must have equal shapes'
                )

        model_output = self.model(
            input_ids,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            is_causal=True,
            kernel=attention_kernel,
        )
        hidden_states = model_output.x[:, :-1, :]
        shifted_labels = labels[:, 1:]
        target_mask = token_mask[:, 1:] if token_mask is not None else None
        if position_ids is not None:
            within_sequence = position_ids[:, 1:] != 0
            target_mask = (
                within_sequence
                if target_mask is None
                else target_mask & within_sequence
            )

        num_positions = hidden_states.shape[1]
        chunk_size = min(logits_chunk_size, num_positions)
        num_chunks = (num_positions + chunk_size - 1) // chunk_size
        padding = num_chunks * chunk_size - num_positions
        if padding:
            hidden_states = jnp.pad(
                hidden_states,
                ((0, 0), (0, padding), (0, 0)),
            )
            shifted_labels = jnp.pad(
                shifted_labels,
                ((0, 0), (0, padding)),
                constant_values=ignore_index,
            )
            if target_mask is not None:
                target_mask = jnp.pad(
                    target_mask,
                    ((0, 0), (0, padding)),
                    constant_values=False,
                )

        from taktiny.trainer.loss.classification import cross_entropy_loss

        lm_weight = self._lm_weight()

        @jax.checkpoint
        def scan_body(
            carry: tuple[jax.Array, jax.Array],
            index: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            start = index * chunk_size
            hidden_chunk = jax.lax.dynamic_slice_in_dim(
                hidden_states,
                start,
                chunk_size,
                axis=1,
            )
            label_chunk = jax.lax.dynamic_slice_in_dim(
                shifted_labels,
                start,
                chunk_size,
                axis=1,
            )
            mask_chunk = None
            if target_mask is not None:
                mask_chunk = jax.lax.dynamic_slice_in_dim(
                    target_mask,
                    start,
                    chunk_size,
                    axis=1,
                )

            logits = self.compute_logits(hidden_chunk, lm_weight)
            logits = self._process_logits(logits)
            chunk_loss = cross_entropy_loss(
                logits,
                label_chunk,
                mask=mask_chunk,
                ignore_index=ignore_index,
                reduction='sum',
            )
            selected = label_chunk != ignore_index
            if mask_chunk is not None:
                selected &= mask_chunk
            chunk_count = jnp.sum(selected, dtype=jnp.float32)
            loss_sum, count_sum = carry
            return (
                loss_sum + chunk_loss,
                count_sum + chunk_count,
            ), None

        (loss_sum, count), _ = jax.lax.scan(
            scan_body,
            (
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            jnp.arange(num_chunks, dtype=jnp.int32),
        )
        if reduction == 'sum':
            return loss_sum
        return loss_sum / jnp.maximum(count, 1.0)

    def _sample(
        self,
        logits: jax.Array,
        temperature: float,
        top_k: int,
        top_p: float,
        key: jax.Array,
        seen_tokens: jax.Array | None = None,
        repetition_penalty: float = 1.0,
    ) -> jax.Array:
        if logits.ndim != 2:
            raise ValueError('logits should have shape [batch, vocabulary]')
        if seen_tokens is not None:
            if seen_tokens.shape != logits.shape:
                raise ValueError(
                    'seen_tokens should have the same shape as logits'
                )
            penalized = jnp.where(
                logits < 0,
                logits * repetition_penalty,
                logits / repetition_penalty,
            )
            logits = jnp.where(seen_tokens, penalized, logits)

        greedy_tokens = jnp.argmax(logits, axis=-1)[:, None]
        logits = logits / jnp.maximum(temperature, 1e-5)

        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            top_k_logits, _ = jax.lax.top_k(logits, top_k)
            min_top_k = top_k_logits[:, -1:]
            logits = jnp.where(logits >= min_top_k, logits, -jnp.inf)

        if top_p < 1.0:
            sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
            sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
            cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask to the right to keep the first token that crosses the threshold
            sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
            sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)

            # Map back to original order
            indices_to_remove = jnp.empty_like(sorted_indices_to_remove)
            indices_to_remove = indices_to_remove.at[
                jnp.arange(logits.shape[0])[:, None], sorted_indices
            ].set(sorted_indices_to_remove)

            logits = jnp.where(indices_to_remove, -jnp.inf, logits)

        sampled_tokens = jax.random.categorical(key, logits)[:, None]
        return jnp.where(temperature <= 0, greedy_tokens, sampled_tokens)

    @staticmethod
    def _canonical_attention_kernel(kernel: str) -> str:
        if not isinstance(kernel, str):
            raise TypeError('attention kernel names must be strings')
        normalized = kernel.strip().lower()
        aliases = {
            'standard': 'dot_product',
            'jax': 'dot_product',
            'flash_attention': 'flash',
        }
        normalized = aliases.get(normalized, normalized)
        supported = {'auto', 'dot_product', 'flash'}
        if normalized not in supported:
            choices = ', '.join(sorted(supported))
            raise ValueError(
                f'unsupported attention kernel {kernel!r}; choose from '
                f'{choices}'
            )
        return normalized

    def _resolve_generation_attention_kernels(
        self,
        attention_kernel: str | Mapping[str, str],
    ) -> tuple[str, str]:
        if isinstance(attention_kernel, str):
            prefill = decode = self._canonical_attention_kernel(
                attention_kernel
            )
        elif isinstance(attention_kernel, Mapping):
            unknown = set(attention_kernel) - {'prefill', 'decode'}
            if unknown:
                names = ', '.join(sorted(map(str, unknown)))
                raise ValueError(
                    f'unknown attention_kernel phase keys: {names}'
                )
            prefill = self._canonical_attention_kernel(
                attention_kernel.get('prefill', 'auto')
            )
            decode = self._canonical_attention_kernel(
                attention_kernel.get('decode', 'auto')
            )
        else:
            raise TypeError(
                'attention_kernel must be a string or a mapping with '
                'prefill and decode keys'
            )

        if prefill == 'auto':
            prefill = 'dot_product'
        if decode == 'auto':
            decode = 'dot_product'
        return prefill, decode

    @partial(
        jax.jit,
        static_argnames=[
            'max_seq_len',
            'top_k',
            'top_p',
            'attention_kernel',
        ],
    )
    def _decode_step(
        self,
        carry: DecodeCarry,
        max_seq_len: int | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_ids: jax.Array | None = None,
        pad_token_id: int = 0,
        attention_kernel: str = 'dot_product',
    ) -> tuple[DecodeCarry, jax.Array]:
        (
            token,
            k_cache,
            v_cache,
            pos,
            rng,
            finished,
            seen_tokens,
        ) = carry

        if max_seq_len is None:
            raise ValueError('max_seq_len is required')
        if eos_token_ids is None:
            eos_token_ids = jnp.empty((0,), dtype=token.dtype)

        position_ids = pos[:, None]
        mask = jnp.arange(max_seq_len)[None, :] <= position_ids
        mask = mask[:, None, None, :]

        output = self(
            token,
            attention_mask=mask,
            kv_cache=(k_cache, v_cache),
            position_ids=position_ids,
            cache_position=position_ids,
            is_causal=False,
            kernel=attention_kernel,
            logits_to_keep=1,
        )
        step_logits = output.logits
        updated_cache = output.kv_cache
        if updated_cache is None:
            raise ValueError('model did not return an updated KV cache')
        updated_k_cache, updated_v_cache = updated_cache

        rng, subkey = jax.random.split(rng)
        next_t = self._sample(
            step_logits[:, -1, :],
            temperature,
            top_k,
            top_p,
            subkey,
            seen_tokens=seen_tokens,
            repetition_penalty=repetition_penalty,
        )
        next_t = jnp.where(
            finished[:, None],
            jnp.asarray(pad_token_id, dtype=next_t.dtype),
            next_t,
        )

        if eos_token_ids.shape[0]:
            newly_finished = jnp.any(
                next_t == eos_token_ids[None, :],
                axis=-1,
            )
        else:
            newly_finished = jnp.zeros_like(finished)

        active = ~finished
        finished = finished | newly_finished
        batch_indices = jnp.arange(next_t.shape[0])
        seen_tokens = seen_tokens.at[
            batch_indices,
            next_t[:, 0],
        ].set(True)

        return (
            next_t,
            updated_k_cache,
            updated_v_cache,
            pos + active.astype(pos.dtype),
            rng,
            finished,
            seen_tokens,
        ), next_t

    def _prepare_generation(
        self,
        input_ids: ArrayLike,
        max_new_tokens: int,
        *,
        attention_mask: ArrayLike | None,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        eos_token_id: int | Sequence[int] | None,
        pad_token_id: int | None,
        seed: int,
        attention_kernel: str | Mapping[str, str],
    ) -> tuple[jax.Array, DecodeCarry, GenerationSettings]:
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError('max_new_tokens should be a positive integer')
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError('top_k should be a non-negative integer')
        if not 0 < top_p <= 1:
            raise ValueError('top_p should be in the interval (0, 1]')
        if repetition_penalty <= 0:
            raise ValueError('repetition_penalty should be positive')
        prefill_kernel, decode_kernel = (
            self._resolve_generation_attention_kernels(attention_kernel)
        )

        input_ids = jnp.asarray(input_ids)
        if input_ids.ndim != 2:
            raise ValueError('input_ids should have shape [batch, sequence]')
        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
        else:
            attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    'attention_mask should have the same shape as input_ids'
                )

        prompt_lengths = jnp.sum(
            attention_mask,
            axis=-1,
            dtype=jnp.int32,
        )
        if bool(jnp.any(prompt_lengths == 0)):
            raise ValueError('each prompt should contain at least one token')

        compact_order = jnp.argsort(
            ~attention_mask,
            axis=-1,
            stable=True,
        )
        compact_ids = jnp.take_along_axis(
            input_ids,
            compact_order,
            axis=-1,
        )
        compact_mask = (
            jnp.arange(seq_len)[None, :] < prompt_lengths[:, None]
        )

        eos_value = (
            getattr(self.config, 'eos_token_id', None)
            if eos_token_id is None
            else eos_token_id
        )
        if eos_value is None:
            eos_values = ()
        elif isinstance(eos_value, (list, tuple)):
            eos_values = tuple(int(token_id) for token_id in eos_value)
        else:
            eos_values = (int(eos_value),)
        eos_token_ids = jnp.asarray(eos_values, dtype=input_ids.dtype)

        if pad_token_id is None:
            pad_token_id = getattr(self.config, 'pad_token_id', None)
        if pad_token_id is None:
            pad_token_id = eos_values[0] if eos_values else 0
        pad_token_id = int(pad_token_id)

        if not isinstance(seed, int):
            raise TypeError('seed should be an integer')
        key = jax.random.key(seed)

        num_layers = len(self.model.layers)
        num_heads = getattr(self.config, 'num_attention_heads', None)
        num_kv_heads = getattr(self.config, 'num_key_value_heads', None)
        hidden_size = getattr(self.config, 'hidden_size', None)
        if num_heads is None:
            raise ValueError('config.num_attention_heads is required')
        if num_kv_heads is None:
            raise ValueError('config.num_key_value_heads is required')
        if hidden_size is None:
            raise ValueError('config.hidden_size is required')

        head_dim = (
            getattr(self.config, 'head_dim', None)
            or hidden_size // num_heads
        )
        max_seq_len = seq_len + max_new_tokens
        model_dtype = jnp.dtype(self.config.dtype)
        if not jnp.issubdtype(model_dtype, jnp.inexact):
            raise TypeError(
                'model compute dtype should be floating-point, '
                f'got {model_dtype}'
            )

        layer_types = getattr(self.config, 'layer_types', None)
        cache_layouts = []
        if layer_types is not None:
            global_head_dim = (
                getattr(self.config, 'global_head_dim', None)
                or head_dim
            )
            global_num_kv_heads = (
                getattr(
                    self.config,
                    'num_global_key_value_heads',
                    None,
                )
                or num_kv_heads
            )
            for layer_type in layer_types[:num_layers]:
                if layer_type in ('full_attention', 'full'):
                    cache_layouts.append(
                        (global_num_kv_heads, global_head_dim)
                    )
                else:
                    cache_layouts.append((num_kv_heads, head_dim))
        if not cache_layouts:
            cache_layouts = [(num_kv_heads, head_dim)] * num_layers

        unique_cache_layouts = set(cache_layouts)
        if len(unique_cache_layouts) != 1:
            raise ValueError(
                'generation currently requires one KV-cache layout across all '
                'layers'
            )
        cache_num_heads, cache_head_dim = cache_layouts[0]
        cache_shape = (
            num_layers,
            batch_size,
            max_seq_len,
            cache_num_heads,
            cache_head_dim,
        )
        key_cache = jnp.zeros(cache_shape, dtype=model_dtype)
        value_cache = jnp.zeros(cache_shape, dtype=model_dtype)
        position_ids = jnp.broadcast_to(
            jnp.arange(seq_len, dtype=jnp.int32)[None, :],
            (batch_size, seq_len),
        )
        prefill_mask = (
            jnp.arange(max_seq_len)[None, :]
            < prompt_lengths[:, None]
        )
        prefill_mask = prefill_mask[:, None, None, :]
        output = self(
            compact_ids,
            attention_mask=prefill_mask,
            kv_cache=(key_cache, value_cache),
            position_ids=position_ids,
            cache_position=position_ids,
            is_causal=True,
            kernel=prefill_kernel,
            logits_to_keep=prompt_lengths - 1,
        )
        logits = output.logits
        updated_cache = output.kv_cache
        if updated_cache is None:
            raise ValueError('model did not return an updated KV cache')
        key_cache, value_cache = updated_cache

        seen_tokens = jnp.zeros(
            (batch_size, self.config.vocab_size),
            dtype=jnp.int32,
        )
        batch_indices = jnp.broadcast_to(
            jnp.arange(batch_size)[:, None],
            compact_ids.shape,
        )
        seen_tokens = seen_tokens.at[
            batch_indices,
            compact_ids,
        ].add(compact_mask.astype(jnp.int32))
        seen_tokens = seen_tokens > 0

        key, subkey = jax.random.split(key)
        next_token = self._sample(
            logits[:, -1, :],
            temperature,
            top_k,
            top_p,
            subkey,
            seen_tokens=seen_tokens,
            repetition_penalty=repetition_penalty,
        )
        if eos_token_ids.shape[0]:
            finished = jnp.any(
                next_token == eos_token_ids[None, :],
                axis=-1,
            )
        else:
            finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        seen_tokens = seen_tokens.at[
            jnp.arange(batch_size),
            next_token[:, 0],
        ].set(True)

        carry = (
            next_token,
            key_cache,
            value_cache,
            prompt_lengths,
            key,
            finished,
            seen_tokens,
        )
        settings = (
            max_seq_len,
            eos_token_ids,
            pad_token_id,
            decode_kernel,
        )
        return input_ids, carry, settings

    def generate(
        self,
        input_ids: ArrayLike,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        seed: int = 42,
        attention_mask: ArrayLike | None = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int | Sequence[int] | None = None,
        pad_token_id: int | None = None,
        streamer: tp.Any = None,
        attention_kernel: str | Mapping[str, str] = 'auto',
    ) -> jax.Array:
        """Generate tokens using a direct tuple KV cache.

        ``attention_kernel`` accepts one kernel for both phases or a mapping
        such as ``{'prefill': 'flash', 'decode': 'dot_product'}``. ``'auto'``
        uses JAX dot-product attention for both phases; Flash Attention remains
        available as an explicit phase override.
        """
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError('max_new_tokens should be a non-negative integer')

        input_ids = jnp.asarray(input_ids)
        if input_ids.ndim != 2:
            raise ValueError('input_ids should have shape [batch, sequence]')
        if streamer is not None:
            if not callable(getattr(streamer, 'put', None)):
                raise TypeError('streamer should provide a callable put method')
            if not callable(getattr(streamer, 'end', None)):
                raise TypeError('streamer should provide a callable end method')

            streamer.put(jax.device_get(input_ids))
            generated = []
            try:
                for token_ids in self.stream_generate(
                    input_ids,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                    attention_mask=attention_mask,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    attention_kernel=attention_kernel,
                ):
                    streamer.put(jax.device_get(token_ids))
                    generated.append(token_ids)
            finally:
                streamer.end()

            if generated:
                return jnp.concatenate(
                    [input_ids, *generated],
                    axis=1,
                )
            return input_ids

        if max_new_tokens == 0:
            return input_ids

        input_ids, carry, settings = self._prepare_generation(
            input_ids,
            max_new_tokens,
            attention_mask=attention_mask,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            seed=seed,
            attention_kernel=attention_kernel,
        )
        (
            max_seq_len,
            eos_token_ids,
            pad_token_id,
            decode_kernel,
        ) = settings
        batch_size = input_ids.shape[0]
        generated = jnp.full(
            (batch_size, max_new_tokens),
            pad_token_id,
            dtype=input_ids.dtype,
        )
        generated = generated.at[:, 0].set(carry[0][:, 0])

        def cond_fn(
            loop_state: tuple[jax.Array, DecodeCarry, jax.Array],
        ) -> jax.Array:
            step, decode_carry, _ = loop_state
            return (step < max_new_tokens) & ~jnp.all(decode_carry[5])

        def body_fn(
            loop_state: tuple[jax.Array, DecodeCarry, jax.Array],
        ) -> tuple[jax.Array, DecodeCarry, jax.Array]:
            step, decode_carry, tokens = loop_state
            decode_carry, next_token = self._decode_step(
                decode_carry,
                max_seq_len=max_seq_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_ids=eos_token_ids,
                pad_token_id=pad_token_id,
                attention_kernel=decode_kernel,
            )
            tokens = tokens.at[:, step].set(next_token[:, 0])
            return step + 1, decode_carry, tokens

        generated_count, _, generated = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (jnp.asarray(1, dtype=jnp.int32), carry, generated),
        )
        generated_count = int(jax.device_get(generated_count))
        return jnp.concatenate(
            [input_ids, generated[:, :generated_count]],
            axis=1,
        )

    def stream_generate(
        self,
        input_ids: ArrayLike,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        seed: int = 42,
        attention_mask: ArrayLike | None = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int | Sequence[int] | None = None,
        pad_token_id: int | None = None,
        attention_kernel: str | Mapping[str, str] = 'auto',
    ) -> Iterator[jax.Array]:
        """Yield generated tokens using the same kernel policy as ``generate``."""
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError('max_new_tokens should be a non-negative integer')
        if max_new_tokens == 0:
            return

        _, carry, settings = self._prepare_generation(
            input_ids,
            max_new_tokens,
            attention_mask=attention_mask,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            seed=seed,
            attention_kernel=attention_kernel,
        )
        (
            max_seq_len,
            eos_token_ids,
            pad_token_id,
            decode_kernel,
        ) = settings
        yield carry[0]

        for _ in range(max_new_tokens - 1):
            if bool(jax.device_get(jnp.all(carry[5]))):
                break
            carry, next_token = self._decode_step(
                carry,
                max_seq_len=max_seq_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_ids=eos_token_ids,
                pad_token_id=pad_token_id,
                attention_kernel=decode_kernel,
            )
            yield next_token


class StackLayer(nn.Module):
    """Construct and execute repeated layers through one container-neutral API.

    ``nn.List`` executes layers with a Python loop while ``nn.SeqStack`` scans
    stacked parameters with :func:`jax.lax.scan`. ``call_stack`` presents the
    same callback contract for both containers and optionally slices a PyTree
    whose leading axis contains per-layer values, such as a KV cache.
    """

    @classmethod
    def init_stack(
        cls,
        layer_type: type[nn.Module],
        *args: tp.Any,
        num_stacks: int,
        stack_type: tp.Literal['list', 'stack'] | None = None,
        **kwargs: tp.Any,
    ) -> nn.List | nn.SeqStack:
        stack_type = stack_type or 'stack'
        if not isinstance(layer_type, type) or not issubclass(
            layer_type,
            nn.Module,
        ):
            raise TypeError('layer_type must be an nn.Module subclass')
        if not isinstance(num_stacks, int) or isinstance(num_stacks, bool):
            raise TypeError('num_stacks must be an integer')
        if num_stacks <= 0:
            raise ValueError('num_stacks must be positive')
        if stack_type not in {'list', 'stack'}:
            raise ValueError("stack_type must be 'list' or 'stack'")

        layers = []
        for layer_idx in range(num_stacks):
            layer = layer_type(
                *args,
                layer_idx=layer_idx,
                **kwargs,
            )
            layers.append(layer)

        if stack_type == 'list':
            return nn.List(layers)
        return nn.SeqStack(layers)

    @staticmethod
    def _validate_per_layer(
        per_layer: tp.Any,
        num_stacks: int,
    ) -> None:
        if per_layer is None:
            return
        for leaf in jax.tree.leaves(per_layer):
            shape = getattr(leaf, 'shape', None)
            if not shape:
                raise ValueError(
                    'every per_layer leaf must have a leading layer axis'
                )
            if shape[0] != num_stacks:
                raise ValueError(
                    'per_layer leading axes must match the number of layers; '
                    f'got {shape[0]} and {num_stacks}'
                )

    @staticmethod
    def _stack_outputs(outputs: tp.Sequence[tp.Any]) -> tp.Any:
        if not outputs or all(output is None for output in outputs):
            return None
        if any(output is None for output in outputs):
            raise ValueError(
                'all layers must either return an output or return None'
            )
        return jax.tree.map(lambda *values: jnp.stack(values), *outputs)

    @classmethod
    def call_stack(
        cls,
        layers: nn.List | nn.SeqStack,
        function: tp.Callable[
            [nn.Module, tp.Any, tp.Any],
            tuple[tp.Any, tp.Any],
        ],
        carry: tp.Any,
        *,
        per_layer: tp.Any = None,
        with_layer_index: bool = False,
    ) -> tuple[tp.Any, tp.Any]:
        if not isinstance(layers, (nn.List, nn.SeqStack)):
            raise TypeError('layers must be nn.List or nn.SeqStack')
        if not callable(function):
            raise TypeError('function must be callable')

        num_stacks = len(layers)
        cls._validate_per_layer(per_layer, num_stacks)

        if isinstance(layers, nn.List):
            outputs = []
            for layer_idx, layer in enumerate(layers):
                layer_input = None
                if per_layer is not None:
                    layer_input = jax.tree.map(
                        lambda value: value[layer_idx],
                        per_layer,
                    )
                if with_layer_index:
                    carry, output = function(
                        layer,
                        carry,
                        layer_input,
                        jnp.asarray(layer_idx, dtype=jnp.int32),
                    )
                else:
                    carry, output = function(layer, carry, layer_input)
                outputs.append(output)
            return carry, cls._stack_outputs(outputs)

        def scan_layer(
            layer: nn.Module,
            scan_carry: tuple[tp.Any, jax.Array],
        ) -> tuple[tuple[tp.Any, jax.Array], tp.Any]:
            current_carry, layer_idx = scan_carry
            layer_input = None
            if per_layer is not None:
                layer_input = jax.tree.map(
                    lambda value: jax.lax.dynamic_index_in_dim(
                        value,
                        layer_idx,
                        axis=0,
                        keepdims=False,
                    ),
                    per_layer,
                )
            if with_layer_index:
                current_carry, output = function(
                    layer,
                    current_carry,
                    layer_input,
                    layer_idx,
                )
            else:
                current_carry, output = function(
                    layer,
                    current_carry,
                    layer_input,
                )
            return (current_carry, layer_idx + 1), output

        (carry, _), outputs = layers(
            scan_layer,
            (carry, jnp.asarray(0, dtype=jnp.int32)),
        )
        return carry, outputs


__all__ = [
    'ModelOutput',
    'StackLayer',
    'PositionEmbedding',
    'PositionEmbeddings',
    'TransformerDecoderLayer',
    'TransformerModel',
    'TransformerCausalLM',
    'ConditionalTransformerLayer',
    'JointTransformerLayer',
    'GatedParallelTransformerLayer',
    'DiffusionTransformerModel',
    'TransformerContext',
    'TransformerMultimodalLM',
]
