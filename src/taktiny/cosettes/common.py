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
from taktiny.cosettes._base import PretrainedModel
from taktiny.maestro.config import ModelConfig
from taktiny.utils.typing import (
    ArrayLike,
    DType,
    LogicalRules,
    PathLike,
    ShardMode,
)
from taktiny.utils.sharding import create_sharding
from taktiny.layers import RotaryEmbedding, GateMLP, Attention


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
GenerationSettings = tuple[int, jax.Array, int]


def _approximate_gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x, approximate=True)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=['key_cache', 'value_cache', 'position_idx'],
    meta_fields=['is_causal'],
)
@dataclass(frozen=True)
class TransformerContext:
    key_cache: tp.Optional[jax.Array] = None
    value_cache: tp.Optional[jax.Array] = None
    position_idx: tp.Optional[jax.Array] = None
    is_causal: tp.Optional[bool] = None

class TransformerDecoderLayer(nn.Module):
    """An ordered transformer decoder block assembled from module types.

    Modules are created and executed in the same order as the keyword arguments
    passed to the constructor. Normalization modules transform the current
    hidden state, while attention and feed-forward modules form residual
    branches. Consecutive normalization modules allow architectures such as
    Gemma 2 to place norms on both sides of a residual branch.

    Attention modules receive the attention mask, causal flag, position IDs,
    and optional KV cache. The returned cache has the same per-layer
    ``(key_cache, value_cache)`` structure as the input cache.

    Args:
        config: Model configuration containing the hidden size, attention
            dimensions, positional embedding settings, and MLP settings.
        rngs: Random number generator used to initialize parameterized modules.
        layer_idx: Index of this layer in the model. This selects per-layer
            attention modes such as Gemma2's alternating sliding/full pattern.
        **modules: Ordered mapping from checkpoint-facing module names to
            ``nn.Module`` subclasses or initialized module instances. Supported
            types are normalization, ``Attention``, and ``GateMLP`` modules.

    Returns:
        A tuple containing the transformed hidden states and the updated KV
        cache, or ``None`` when no cache was supplied.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: tp.Optional[int] = None,
        **modules: nn.Module | type[nn.Module],
    ) -> None:
        if not modules:
            raise ValueError('TransformerDecoderLayer requires at least one module')

        shard_mode              = config.shard_mode or ShardMode.AUTO
        quant                   = config.quant
        dot_general             = config.dot_general
        hidden_size             = config.hidden_size
        num_heads               = config.num_attention_heads
        num_kv_heads            = config.num_key_value_heads
        max_position_embeddings = config.max_position_embeddings
        rope_parameters         = config.rope_parameters
        intermediate_size       = config.intermediate_size

        if isinstance(rope_parameters, dict):
            rope_theta_param = rope_parameters.get('rope_theta', None)
        elif rope_parameters is not None:
            rope_theta_param = getattr(rope_parameters, 'rope_theta', None)
        else:
            rope_theta_param = None

        rope_theta = (
            getattr(config, 'rope_theta', None)
            or rope_theta_param
            or 10000.0
        )

        required = {
            'hidden_size'               : hidden_size,
            'num_attention_heads'       : num_heads,
            'num_key_value_heads'       : num_kv_heads,
            'max_position_embeddings'   : max_position_embeddings,
            'rope_theta'                : rope_theta,
            'intermediate_size'         : intermediate_size,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"Missing required transformer config values: {', '.join(missing)}"
            )

        dtype = (
            getattr(config, 'torch_dtype', None)
            or getattr(config, 'dtype', None)
            or 'bfloat16'
        )
        head_dim        = config.head_dim or hidden_size // num_heads
        mlp_bias        = config.mlp_bias or False
        attention_bias  = config.attention_bias or False
        eps             = config.rms_norm_eps or 1e-6
        rope_scaling    = config.rope_scaling
        sliding_window  = config.sliding_window or config.use_sliding_window
        layer_types     = config.layer_types
        if layer_idx is not None and layer_types is not None:
            layer_type = layer_types[layer_idx]
            if layer_type in ('full_attention', 'full'):
                sliding_window  = None

        hidden_act = config.hidden_act or config.hidden_activation or config.act or 'silu'
        if hidden_act in ('gelu_pytorch_tanh', 'gelu_new', 'gelu_fast'):
            hidden_act          = _approximate_gelu
        attention_scaling       = None
        query_pre_attn_scalar   = config.query_pre_attn_scalar
        if query_pre_attn_scalar is not None:
            attention_scaling   = query_pre_attn_scalar ** -0.5
        attention_softcap       = config.attn_logit_softcapping
        attention_dropout       = config.attention_dropout or  0.0

        if hidden_size % num_heads != 0 and config.head_dim is None:
            raise ValueError(
                'hidden_size should be divisible by num_attention_heads when '
                'head_dim is not configured'
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                'num_attention_heads should be divisible by num_key_value_heads'
            )

        self.hidden_size = hidden_size
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.mlp_bias = mlp_bias
        self.eps = eps
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.layer_idx = layer_idx

        module_order = []
        module_kinds = []
        for name, module_type in modules.items():
            module, kind = self._create_module(
                name=name,
                module_type=module_type,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                sliding_window=sliding_window,
                hidden_act=hidden_act,
                attention_bias=attention_bias,
                attention_scaling=attention_scaling,
                attention_softcap=attention_softcap,
                attention_dropout=attention_dropout,
                mlp_bias=mlp_bias,
                eps=eps,
                dtype=dtype,
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
                rngs=rngs,
            )
            setattr(self, name, module)
            module_order.append(name)
            module_kinds.append(kind)

        self.module_order = tuple(module_order)
        self.module_kinds = tuple(module_kinds)

    @staticmethod
    def _create_module(
        *,
        name: str,
        module_type: nn.Module | type[nn.Module],
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: Mapping[str, tp.Any] | None,
        sliding_window: int | None,
        hidden_act: str | Callable[[jax.Array], jax.Array],
        attention_bias: bool,
        attention_scaling: float | None,
        attention_softcap: float | None,
        attention_dropout: float,
        mlp_bias: bool,
        eps: float,
        dtype: DType | str,
        shard_mode: ShardMode,
        quant: tp.Any,
        dot_general: Callable[..., jax.Array] | None,
        rngs: nn.Rngs,
    ) -> tuple[nn.Module, str]:
        if isinstance(module_type, nn.Module):
            module = module_type
            module_type = type(module)
        elif not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
            raise TypeError(f'{name} should be an nn.Module subclass or instance')
        elif issubclass(module_type, nn.RMSNorm):
            module = module_type(
                hidden_size,
                eps=eps,
                dtype=jnp.float32,
                shard_mode=shard_mode,
                axis_names=('embed',),
            )
        elif issubclass(module_type, nn.LayerNorm):
            module = module_type(
                hidden_size,
                eps=eps,
                axis_names=('embed',),
            )
        elif issubclass(module_type, Attention):
            module = module_type(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                pos_emb=RotaryEmbedding(
                    head_dim,
                    max_position_embeddings,
                    rope_theta,
                    rope_scaling,
                ),
                bias=False,
                qkv_norm_eps=eps,
                q_bias=attention_bias,
                k_bias=attention_bias,
                v_bias=attention_bias,
                o_bias=attention_bias,
                dtype=dtype,
                rngs=rngs,
                q_axis_names=('embed', 'heads', 'head_dim'),
                k_axis_names=('embed', 'kv_heads', 'head_dim'),
                v_axis_names=('embed', 'kv_heads', 'head_dim'),
                o_axis_names=('heads', 'head_dim', 'embed'),
                window_size=sliding_window,
                scaling=attention_scaling,
                softcap=attention_softcap,
                dropout=attention_dropout,
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
            )
        elif issubclass(module_type, GateMLP):
            module = module_type(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=hidden_act,
                bias=mlp_bias,
                dtype=dtype,
                rngs=rngs,
                gate_axis_names=('embed', 'mlp'),
                up_axis_names=('embed', 'mlp'),
                down_axis_names=('mlp', 'embed'),
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
            )
        else:
            raise TypeError(
                f'Unsupported decoder module {name}: {module_type.__name__}'
            )

        if issubclass(module_type, (nn.RMSNorm, nn.LayerNorm)):
            kind = 'norm'
        elif issubclass(module_type, Attention):
            kind = 'attention'
        elif issubclass(module_type, GateMLP):
            kind = 'residual'
        else:
            raise TypeError(
                f'Unsupported decoder module {name}: {module_type.__name__}'
            )

        return module, kind

    @staticmethod
    def _apply_norm(
        module: nn.Module,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None,
    ) -> jax.Array:
        if isinstance(module, nn.RMSNorm):
            return module(x, out_sharding=out_sharding)
        return module(x)

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_cache: KVCache | None = None,
        position_idx: jax.Array | None = None,
        is_causal: bool = False,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, KVCache | None]:
        residual = x
        pending = None
        new_cache = None

        for index, (name, kind) in enumerate(
            zip(self.module_order, self.module_kinds)
        ):
            module = getattr(self, name)
            if kind == 'norm':
                if pending is None:
                    x = self._apply_norm(module, x, out_sharding)
                    continue

                remaining_kinds = self.module_kinds[index:]
                next_residual = next(
                    (
                        offset
                        for offset, next_kind in enumerate(remaining_kinds)
                        if next_kind != 'norm'
                    ),
                    None,
                )

                if next_residual == 1:
                    x = residual + pending
                    residual = x
                    pending = None
                    x = self._apply_norm(module, x, out_sharding)
                else:
                    pending = self._apply_norm(module, pending, out_sharding)
                    if next_residual is not None:
                        x = residual + pending
                        residual = x
                        pending = None
                continue

            if pending is not None:
                x = residual + pending
                residual = x
                pending = None

            if kind == 'attention':
                pending, new_cache = module(
                    x,
                    attention_mask=attention_mask,
                    is_causal=is_causal,
                    kv_cache=kv_cache,
                    position_idx=position_idx,
                    out_sharding=out_sharding,
                )
            else:
                pending = module(x, out_sharding=out_sharding)

        if pending is not None:
            x = residual + pending

        return x, new_cache


class TransformerModel(nn.Module):
    """Token embedding followed by a list of transformer decoder layers.

    The supplied decoder type is instantiated ``config.num_hidden_layers``
    times. Each layer owns independent parameters initialized from the shared
    RNG stream. During a forward pass, a stacked KV cache is sliced by layer and
    rebuilt with the updated per-layer cache values. Per-token position IDs
    are shared by all layers and preserve packed-example boundaries when they
    reset to zero.

    Args:
        config: Model configuration containing ``num_hidden_layers``,
            ``vocab_size``, ``hidden_size``, and the decoder settings.
        rngs: Random number generator used for embeddings and decoder layers.
        module: Decoder-layer ``nn.Module`` subclass to repeat. It receives its
            zero-based ``layer_idx`` when instantiated.
        embedding: Embedding ``nn.Module`` subclass or initialized instance.
        norm: Optional final normalization module type or instance.
        use_list: Store decoder layers in an ``nn.List``. When false, store
            them in an ``nn.SeqStack`` and execute them with ``scan``.

    Returns:
        A tuple containing the final hidden states and an updated stacked
        ``(key_cache, value_cache)``, or ``None`` when caching is disabled.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        module: type[nn.Module],
        embedding: nn.Module | type[nn.Module],
        norm: nn.Module | type[nn.Module] | None = None,
        use_list: bool = True,
    ) -> None:
        if (num_hidden_layers := config.num_hidden_layers) is None:
            raise ValueError(
                'Missing required transformer config value: num_hidden_layers'
            )
        if (vocab_size := config.vocab_size) is None:
            raise ValueError('Missing required transformer config value: vocab_size')
        if (hidden_size := config.hidden_size) is None:
            raise ValueError('Missing required transformer config value: hidden_size')
        if not isinstance(num_hidden_layers, int) or num_hidden_layers < 1:
            raise ValueError('num_hidden_layers should be a positive integer')
        if not isinstance(module, type) or not issubclass(module, nn.Module):
            raise TypeError('module should be an nn.Module subclass')

        dtype = config.torch_dtype or config.dtype or 'bfloat16'
        quant = config.quant
        shard_mode = config.shard_mode or ShardMode.AUTO
        if isinstance(embedding, nn.Module):
            embed_tokens = embedding
        elif isinstance(embedding, type) and issubclass(embedding, nn.Module):
            embedding_kwargs = {
                'rngs': rngs,
                'dtype': dtype,
                'quant': quant,
            }
            if issubclass(embedding, nn.Embedding):
                embedding_kwargs.update(
                    axis_names=('vocab', 'embed'),
                    shard_mode=shard_mode,
                )
            embed_tokens = embedding(
                vocab_size,
                hidden_size,
                **embedding_kwargs,
            )
        else:
            raise TypeError('embedding should be an nn.Module subclass or instance')

        self.config             = config
        self.num_hidden_layers  = num_hidden_layers
        self.vocab_size         = vocab_size
        self.hidden_size        = hidden_size
        self.embed_tokens       = embed_tokens
        if hasattr(self.embed_tokens, 'embedding'):
            self.embed_tokens.embedding.axis_names = ('vocab', 'embed')
        if isinstance(self.embed_tokens, nn.Embedding):
            self.embed_tokens.shard_mode = shard_mode

        self.use_list = use_list
        if use_list:
            self.layers = nn.List(
                [
                    module(config, rngs=rngs, layer_idx=layer_idx)
                    for layer_idx in range(num_hidden_layers)
                ]
            )
        else:
            def stacked_layers() -> Iterator[nn.Module]:
                for layer_idx in range(num_hidden_layers):
                    layer = module(config, rngs=rngs, layer_idx=layer_idx)
                    layer.layer_idx = None
                    yield layer

            self.layers = nn.SeqStack(stacked_layers())

        self.norm = None
        if isinstance(norm, nn.Module):
            self.norm = norm
        elif isinstance(norm, type) and issubclass(norm, nn.RMSNorm):
            self.norm = norm(
                hidden_size,
                eps=config.rms_norm_eps or 1e-6,
                dtype=jnp.float32,
                shard_mode=config.shard_mode or ShardMode.AUTO,
                axis_names=('embed',),
            )
        elif isinstance(norm, type) and issubclass(norm, nn.LayerNorm):
            self.norm = norm(
                hidden_size,
                eps=config.layer_norm_eps or 1e-5,
                axis_names=('embed',),
            )
        elif norm is not None:
            raise TypeError(
                'norm should be a normalization nn.Module subclass or instance'
            )

        self.remat = False

    def enable_remat(self) -> None:
        self.remat = True

    def disable_remat(self) -> None:
        self.remat = False

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_cache: KVCache | None = None,
        position_idx: jax.Array | None = None,
        is_causal: bool = False,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, KVCache | None]:
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            if isinstance(self.embed_tokens, nn.Embedding):
                x = self.embed_tokens(x, out_sharding=out_sharding)
            else:
                x = self.embed_tokens(x)

        def call_layer(layer: tp.Any, hidden_states: tp.Any, layer_cache: tp.Any) -> tp.Any:
            return layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                position_idx=position_idx,
                is_causal=is_causal,
                out_sharding=out_sharding,
            )

        if self.remat:
            call_layer = jax.checkpoint(
                call_layer,
                prevent_cse=self.use_list,
            )

        if kv_cache is not None:
            if len(kv_cache) != 2:
                raise ValueError('kv_cache should contain key and value caches')

            key_cache, value_cache = kv_cache
            if key_cache.shape[0] != self.num_hidden_layers:
                raise ValueError(
                    'key cache should have one entry for each transformer layer'
                )
            if value_cache.shape[0] != self.num_hidden_layers:
                raise ValueError(
                    'value cache should have one entry for each transformer layer'
                )

        if not self.use_list:
            def apply_layer(layer: tp.Any, carry: tp.Any) -> tuple[tp.Any, ...]:
                hidden_states, layer_idx = carry
                layer_cache = None
                if kv_cache is not None:
                    layer_cache = (
                        jax.lax.dynamic_index_in_dim(
                            key_cache,
                            layer_idx,
                            axis=0,
                            keepdims=False,
                        ),
                        jax.lax.dynamic_index_in_dim(
                            value_cache,
                            layer_idx,
                            axis=0,
                            keepdims=False,
                        ),
                    )

                hidden_states, updated_cache = call_layer(
                    layer,
                    hidden_states,
                    layer_cache,
                )
                return (hidden_states, layer_idx + 1), updated_cache

            (x, _), new_cache = self.layers(
                apply_layer,
                (x, jnp.asarray(0, dtype=jnp.int32)),
            )
        else:
            new_key_cache = []
            new_value_cache = []
            for layer_idx, layer in enumerate(self.layers):
                layer_cache = None
                if kv_cache is not None:
                    layer_cache = (
                        key_cache[layer_idx],
                        value_cache[layer_idx],
                    )

                x, new_cache = call_layer(
                    layer,
                    x,
                    layer_cache,
                )

                if new_cache is not None:
                    new_key_cache.append(new_cache[0])
                    new_value_cache.append(new_cache[1])

            new_cache = None
            if new_key_cache:
                new_cache = (
                    jnp.stack(new_key_cache),
                    jnp.stack(new_value_cache),
                )

        if self.norm is not None:
            if isinstance(self.norm, nn.RMSNorm):
                x = self.norm(x, out_sharding=out_sharding)
            else:
                x = self.norm(x)

        return x, new_cache


class TransformerCausalLM(PretrainedModel):
    """Causal language model composed from an embedding, decoder, and LM head.

    ``TransformerModel`` owns the token embedding and repeated decoder layers.
    This wrapper projects its final hidden states to vocabulary logits. When
    word embeddings are tied and no explicit LM head is supplied, logits are
    computed directly with the embedding matrix so the parameter is registered
    only once. Otherwise, ``nn.Linear`` is used as the default LM head.

    Tying is read from ``config.tie_word_embeddings`` and also accepts the
    legacy ``tied_word_embeddings`` and ``tied_word_embedding`` spellings.

    Args:
        config: Model configuration containing vocabulary, hidden, decoder, and
            optional weight-tying settings.
        rngs: Random number generator used to initialize the model.
        embedding: Optional embedding module type or instance. Defaults to
            ``nn.Embedding``.
        decoder: Required decoder-layer ``nn.Module`` subclass repeated by
            ``TransformerModel``.
        norm: Optional final normalization module type or instance.
        lm_head: Optional output-head module type or instance. When omitted, a
            tied embedding projection or ``nn.Linear`` is selected from config.
        mesh: Optional JAX device mesh used for explicit sharding.
        sharding_rules: Optional logical-to-mesh axis mapping rules.
        use_list: Store decoder layers in an ``nn.List``. When false, use an
            ``nn.SeqStack`` executed with ``scan``.

    Returns:
        A tuple containing vocabulary logits and the updated
        ``TransformerContext``, or ``None`` when no context was supplied.

    ``position_ids`` passed to :meth:`__call__` may contain one position per
    token. Resetting positions to zero marks packed sequence boundaries for
    attention kernels without requiring a dense block-diagonal mask.
    """

    default_sharding_rules = (
        ('vocab', 'tp'),
        ('embed', None),
        ('heads', 'tp'),
        ('kv_heads', 'tp'),
        ('head_dim', None),
        ('mlp', 'tp'),
        ('batch', 'fsdp'),
        ('sequence', None),
    )

    def __init__(
        self, config: ModelConfig,
        *, rngs: nn.Rngs,
        embedding: nn.Module | type[nn.Module] | None = None,
        decoder: type[nn.Module] | None = None,
        norm: nn.Module | type[nn.Module] | None = None,
        lm_head: nn.Module | type[nn.Module] | None = None,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        use_list: bool = True,
    ) -> None:
        if decoder is None:
            raise ValueError('decoder is required')

        if embedding is None:
            embedding = nn.Embedding

        if (vocab_size := getattr(config, 'vocab_size', None)) is None:
            raise ValueError('Missing required transformer config value: vocab_size')

        if (hidden_size := getattr(config, 'hidden_size', None)) is None:
            raise ValueError('Missing required transformer config value: hidden_size')

        self.config = config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.shard_mode = config.shard_mode or ShardMode.AUTO
        self.quant = config.quant
        self.dot_general = config.dot_general
        self.dtype = config.torch_dtype or config.dtype or 'bfloat16'
        self.model = TransformerModel(
            config,
            rngs=rngs,
            module=decoder,
            embedding=embedding,
            norm=norm,
            use_list=use_list,
        )

        tied = config.tie_word_embeddings or config.tied_word_embedding or False
        self.tied_word_embeddings = tied and lm_head is None
        if self.tied_word_embeddings:
            if not hasattr(self.model.embed_tokens, 'embedding'):
                raise TypeError(
                    'A tied embedding should expose its weight as `embedding`'
                )
        else:
            if lm_head is None:
                lm_head = nn.Linear

            if isinstance(lm_head, nn.Module):
                self.lm_head = lm_head
            elif isinstance(lm_head, type) and issubclass(lm_head, nn.Module):
                self.lm_head = lm_head(
                    hidden_size,
                    vocab_size,
                    bias=False,
                    dtype=self.dtype,
                    rngs=rngs,
                    axis_names=('embed', 'vocab'),
                    shard_mode=self.shard_mode,
                    quant=self.quant,
                    dot_general=self.dot_general,
                )
            else:
                raise TypeError('lm_head should be an nn.Module subclass or instance')

        if sharding_rules is None:
            sharding_rules = self.default_sharding_rules

        self.model_out_sharding     = None
        self.logits_out_sharding    = None
        if mesh is not None and self.shard_mode == ShardMode.EXPLICIT:
            self.model_out_sharding = create_sharding(
                mesh,
                ('batch', 'sequence', 'embed'),
                rules=sharding_rules,
            )
            self.logits_out_sharding = create_sharding(
                mesh,
                ('batch', 'sequence', 'vocab'),
                rules=sharding_rules,
            )

    def enable_remat(self) -> None:
        self.model.enable_remat()

    def disable_remat(self) -> None:
        self.model.disable_remat()

    def __getattr__(self, name: str) -> tp.Any:
        if (
            name == 'lm_head'
            and self.__dict__.get('tied_word_embeddings', False)
        ):
            return self.model.embed_tokens
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        ctx: TransformerContext | None = None,
        logits_to_keep: int | jax.Array = 0,
    ) -> tuple[jax.Array, TransformerContext | None]:
        if ctx is not None and not isinstance(ctx, TransformerContext):
            raise TypeError('ctx should be a TransformerContext or None')

        kv_cache = None
        position_idx = None
        is_causal = False
        if ctx is not None:
            position_idx = ctx.position_idx
            is_causal = ctx.is_causal
            if (ctx.key_cache is None) != (ctx.value_cache is None):
                raise ValueError(
                    'TransformerContext should contain both key and value caches'
                )
            if ctx.key_cache is not None:
                kv_cache = (ctx.key_cache, ctx.value_cache)

        if position_ids is not None:
            if kv_cache is not None:
                raise ValueError(
                    'position_ids cannot be supplied with a KV cache; '
                    'TransformerContext.position_idx owns cached positions'
                )
            position_idx = jnp.asarray(position_ids, dtype=jnp.int32)

        x, new_cache = self.model(
            x,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_idx=position_idx,
            is_causal=is_causal,
            out_sharding=self.model_out_sharding,
        )

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

        if self.tied_word_embeddings:
            weight = self.model.embed_tokens.embedding.value
            if isinstance(weight, qwix.QArray):
                weight = weight.T
                dimension_numbers = (
                    (((x.ndim - 1,), (0,))),
                    ((), ()),
                )
                logits = qwix.dot_general(
                    x,
                    weight,
                    dimension_numbers,
                )
            else:
                logits = jnp.einsum('...d,vd->...v', x, weight)
            if self.logits_out_sharding is not None:
                logits = jax.lax.with_sharding_constraint(
                    logits,
                    self.logits_out_sharding,
                )
        else:
            logits = self.lm_head(
                x,
                out_sharding=self.logits_out_sharding,
            )

        if ctx is not None and new_cache is not None:
            ctx = replace(
                ctx,
                key_cache=new_cache[0],
                value_cache=new_cache[1],
            )

        return logits, ctx

    @classmethod
    def _load_from_pretrained(
        cls,
        path_or_repo: PathLike,
        config: ModelConfig,
        module_map: Mapping[str, str] | Sequence[tuple[str, str]] | None,
        **kwargs: tp.Any,
    ) -> tp.Self:
        module_map = module_map or []
        if isinstance(module_map, dict):
            module_map = list(module_map.items())

        new_module_map = list(module_map)
        tied = config.tie_word_embeddings or False
        if tied:
            embedding_target = None
            for rule in module_map:
                if len(rule) != 2:
                    continue
                _, target = rule
                if (
                    isinstance(target, str)
                    and target.endswith('embed_tokens.embedding')
                ):
                    embedding_target = target
                    break
            if embedding_target is not None:
                new_module_map.append(
                    ('lm_head.weight', embedding_target)
                )

        return super().from_pretrained(
            path_or_repo,
            config=config,
            module_map=new_module_map,
            **kwargs
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        local: bool = False,
        **kwargs: tp.Any,
    ) -> tp.Self:
        # Load config
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)

        # Define how HuggingFace weights map to components using new Tuple format
        module_map = [
            ("model.embed_tokens.weight", "model.embed_tokens.embedding"),
        ]

        # Call the base class safetensors loader
        # TODO: PretrainedModel.from_pretrained will need to be updated to pass mesh and sharding_rules down
        return cls._load_from_pretrained(
            path_or_repo,
            config,
            module_map,
            local=local,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs
        )

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
        if seen_tokens is not None:
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

    @partial(jax.jit, static_argnames=['max_seq_len', 'top_k', 'top_p'])
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

        decode_ctx = TransformerContext(
            key_cache=k_cache,
            value_cache=v_cache,
            position_idx=pos,
            is_causal=False
        )

        mask = jnp.arange(max_seq_len)[None, :] <= pos[:, None]
        mask = mask[:, None, None, :]

        step_logits, decode_ctx = self(
            token,
            attention_mask=mask,
            ctx=decode_ctx,
            logits_to_keep=1,
        )

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
            decode_ctx.key_cache,
            decode_ctx.value_cache,
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
    ) -> tuple[jax.Array, DecodeCarry, GenerationSettings]:
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError('max_new_tokens should be a positive integer')
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError('top_k should be a non-negative integer')
        if not 0 < top_p <= 1:
            raise ValueError('top_p should be in the interval (0, 1]')
        if repetition_penalty <= 0:
            raise ValueError('repetition_penalty should be positive')

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

        num_layers = getattr(self.config, 'num_hidden_layers', None)
        num_heads = getattr(self.config, 'num_attention_heads', None)
        num_kv_heads = getattr(self.config, 'num_key_value_heads', None)
        hidden_size = getattr(self.config, 'hidden_size', None)
        if num_layers is None:
            raise ValueError('config.num_hidden_layers is required')
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
        model_dtype = jnp.dtype(self.dtype)
        if not jnp.issubdtype(model_dtype, jnp.inexact):
            raise TypeError(
                'model compute dtype should be floating-point, '
                f'got {model_dtype}'
            )

        cache_shape = (
            num_layers,
            batch_size,
            max_seq_len,
            num_kv_heads,
            head_dim,
        )
        key_cache = jnp.zeros(cache_shape, dtype=model_dtype)
        value_cache = jnp.zeros(cache_shape, dtype=model_dtype)
        ctx = TransformerContext(
            key_cache=key_cache,
            value_cache=value_cache,
            position_idx=jnp.asarray(0, dtype=jnp.int32),
            is_causal=True,
        )
        prefill_mask = (
            jnp.arange(max_seq_len)[None, :]
            < prompt_lengths[:, None]
        )
        prefill_mask = prefill_mask[:, None, None, :]
        logits, ctx = self(
            compact_ids,
            attention_mask=prefill_mask,
            ctx=ctx,
            logits_to_keep=prompt_lengths - 1,
        )

        seen_tokens = jnp.zeros(
            (batch_size, self.vocab_size),
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
            ctx.key_cache,
            ctx.value_cache,
            prompt_lengths,
            key,
            finished,
            seen_tokens,
        )
        settings = (
            max_seq_len,
            eos_token_ids,
            pad_token_id,
        )
        return input_ids, carry, settings

    def generate(
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
        streamer: tp.Any = None,
    ) -> jax.Array:
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError('max_new_tokens should be a non-negative integer')

        input_ids = jnp.asarray(input_ids)
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
        )
        max_seq_len, eos_token_ids, pad_token_id = settings
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
    ) -> Iterator[jax.Array]:
        """Yield one generated token per batch row at each decode step."""
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
        )
        max_seq_len, eos_token_ids, pad_token_id = settings
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
            )
            yield next_token


class TransformerConditionalGeneration(PretrainedModel):
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
        decoder: type[nn.Module] | None = None,
        embedding: nn.Module | type[nn.Module] | None = None,
        norm: nn.Module | type[nn.Module] | None = None,
        lm_head: nn.Module | type[nn.Module] | None = None,
        vision_tower: nn.Module | type[nn.Module] | None = None,
        multi_modal_projector: nn.Module | type[nn.Module] | None = None,
        audio_tower: nn.Module | type[nn.Module] | None = None,
        audio_projector: nn.Module | type[nn.Module] | None = None,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        audio_token_id: int | None = None,
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
                    decoder=decoder,
                    embedding=embedding,
                    norm=norm,
                    lm_head=lm_head,
                    mesh=mesh,
                    sharding_rules=sharding_rules,
                    **kwargs,
                )
            else:
                self.language_model = language_model
        elif decoder is not None:
            self.language_model = TransformerCausalLM(
                config=config,
                rngs=rngs,
                decoder=decoder,
                embedding=embedding,
                norm=norm,
                lm_head=lm_head,
                mesh=mesh,
                sharding_rules=sharding_rules,
                **kwargs,
            )
        else:
            self.language_model = None

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
            image_token_id
            or getattr(config, 'image_token_id', None)
            or getattr(getattr(config, 'vision_config', None), 'image_token_id', None)
        )
        self.video_token_id = video_token_id or getattr(config, 'video_token_id', None)
        self.audio_token_id = audio_token_id or getattr(config, 'audio_token_id', None)

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
    def from_pretrained(cls, path_or_repo: tp.Any, mesh: tp.Any=None, sharding_rules: tp.Any=None, local: bool=False, **kwargs: tp.Any) -> tp.Any:
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)

        module_map = [
            ("model.embed_tokens.weight", "language_model.model.embed_tokens.embedding"),
            ("language_model.model.embed_tokens.weight", "language_model.model.embed_tokens.embedding"),
        ]

        return cls._load_from_pretrained(
            path_or_repo,
            config,
            module_map,
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
            )
        else:
            raise NotImplementedError("Generation requires a configured language_model")


class DiffusionIM(PretrainedModel):
    """Base class for Diffusion Image Models (e.g., Flux, DiT, Stable Diffusion)."""
    def __init__(self, config: tp.Any=None, *, rngs: nn.Rngs | None = None, **kwargs: tp.Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        self.config = config


class DiffusionLM(PretrainedModel):
    """Base class for Diffusion Language Models (e.g., Diffusion Gemma, Discrete Diffusion LM)."""
    def __init__(self, config: tp.Any=None, *, rngs: nn.Rngs | None = None, **kwargs: tp.Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        self.config = config


__all__ = [
    'TransformerContext',
    'TransformerDecoderLayer',
    'TransformerModel',
    'TransformerCausalLM',
    'TransformerConditionalGeneration',
    'DiffusionLM',
    'DiffusionIM',
]
