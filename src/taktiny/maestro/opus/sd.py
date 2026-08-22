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
"""Pretrained Stable Diffusion transformer architectures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import typing as tp

import jax

from taktiny.cosettes import layers as ly
from taktiny import nn
from taktiny.cosettes.continuo import _config_value
from taktiny.cosettes.transformers.ordinario import (
    DiffusionTransformerModel,
)
from taktiny.cosettes.transformers.sd import (
    SD3PatchEmbedding,
    SD3TransformerLayer,
)
from taktiny.maestro.livret import repertoire
from taktiny.maestro.config import ModelConfig
from taktiny.utils.typing import PathLike


_SD3_MODULE_MAP = (
    ('pos_embed.pos_embed', 'patch_embedding.pos_embed'),
    ('pos_embed.proj', 'patch_embedding.projection'),
    (
        'time_text_embed.timestep_embedder.linear_1',
        'condition_embedding.embeddings.timestep.1.projection',
    ),
    (
        'time_text_embed.timestep_embedder.linear_2',
        'condition_embedding.embeddings.timestep.1.output_projection',
    ),
    (
        'time_text_embed.text_embedder.linear_1',
        'condition_embedding.embeddings.pooled_projection.projection',
    ),
    (
        'time_text_embed.text_embedder.linear_2',
        'condition_embedding.embeddings.pooled_projection.output_projection',
    ),
    ('context_embedder', 'context_embedding'),
    ('transformer_blocks.', 'layers.'),
    ('norm_out', 'output_norm'),
    ('proj_out', 'output_projection'),
    ('.attn2.to_q', '.attn2.q_proj'),
    ('.attn2.to_k', '.attn2.k_proj'),
    ('.attn2.to_v', '.attn2.v_proj'),
    ('.attn2.to_out.0', '.attn2.o_proj'),
    ('.attn2.norm_q', '.attn2.q_norm'),
    ('.attn2.norm_k', '.attn2.k_norm'),
    ('.attn.to_q', '.attn.q_proj_1'),
    ('.attn.to_k', '.attn.k_proj_1'),
    ('.attn.to_v', '.attn.v_proj_1'),
    ('.attn.to_out.0', '.attn.o_proj_1'),
    ('.attn.add_q_proj', '.attn.q_proj_2'),
    ('.attn.add_k_proj', '.attn.k_proj_2'),
    ('.attn.add_v_proj', '.attn.v_proj_2'),
    ('.attn.to_add_out', '.attn.o_proj_2'),
    ('.attn.norm_q', '.attn.q_norm_1'),
    ('.attn.norm_k', '.attn.k_norm_1'),
    ('.attn.norm_added_q', '.attn.q_norm_2'),
    ('.attn.norm_added_k', '.attn.k_norm_2'),
    ('.ff.net.0.proj', '.ff.input'),
    ('.ff.net.2', '.ff.output'),
    ('.ff_context.net.0.proj', '.ff_context.input'),
    ('.ff_context.net.2', '.ff_context.output'),
)


class SD3TransformerModel(DiffusionTransformerModel):
    """Checkpoint-loadable Stable Diffusion 3 MMDiT backbone."""

    def __init__(self, config: ModelConfig, **kwargs: tp.Any) -> None:
        component_kwargs = {
            name: dict(values)
            for name, values in dict(
                kwargs.pop('component_kwargs', {}) or {}
            ).items()
        }
        patch_options = component_kwargs.setdefault('patch_embedding', {})
        patch_options.setdefault(
            'pos_embed_max_size',
            _config_value(config, 'pos_embed_max_size'),
        )
        patch_options.setdefault(
            'interpolation_scale',
            _config_value(
                config,
                'pos_embed_interpolation_scale',
                default=1.0,
            ),
        )
        patch_options.setdefault(
            'pos_embed_type',
            _config_value(config, 'pos_embed_type', default='sincos'),
        )
        patch_options.setdefault(
            'bias',
            bool(_config_value(config, 'patch_bias', default=True)),
        )
        super().__init__(
            config,
            transformer_layer=SD3TransformerLayer,
            patch_embedding=SD3PatchEmbedding,
            condition_embedding=ly.CombinedTimestepTextProjEmbedding,
            context_embedding=nn.Linear,
            output_norm=ly.AdaXNorm,
            output_projection=nn.Linear,
            component_kwargs=component_kwargs,
            **kwargs,
        )

    def __call__(
        self,
        x: jax.Array,
        enc_x: jax.Array,
        pooled_projections: jax.Array,
        timestep: jax.Array,
        controlnet_x: Sequence[jax.Array] | None = None,
        joint_attention_kwargs: Mapping[str, tp.Any] | None = None,
        return_dict: bool = True,
        skip_layers: Sequence[int] | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array | tuple[jax.Array]:
        attention_kwargs = dict(joint_attention_kwargs or {})
        attention_kwargs.pop('scale', None)
        if 'ip_adapter_image_embeds' in attention_kwargs:
            raise NotImplementedError(
                'IP-Adapter conditioning is not implemented by this model'
            )
        return super().__call__(
            x,
            enc_x,
            pooled_projections,
            timestep,
            control_residuals=controlnet_x,
            layer_kwargs=attention_kwargs,
            skip_layers=skip_layers,
            return_dict=return_dict,
            out_sharding=out_sharding,
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        config: ModelConfig | None = None,
        *,
        local: bool = False,
        subfolder: str | None = None,
        module_map: list[tp.Any] | None = None,
        weights_filename: str = 'diffusion_pytorch_model.safetensors',
        **kwargs: tp.Any,
    ) -> SD3TransformerModel:
        """Load a Diffusers-compatible SD3 transformer checkpoint."""
        if config is None:
            config = ModelConfig.load_config(
                path_or_repo,
                subfolder=subfolder,
                local=local,
            )
        if config is None:
            raise ValueError(
                f'unable to load SD3 transformer config from {path_or_repo}'
            )
        mappings = [*_SD3_MODULE_MAP, *(module_map or [])]
        return super().from_pretrained(
            path_or_repo,
            config,
            local=local,
            subfolder=subfolder,
            module_map=mappings,
            weights_filename=weights_filename,
            **kwargs,
        )


class_map = [
    ('SD3Transformer2DModel', SD3TransformerModel),
]

__all__ = []
for name, model_type in class_map:
    repertoire.register(name, model_type)
    __all__.append(model_type.__name__)
