from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import jax

from taktiny import nn
from taktiny.cosettes._ordinario import DiffusionDenoiser
from taktiny.maestro.config import ModelConfig


class _LoadableComponent(nn.Module):
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, config: ModelConfig, *, marker: str = '') -> None:
        self.config = config
        self.marker = marker

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: Path,
        config: ModelConfig,
        *,
        local: bool,
        subfolder: str | None,
        marker: str = '',
        **kwargs: Any,
    ) -> _LoadableComponent:
        cls.calls.append(
            {
                'path': Path(path_or_repo),
                'local': local,
                'subfolder': subfolder,
                'marker': marker,
                'kwargs': kwargs,
            }
        )
        return cls(config, marker=marker)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class _Transformer(_LoadableComponent):
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(
        self,
        config: ModelConfig,
        *,
        marker: str = '',
        use_list: bool = True,
    ) -> None:
        super().__init__(config, marker=marker)
        self.use_list = use_list


class _Autoencoder(_LoadableComponent):
    calls: ClassVar[list[dict[str, Any]]] = []


class _ArrayDenoiser(DiffusionDenoiser):
    component_map = {
        'transformer': _Transformer,
        'vae': (_Autoencoder, 'autoencoder'),
    }

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.vae(self.transformer(x))


def _write_config(root: Path, subfolder: str, name: str) -> None:
    directory = root / subfolder
    directory.mkdir()
    with (directory / 'config.json').open('w') as config_file:
        json.dump({'component_name': name}, config_file)


def test_diffusion_denoiser_loads_each_component_from_its_subfolder(tmp_path):
    _Transformer.calls.clear()
    _Autoencoder.calls.clear()
    _write_config(tmp_path, 'transformer', 'transformer')
    _write_config(tmp_path, 'autoencoder', 'vae')

    model = _ArrayDenoiser.from_pretrained(
        tmp_path,
        local=True,
        marker='shared',
        component_kwargs={'vae': {'marker': 'vae-only'}},
    )

    assert model.component_names == ('transformer', 'vae')
    assert model.transformer.config.component_name == 'transformer'
    assert model.vae.config.component_name == 'vae'
    assert _Transformer.calls[0]['subfolder'] == 'transformer'
    assert _Transformer.calls[0]['marker'] == 'shared'
    assert _Autoencoder.calls[0]['subfolder'] == 'autoencoder'
    assert _Autoencoder.calls[0]['marker'] == 'vae-only'


def test_diffusion_denoiser_routes_use_list_only_to_compatible_components(
    tmp_path,
):
    _Transformer.calls.clear()
    _Autoencoder.calls.clear()
    _write_config(tmp_path, 'transformer', 'transformer')
    _write_config(tmp_path, 'autoencoder', 'vae')

    _ArrayDenoiser.from_pretrained(
        tmp_path,
        local=True,
        use_list=False,
    )

    assert _Transformer.calls[0]['kwargs']['use_list'] is False
    assert 'use_list' not in _Autoencoder.calls[0]['kwargs']


def test_diffusion_denoiser_accepts_preloaded_components(tmp_path):
    _Transformer.calls.clear()
    _Autoencoder.calls.clear()
    _write_config(tmp_path, 'autoencoder', 'vae')
    transformer = _Transformer(ModelConfig(component_name='preloaded'))

    model = _ArrayDenoiser.from_pretrained(
        tmp_path,
        local=True,
        components={'transformer': transformer},
    )

    assert model.transformer is transformer
    assert not _Transformer.calls
    assert len(_Autoencoder.calls) == 1


def test_diffusion_denoiser_rejects_incomplete_manual_composition():
    transformer = _Transformer(ModelConfig())

    try:
        _ArrayDenoiser(transformer=transformer)
    except ValueError as error:
        assert 'missing: vae' in str(error)
    else:
        raise AssertionError('an incomplete component map must fail')
