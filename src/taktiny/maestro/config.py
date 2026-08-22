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
from __future__ import annotations
import typing as tp
from pathlib import Path
import warnings
from huggingface_hub import hf_hub_download
import json
from taktiny.utils.typing import PathLike


class ModelConfig:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = ModelConfig(**v)
            setattr(self, k, v)

    def __getattr__(self, name: str) -> tp.Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)

        # 1. Check nested text_config (common in HuggingFace multimodal models like Gemma 3/4, Qwen VL, Llama Vision)
        text_cfg = self.__dict__.get('text_config', None)
        if text_cfg is not None and text_cfg is not self:
            val = getattr(text_cfg, name, None)
            if val is not None:
                return val

        # 2. Check nested sub-configs (vision_config, encoder, decoder)
        for sub_cfg_name in ('vision_config', 'encoder', 'decoder'):
            sub_cfg = self.__dict__.get(sub_cfg_name, None)
            if sub_cfg is not None and sub_cfg is not self:
                val = getattr(sub_cfg, name, None)
                if val is not None:
                    return val

        # 3. Gracefully return None for missing keys
        return None

    def to_dict(self) -> dict[str, tp.Any]:
        """Recursively serialize ModelConfig to a plain dictionary."""
        output = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ModelConfig):
                output[k] = v.to_dict()
            elif isinstance(v, (list, tuple)):
                output[k] = [
                    elem.to_dict() if isinstance(elem, ModelConfig) else elem
                    for elem in v
                ]
            else:
                output[k] = v
        return output

    def with_overrides(self, overrides: ModelConfig) -> ModelConfig:
        """Return a new config with ``overrides`` layered over this config.

        Nested configuration dictionaries are merged recursively. Neither the
        defaults nor the overrides are mutated, so class-level default configs
        remain reusable across model instances.
        """
        if not isinstance(overrides, ModelConfig):
            raise TypeError('overrides must be a ModelConfig')

        def clone(value: tp.Any) -> tp.Any:
            if isinstance(value, dict):
                return {key: clone(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clone(item) for item in value]
            if isinstance(value, tuple):
                return tuple(clone(item) for item in value)
            return value

        def merge(
            defaults: dict[str, tp.Any],
            supplied: dict[str, tp.Any],
        ) -> dict[str, tp.Any]:
            result = clone(defaults)
            for key, value in supplied.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = clone(value)
            return result

        return ModelConfig(**merge(self.to_dict(), overrides.to_dict()))

    def get(self, key: tp.Any, default: tp.Any=None) -> tp.Any:
        """Return a configuration value using mapping-style semantics."""
        value = getattr(self, key, None)
        return default if value is None else value

    @classmethod
    def load_config(
        cls, path_or_repo: PathLike,
        filename: str = 'config.json',
        subfolder: tp.Any = None,
        local: bool = False
    ) -> tp.Self | None:
        if local:
            config_path = Path(path_or_repo).resolve()
            if subfolder:
                config_path = config_path / subfolder

            config_path = config_path / filename
            if not config_path.is_file():
                raise FileNotFoundError(
                    f'config file not found: {config_path}. With local=True, '
                    f'{path_or_repo!r} must be a local checkpoint directory '
                    f'containing {filename!r}.'
                )
        else:
            try:
                config_path = hf_hub_download(
                    repo_id=str(path_or_repo),
                    subfolder=subfolder if subfolder else None,
                    filename=filename
                )

            except Exception as e:
                print(f'config.json not found in repo: {e}')
                return None

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

        except Exception as e:
            print(f'Error loading config.json: {e}')
            return None

        return cls(**config)

    def __repr__(self) -> str:
        config_str = json.dumps(self.__dict__, indent=2, default=str)
        return f"{self.__class__.__name__} {config_str}"

__all__ = ['ModelConfig']


def _validate_dtype_config(config: ModelConfig):
    dtype = config.dtype or config.torch_dtype
    if dtype is None:
        import warnings
        warnings.warn('Not found `dtype` or `torch_dtype` in model config fallback to float32')
        dtype = 'float32'
    return dtype


def _verify_required_config_attributes(config: ModelConfig, config_attributes: tp.Sequence[str] | None) -> None:
    missing = []
    if config_attributes is None:
        return

    for attr in config_attributes:
        if not hasattr(config, attr):
            missing.append(attr)
    if len(missing) > 0:
        raise ValueError(f'Missing config attributes: {', '.join(missing)}.')
