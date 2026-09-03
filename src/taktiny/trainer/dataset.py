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

import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import jax

from taktiny.trainer.config import DatasetConfig


class _GrainEpochLoader:
    """Build one resumable Grain loader for the selected epoch."""

    def __init__(
        self,
        source: Sequence[Any],
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        if not (
            callable(getattr(source, '__len__', None))
            and callable(getattr(source, '__getitem__', None))
        ):
            raise TypeError(
                'Non-streaming process_fn output must support __len__ and '
                '__getitem__ so Grain can read it'
            )
        self.source = source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        size = len(self.source)
        process_count = jax.process_count()
        process_index = jax.process_index()
        return max(
            0,
            (size + process_count - 1 - process_index) // process_count,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[Any]:
        import grain

        dataloader = grain.load(
            self.source,
            num_epochs=1,
            shuffle=self.shuffle,
            seed=(self.seed + self.epoch) % (2 ** 32),
            shard_options=grain.sharding.ShardByJaxProcess(),
        )
        return iter(dataloader)


def _split_loaded_dataset(dataset: Any) -> tuple[Any, Any | None]:
    if isinstance(dataset, tuple):
        if len(dataset) != 2:
            raise ValueError(
                'process_fn tuple output must contain '
                '(train, validation)'
            )
        return dataset

    if isinstance(dataset, Mapping):
        if 'train' not in dataset:
            raise ValueError(
                'Loaded dataset has no "train" split; process_fn must '
                'return train data or (train, validation)'
            )
        validation = dataset.get('validation')
        return dataset['train'], validation

    return dataset, None


def _load_dataset_splits(config: DatasetConfig) -> tuple[Any, Any | None]:
    from datasets import load_dataset

    token = os.environ.get('HF_TOKEN') or config.hf_token
    dataset = load_dataset(
        config.repo_id,
        streaming=config.streaming,
        token=token,
    )
    if config.process_fn is not None:
        dataset = config.process_fn(dataset)

    train, loaded_validation = _split_loaded_dataset(dataset)
    validation = config.validation_dataloader
    if validation is None:
        validation = loaded_validation
    return train, validation


def _load_dataset_from_repo(config: DatasetConfig) -> tuple[Any, Any | None]:
    train, validation = _load_dataset_splits(config)

    if config.streaming:
        return train, validation

    train = _GrainEpochLoader(
        train,
        shuffle=config.shuffle,
        seed=config.seed,
    )
    if validation is not None and validation is not config.validation_dataloader:
        validation = _GrainEpochLoader(
            validation,
            shuffle=False,
            seed=config.seed,
        )
    return train, validation


__all__ = ['_load_dataset_from_repo']