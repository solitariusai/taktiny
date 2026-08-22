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
"""Supervised fine-tuning dataset configuration and trainer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import grain.python as grain
import numpy as np

from taktiny.data import (
    ApplyTemplate,
    BatchMap,
    CausalLMBatch,
    DatasetUtils,
    Map,
    PackSequences,
)
from taktiny.trainer.config import DatasetConfig, TrainingConfig
from taktiny.trainer.loss import causal_lm_loss
from taktiny.trainer.trainer import Trainer


@dataclass
class SFTDatasetConfig:
    """Build an SFT dataloader from exactly one source.

    ``dataloader`` takes precedence and disables all pipeline building, so the
    trainer then behaves exactly like the plain :class:`~taktiny.trainer.Trainer`.
    ``dataset`` may be an iterable or a random-access mapping (e.g. a Hugging
    Face ``Dataset``); ``repo_id`` loads from the Hub (non-streaming).

    ``text_field`` names the raw string column to tokenize with ``tokenizer``;
    each record then yields ``input_ids`` and ``labels`` (a full-text causal
    copy). When ``text_field`` is omitted, records are assumed to already carry
    ``input_ids`` (optionally ``labels`` and ``attention_mask``), and
    ``operations`` can prepend templates or other preprocessing. ``packing``
    concatenates sequences into fixed ``max_length`` records with reset
    ``position_ids`` and block-diagonal masking.
    """

    # Source (``dataloader`` wins and ignores the others).
    dataloader: Any = None
    dataset: Any = None
    repo_id: str | None = None
    streaming: bool = False

    # Tokenization.
    tokenizer: Any = None
    text_field: str | None = None
    max_length: int | None = None
    assistant_only: bool = False
    labels_fn: Any = None

    # Pipeline.
    operations: Sequence[Any] = ()
    packing: bool = False
    batch_size: int = 1
    epochs: int = 1
    shuffle: bool = True
    seed: int = 42
    drop_remainder: bool = False
    prefetch_size: int = 2

    # Optional evaluation.
    validation_dataloader: Any = None

    def _tokenize_record(
        self,
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Tokenize one record into variable-length input_ids/labels."""
        if self.assistant_only:
            messages = record.get('messages')
            if not messages:
                raise KeyError('assistant_only records need a messages field')
            full = self.tokenizer.apply_chat_template(
                [messages],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
            )[0]
            prompt = self.tokenizer.apply_chat_template(
                [messages[:-1]],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )[0]
            prompt_length = 0
            for prompt_id, input_id in zip(prompt, full):
                if prompt_id != input_id:
                    break
                prompt_length += 1
            if prompt_length == 0:
                raise ValueError('prompt and full sample have no common prefix')
            if self.max_length is not None:
                full = full[:self.max_length]
            input_ids = np.asarray(full, dtype=np.int32)
            labels = input_ids.copy()
            labels[:min(prompt_length, len(labels))] = -100
        elif self.text_field is not None:
            tokens = self.tokenizer(
                record[self.text_field],
                return_tensors='np',
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = np.asarray(tokens['input_ids'][0], dtype=np.int32)
            labels = (
                np.asarray(self.labels_fn(record), dtype=np.int32)
                if self.labels_fn is not None
                else input_ids.copy()
            )
        else:
            input_ids = np.asarray(record['input_ids'], dtype=np.int32)
            labels = np.asarray(
                record.get('labels', input_ids),
                dtype=np.int32,
            )

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': np.ones_like(input_ids, dtype=np.bool_),
        }

    def _streaming_iterable(self, source: Any) -> Any:
        """A non-Grain generator loader for streaming datasets."""
        if self.packing:
            raise NotImplementedError(
                'packing is not supported for streaming; use a non-streaming '
                'dataset or pass a dataloader'
            )
        if self.text_field is None and not self.assistant_only:
            if self.tokenizer is not None:
                raise ValueError(
                    'tokenizer supplied without text_field/assistant_only'
                )

        def generate() -> Iterator[dict[str, Any]]:
            batch: list[dict[str, Any]] = []
            for record in source:
                if not isinstance(record, Mapping):
                    raise TypeError('streaming records must be mappings')
                processed = self._tokenize_record(record)
                if self.max_length is not None:
                    processed = self._pad_record(processed)
                batch.append(processed)
                if len(batch) == self.batch_size:
                    yield self._stack(batch)
                    batch = []
            if batch and not self.drop_remainder:
                yield self._stack(batch)

        return generate()

    def _pad_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        max_length = self.max_length
        tokenizer = self.tokenizer
        pad_id = (
            getattr(tokenizer, 'pad_token_id', None)
            if tokenizer is not None
            else None
        )
        if pad_id is None:
            pad_id = 0
        input_ids = np.asarray(record['input_ids'])
        length = len(input_ids)
        if length > max_length:
            input_ids = input_ids[:max_length]
            length = max_length
        pad_len = max_length - length
        padded = np.pad(
            input_ids,
            (0, pad_len),
            constant_values=pad_id,
        ).astype(np.int32)
        mask = np.ones(max_length, dtype=np.bool_)
        mask[length:] = False
        labels = np.pad(
            np.asarray(record['labels'])[:length],
            (0, pad_len),
            constant_values=-100,
        ).astype(np.int32)
        return {
            'input_ids': padded,
            'labels': labels,
            'attention_mask': mask,
        }

    @staticmethod
    def _stack(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            key: np.stack([np.asarray(record[key]) for record in records])
            for key in records[0]
        }

    def _tokenize_operation(self) -> Any:
        if self.assistant_only:
            return self._assistant_only_tokenize()
        return self._text_tokenize()

    def _text_tokenize(self) -> Any:
        """Tokenize a raw ``text_field`` column into ``input_ids`` + labels."""
        tokenizer = self.tokenizer
        text_field = self.text_field
        max_length = self.max_length
        labels_fn = self.labels_fn

        def tokenize_batch(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            encoded: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, Mapping) or text_field not in row:
                    raise KeyError(
                        f'text_field {text_field!r} not found in record {row!r}'
                    )
                tokens = tokenizer(
                    row[text_field],
                    return_tensors='np',
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = np.asarray(
                    tokens['input_ids'][0],
                    dtype=np.int32,
                )
                if labels_fn is not None:
                    labels = np.asarray(labels_fn(row), dtype=np.int32)
                else:
                    labels = input_ids.copy()
                encoded.append({
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': np.ones_like(input_ids, dtype=np.bool_),
                })
            return encoded

        return BatchMap(tokenize_batch, batch_size=512)

    def _assistant_only_tokenize(self) -> Any:
        """Tokenize conversational records, masking prompt tokens in labels.

        Records must carry a ``messages`` field (a list of ``{'role',
        'content'}`` dicts). Prompt tokens are set to ``ignore_index`` so the
        loss is computed only on assistant turns, following
        ``apply_chat_template``.
        """
        tokenizer = self.tokenizer
        max_length = self.max_length
        if tokenizer is None or not callable(
            getattr(tokenizer, 'apply_chat_template', None)
        ):
            raise ValueError(
                'assistant_only requires a tokenizer with apply_chat_template'
            )

        def tokenize_batch(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            conversations = []
            for row in rows:
                messages = row.get('messages') if isinstance(row, Mapping) else None
                if not messages:
                    raise KeyError('assistant_only records need a messages field')
                conversations.append(messages)

            full = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
            )
            prompts = tokenizer.apply_chat_template(
                [messages[:-1] for messages in conversations],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )

            encoded: list[dict[str, Any]] = []
            for input_ids, prompt_ids in zip(full, prompts, strict=True):
                prompt_length = 0
                for prompt_id, input_id in zip(prompt_ids, input_ids):
                    if prompt_id != input_id:
                        break
                    prompt_length += 1
                if prompt_length == 0:
                    raise ValueError(
                        'prompt and full sample have no common prefix'
                    )
                if max_length is not None:
                    input_ids = input_ids[:max_length]
                input_ids = np.asarray(input_ids, dtype=np.int32)
                labels = input_ids.copy()
                labels[:min(prompt_length, len(labels))] = -100
                encoded.append({
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': np.ones_like(input_ids, dtype=np.bool_),
                })
            return encoded

        return BatchMap(tokenize_batch, batch_size=512)

    def _pad_operation(self) -> Any:
        """Right-pad a record to ``max_length`` for the non-packing path."""
        max_length = self.max_length
        tokenizer = self.tokenizer
        pad_id = (
            getattr(tokenizer, 'pad_token_id', None)
            if tokenizer is not None
            else None
        )
        if pad_id is None:
            pad_id = 0

        def pad(record: Mapping[str, Any]) -> dict[str, Any]:
            input_ids = np.asarray(record['input_ids'])
            length = len(input_ids)
            if length > max_length:
                input_ids = input_ids[:max_length]
                length = max_length
            pad_len = max_length - length
            padded = np.pad(
                input_ids,
                (0, pad_len),
                constant_values=pad_id,
            ).astype(np.int32)
            mask = np.ones(max_length, dtype=np.bool_)
            mask[length:] = False
            labels = np.asarray(record.get('labels', input_ids))
            labels = np.pad(
                labels[:length],
                (0, pad_len),
                constant_values=-100,
            ).astype(np.int32)
            return {
                'input_ids': padded,
                'labels': labels,
                'attention_mask': mask,
            }

        return Map(pad)

    def _operations(self) -> list[Any]:
        if self.text_field is not None and self.assistant_only:
            raise ValueError(
                'text_field and assistant_only are mutually exclusive'
            )
        ops = list(self.operations)
        if self.text_field is not None or self.assistant_only:
            if self.tokenizer is None:
                raise ValueError(
                    'text_field/assistant_only requires a tokenizer'
                )
            ops.append(self._tokenize_operation())
        if self.packing:
            if self.max_length is None:
                raise ValueError('packing requires max_length')
            ops.append(PackSequences(self.max_length, overflow='truncate'))
        elif self.max_length is not None:
            ops.append(self._pad_operation())
        ops.append(
            grain.Batch(
                self.batch_size,
                drop_remainder=self.drop_remainder,
            )
        )
        if self.packing:
            ops.append(CausalLMBatch())
        return ops

    def build(self) -> Any:
        """Return a train dataloader for the configured source."""
        if self.dataloader is not None:
            return self.dataloader

        if self.dataset is not None:
            return DatasetUtils.from_datasets(
                self.dataset,
                operations=self._operations(),
                shuffle=self.shuffle,
                seed=self.seed,
                num_epochs=self.epochs,
            )

        if self.repo_id is not None:
            from datasets import load_dataset

            dataset = load_dataset(self.repo_id, streaming=self.streaming)
            split = (
                dataset['train']
                if isinstance(dataset, Mapping)
                else dataset
            )
            if self.streaming:
                return self._streaming_iterable(split)
            return DatasetUtils.from_datasets(
                split,
                operations=self._operations(),
                shuffle=self.shuffle,
                seed=self.seed,
                num_epochs=self.epochs,
            )

        raise ValueError(
            'SFTDatasetConfig requires one of dataloader, dataset, or repo_id'
        )


class SFTTrainer(Trainer):
    """Trainer specialized for supervised fine-tuning.

    Uses :class:`SFTDatasetConfig` to build a dataloader (or pass a supplied
    one through) and defaults the loss to :func:`causal_lm_loss`.
    """

    def __init__(
        self,
        model: Any,
        training_config: TrainingConfig | None = None,
        dataset_config: SFTDatasetConfig | None = None,
        loss_fn: Any = None,
        **kwargs: Any,
    ) -> None:
        if dataset_config is None:
            raise ValueError('SFTTrainer requires an SFTDatasetConfig')
        if training_config is None:
            training_config = TrainingConfig()

        dataloader = dataset_config.build()
        train_dataset = DatasetConfig(
            dataloader,
            validation_dataloader=dataset_config.validation_dataloader,
            prefetch_size=dataset_config.prefetch_size,
            shuffle=dataset_config.shuffle,
            seed=dataset_config.seed,
        )
        if loss_fn is None:
            loss_fn = causal_lm_loss

        super().__init__(
            model,
            training_config,
            train_dataset,
            loss_fn=loss_fn,
            **kwargs,
        )


__all__ = ['SFTDatasetConfig', 'SFTTrainer']
