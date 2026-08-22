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
"""Trainer config"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from taktiny.utils.typing import Batch, PathLike, PyTree


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int | None = None
    learning_rate: float = 1e-3
    schedule: Callable[[Any], Any] | None = None
    optimizer: Any = None  # Optax optimizer, defaults to AdamW.
    weight_decay: float = 0.0
    log_interval: int = 10
    seed: int = 42
    jit_compile: bool = True
    donate_batch: bool = False
    output_dir: str | PathLike | None = None
    save_steps: int | None = None
    save_total_limit: int | None = None
    save_at_end: bool = False
    save_optimizer_state: bool = True
    save_async: bool = False
    max_shard_size: int | str = '5GB'
    eval_strategy: str = 'no'
    eval_steps: int | None = None
    metric_for_best_model: str = 'eval_loss'
    greater_is_better: bool | None = None
    load_best_model_at_end: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = None
    compute_grad_norm: bool = True
    skip_non_finite: bool = True
    ema_decay: float | None = None
    loss_scale: float | str | None = None
    initial_loss_scale: float = 32768.0
    loss_scale_growth_interval: int = 2000

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError('seed should be an integer')

        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError('max_steps should be a positive integer')

        if self.log_interval < 1:
            raise ValueError('log_interval should be a positive integer')

        if self.schedule is not None and not callable(self.schedule):
            raise TypeError('schedule should be callable')

        if (
            self.save_steps is not None
            and (
                isinstance(self.save_steps, bool)
                or self.save_steps < 1
            )
        ):
            raise ValueError('save_steps must be a positive integer or None')
        if (
            self.save_total_limit is not None
            and (
                isinstance(self.save_total_limit, bool)
                or self.save_total_limit < 1
            )
        ):
            raise ValueError(
                'save_total_limit should be a positive integer'
            )

        if not isinstance(self.save_at_end, bool):
            raise TypeError('save_at_end should be a boolean')

        if not isinstance(self.save_optimizer_state, bool):
            raise TypeError('save_optimizer_state should be a boolean')

        if not isinstance(self.save_async, bool):
            raise TypeError('save_async should be a boolean')

        if self.eval_strategy not in {'no', 'steps', 'epoch'}:
            raise ValueError(
                'eval_strategy should be "no", "steps", or "epoch"'
            )

        if (
            self.eval_steps is not None
            and (
                isinstance(self.eval_steps, bool)
                or not isinstance(self.eval_steps, int)
                or self.eval_steps < 1
            )
        ):
            raise ValueError('eval_steps should be a positive integer')

        if self.eval_strategy == 'steps' and self.eval_steps is None:
            raise ValueError(
                'eval_steps is required when eval_strategy="steps"'
            )

        if not (
            isinstance(self.metric_for_best_model, str)
            and self.metric_for_best_model
        ):
            raise TypeError(
                'metric_for_best_model should be a non-empty string'
            )

        if (
            self.greater_is_better is not None
            and not isinstance(self.greater_is_better, bool)
        ):
            raise TypeError('greater_is_better should be a boolean')

        if not isinstance(self.load_best_model_at_end, bool):
            raise TypeError('load_best_model_at_end should be a boolean')

        if not isinstance(self.compute_grad_norm, bool):
            raise TypeError('compute_grad_norm should be a boolean')

        if self.ema_decay is not None and not (
            0.0 < self.ema_decay < 1.0
        ):
            raise ValueError(
                'ema_decay must be None or in (0, 1)'
            )

        if (
            self.load_best_model_at_end
            and self.eval_strategy == 'no'
        ):
            raise ValueError(
                'load_best_model_at_end requires evaluation'
            )

        if (
            isinstance(self.gradient_accumulation_steps, bool)
            or not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError(
                'gradient_accumulation_steps should be a positive integer'
            )

        if (
            self.max_grad_norm is not None
            and (
                isinstance(self.max_grad_norm, bool)
                or not isinstance(self.max_grad_norm, (int, float))
                or self.max_grad_norm <= 0
            )
        ):
            raise ValueError('max_grad_norm should be positive')

        if not isinstance(self.skip_non_finite, bool):
            raise TypeError('skip_non_finite should be a boolean')

        if not (
            self.loss_scale is None
            or self.loss_scale == 'dynamic'
            or (
                isinstance(self.loss_scale, (int, float))
                and not isinstance(self.loss_scale, bool)
                and self.loss_scale > 0
            )
        ):
            raise ValueError(
                'loss_scale should be "dynamic", or a positive number'
            )

        if (
            isinstance(self.initial_loss_scale, bool)
            or not isinstance(self.initial_loss_scale, (int, float))
            or self.initial_loss_scale <= 0
        ):
            raise ValueError('initial_loss_scale should be positive')

        if (
            isinstance(self.loss_scale_growth_interval, bool)
            or not isinstance(self.loss_scale_growth_interval, int)
            or self.loss_scale_growth_interval < 1
        ):
            raise ValueError(
                'loss_scale_growth_interval should be a positive integer'
            )

        if (
            self.output_dir is None
            and (
                self.save_steps is not None
                or self.save_at_end
                or self.load_best_model_at_end
            )
        ):
            raise ValueError(
                'output_dir is required when checkpoint saving is enabled'
            )

@dataclass(frozen=True)
class DatasetConfig:
    """Configure an existing dataloader or an automatic HF dataset source."""

    # A generic iterable that yields batches (e.g. Grain, PyTorch, or custom).
    # When supplied, all repo-loading options below are ignored.
    train_dataloader: Iterable[Batch] | None = None
    validation_dataloader: Iterable[Batch] | None = None
    # Sharding applied to every batch leaf, or a matching sharding PyTree.
    batch_sharding: PyTree | None = None
    shuffle: bool = True
    seed: int = 42
    prefetch_size: int = 2

    def __post_init__(self) -> None:
        if self.train_dataloader is None:
            raise TypeError('train_dataloader is required')

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError('seed should be an integer')

        if (
            isinstance(self.prefetch_size, bool)
            or not isinstance(self.prefetch_size, int)
            or self.prefetch_size < 0
        ):
            raise ValueError('prefetch_size should be a non-negative integer')


__all__ = [
    'TrainingConfig',
    'DatasetConfig',
]
