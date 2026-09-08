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
from numbers import Number
from typing import Any


def _numeric_logs(logs: Any) -> Any:
    return {
        name: value
        for name, value in logs.items()
        if isinstance(value, Number)
        and not isinstance(value, bool)
    }


class TrainerCallback:
    """Base class for Trainer lifecycle callbacks."""

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, logs: Any) -> None:
        pass

    def on_log(self, trainer: Any, logs: Any) -> None:
        pass

    def on_save(self, trainer: Any, checkpoint_path: str) -> None:
        pass

    def on_evaluate(self, trainer: Any, metrics: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass


class TensorBoardCallback(TrainerCallback):
    """Write Trainer logs through a TensorBoard SummaryWriter."""

    def __init__(self, log_dir: Any=None, *, writer: Any=None) -> None:
        self.log_dir = log_dir
        self.writer = writer
        self._owns_writer = writer is None

    def _get_writer(self, trainer: Any) -> Any:
        if self.writer is not None:
            return self.writer

        log_dir = self.log_dir
        if log_dir is None:
            output_dir = trainer.training_config.output_dir
            log_dir = os.path.join(
                os.fspath(output_dir) if output_dir is not None else '.',
                'runs',
            )
        try:
            from tensorboardX import SummaryWriter  # ty: ignore[unresolved-import]
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise ImportError(
                    'TensorBoardCallback requires tensorboardX or '
                    'torch with tensorboard support'
                ) from error
        self.writer = SummaryWriter(log_dir=log_dir)
        return self.writer

    def on_log(self, trainer: Any, logs: Any) -> None:
        writer = self._get_writer(trainer)
        step = int(logs.get('step', trainer.global_step))
        is_evaluation = any(
            name.startswith('eval_')
            for name in logs
        )
        namespace = 'eval' if is_evaluation else 'train'
        for name, value in _numeric_logs(logs).items():
            if name in {'step', 'epoch'}:
                continue
            if is_evaluation and name.startswith('eval_'):
                name = name.removeprefix('eval_')
            writer.add_scalar(f'{namespace}/{name}', value, step)

    def on_save(self, trainer: Any, checkpoint_path: str) -> None:
        if self.writer is not None:
            self.writer.flush()

    def on_train_end(self, trainer: Any) -> None:
        if self.writer is None:
            return
        self.writer.flush()
        if self._owns_writer:
            self.writer.close()
            self.writer = None


class WandbCallback(TrainerCallback):
    """Report Trainer logs to Weights & Biases."""

    def __init__(
        self,
        project: Any=None,
        *,
        name: str | None=None,
        config: Any=None,
        run: Any=None,
        **init_kwargs: Any,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config
        self.run = run
        self.init_kwargs = init_kwargs
        self._owns_run = run is None

    def _get_run(self) -> Any:
        if self.run is not None:
            return self.run
        try:
            import wandb  # ty: ignore[unresolved-import]
        except ImportError as error:
            raise ImportError(
                'WandbCallback requires the wandb package'
            ) from error
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            **self.init_kwargs,
        )
        return self.run

    def on_log(self, trainer: Any, logs: Any) -> None:
        run = self._get_run()
        values = _numeric_logs(logs)
        step = int(values.pop('step', trainer.global_step))
        run.log(values, step=step)

    def on_train_end(self, trainer: Any) -> None:
        if self.run is not None and self._owns_run:
            self.run.finish()
            self.run = None


__all__ = [
    'TensorBoardCallback',
    'TrainerCallback',
    'WandbCallback',
]
