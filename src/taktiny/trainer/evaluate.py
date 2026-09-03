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

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from taktiny.utils.trainer import (
    _parameter_mesh,
    _prefetch,
    _sharding_mesh,
    _tree_shardings,
    _validate_parameter_placement,
)
from taktiny.utils.typing import PyTree


class TrainerEvaluateMixin:
    def _evaluate_params(self, params: PyTree) -> dict[str, float]:
        dataloader = self._validation_dataloader
        if dataloader is None:
            raise ValueError(
                'validation_dataloader is required for evaluation'
            )

        losses = []
        metric_values = {}
        expected_metric_names = None
        evaluation_rng = jax.random.fold_in(
            jax.random.key(self.training_config.seed),
            self.global_step,
        )
        batches = _prefetch(
            dataloader,
            self._place_batch,
            self.dataset_config.prefetch_size,
        )

        for batch in batches:
            evaluation_rng, batch_rng = jax.random.split(evaluation_rng)
            if (
                self._compiled_eval_step is None
                and self.training_config.jit_compile
            ):
                def evaluate_loss(candidate: Any, value: Any, rng: Any) -> Any:
                    if self._loss_accepts_rng:
                        return self.loss_fn(
                            candidate,
                            value,
                            rng=rng,
                        )
                    return self.loss_fn(candidate, value)

                self._compiled_eval_step = jax.jit(
                    evaluate_loss,
                    in_shardings=(
                        _tree_shardings(params),
                        _tree_shardings(batch),
                        None,
                    ),
                    out_shardings=None,
                )
            if self._compiled_eval_step is not None:
                value = self._compiled_eval_step(
                    params,
                    batch,
                    batch_rng,
                )

            elif self._loss_accepts_rng:
                value = self.loss_fn(params, batch, rng=batch_rng)

            else:
                value = self.loss_fn(params, batch)

            if isinstance(value, jax.Array):
                value = value.item()

            losses.append(float(value))
            if self.compute_metrics is not None:
                batch_metrics = self.compute_metrics(params, batch)
                if not isinstance(batch_metrics, Mapping):
                    raise TypeError(
                        'compute_metrics must return a mapping'
                    )

                batch_metric_names = set(batch_metrics)
                if expected_metric_names is None:
                    expected_metric_names = batch_metric_names

                elif batch_metric_names != expected_metric_names:
                    raise ValueError(
                        'compute_metrics must return the same metric names '
                        'for every validation batch'
                    )

                for name, metric_value in batch_metrics.items():
                    if not isinstance(name, str) or not name:
                        raise TypeError(
                            'Custom metric names must be non-empty strings'
                        )

                    metric_name = (
                        name if name.startswith('eval_') else f'eval_{name}'
                    )
                    if metric_name == 'eval_loss':
                        raise ValueError(
                            'compute_metrics cannot replace eval_loss'
                        )

                    metric_array = jnp.asarray(metric_value)
                    if metric_array.ndim != 0:
                        raise ValueError(
                            f'Custom metric {name!r} must be scalar'
                        )

                    metric_values.setdefault(metric_name, []).append(
                        float(jax.device_get(metric_array))
                    )

        batches.close()

        if not losses:
            raise ValueError(
                'validation_dataloader produced no evaluation batches'
            )

        metrics = {
            'eval_loss': sum(losses) / len(losses),
        }
        metrics.update({
            name: sum(values) / len(values)
            for name, values in metric_values.items()
        })

        return metrics


    def evaluate(self) -> dict[str, float]:
        """Evaluate the current model using ``validation_dataloader``."""
        params = self.extract_params()
        parameter_mesh = _parameter_mesh(params)
        batch_mesh = _sharding_mesh(self.dataset_config.batch_sharding)
        _validate_parameter_placement(params, batch_mesh)
        self._mesh = parameter_mesh or batch_mesh
        metrics = self._evaluate_params(params)
        record = {
            'step': self.global_step,
            **metrics,
        }
        self.log_history.append(record)
        self._call_event('on_log', logs=dict(record))
        self._call_event('on_evaluate', metrics=dict(record))
        return metrics


    def _record_evaluation(
        self,
        params: PyTree,
        *,
        step: int,
        epoch: int,
    ) -> tuple[dict[str, float], bool]:
        metrics = self._evaluate_params(params)
        record = {
            'step': step,
            **metrics,
        }
        self.log_history.append(record)

        metric_name = self.training_config.metric_for_best_model
        if not metric_name.startswith('eval_'):
            metric_name = f'eval_{metric_name}'

        if metric_name not in metrics:
            raise ValueError(
                f'Evaluation did not produce metric {metric_name!r}'
            )

        metric = float(metrics[metric_name])
        greater_is_better = self.training_config.greater_is_better
        if greater_is_better is None:
            greater_is_better = not metric_name.endswith('loss')

        is_best = (
            self.best_metric is None
            or (
                metric > self.best_metric
                if greater_is_better
                else metric < self.best_metric
            )
        )

        if is_best:
            self.best_metric = metric
            self._best_step = step
            self.best_model_checkpoint = None

        self._call_event('on_log', logs=dict(record))
        self._call_event('on_evaluate', metrics=dict(record))
        return metrics, is_best


__all__ = ['TrainerEvaluateMixin']