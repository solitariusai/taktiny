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

import inspect
import math
import os
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from itertools import islice
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from taktiny.nn import Rngs
from taktiny.nn.base import Module
from taktiny.trainer.checkpoint import TrainerCheckpointMixin
from taktiny.trainer.config import DatasetConfig, TrainingConfig
from taktiny.trainer.evaluate import TrainerEvaluateMixin
from taktiny.utils.trainer import (
    _accumulate_grads,
    _combine_params,
    _copy_tree,
    _ema_update,
    _format_iteration_time,
    _global_grad_norm,
    _parameter_labels,
    _parameter_mesh,
    _partition_params,
    _place_optimizer_state,
    _place_trainable_params,
    _prefetch,
    _sharding_mesh,
    _tree_shardings,
    _validate_parameter_placement,
    _zeros_like_grads,
)
from taktiny.utils.typing import Batch, LossFn, PathLike, PyTree


class Trainer(TrainerEvaluateMixin, TrainerCheckpointMixin):
    def __init__(
        self,
        model: Any,
        training_config: TrainingConfig,
        dataset_config: DatasetConfig,
        *,
        loss_fn: LossFn,
        loss_has_aux: bool = False,
        callbacks: Iterable[Any] | Any | None = None,
        compute_metrics: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.loss_has_aux = loss_has_aux
        self.training_config = training_config
        self.dataset_config = dataset_config
        self._train_dataloader = dataset_config.train_dataloader
        self._validation_dataloader = dataset_config.validation_dataloader

        self.compute_metrics = compute_metrics
        if compute_metrics is not None and not callable(compute_metrics):
            raise TypeError('compute_metrics should be callable')

        if callbacks is None:
            self.callbacks = []

        elif any(
            callable(getattr(callbacks, event, None))
            for event in (
                'on_step_end',
                'on_log',
                'on_save',
                'on_evaluate',
            )
        ):
            self.callbacks = [callbacks]

        else:
            self.callbacks = list(callbacks)

        for callback in self.callbacks:
            self._validate_callback(callback)

        self.model_type = self._diagnose_model_type(model)
        self._mesh = None
        self.global_step = 0
        self.saved_checkpoints = []
        self.log_history = []
        self.best_metric = None
        self.best_model_checkpoint = None
        self._best_step = None
        self._compiled_eval_step = None
        self.loss_scale = self._initial_loss_scale()
        self.loss_scale_good_steps = 0
        self.skipped_updates = 0
        self.micro_step = 0
        self.last_grad_norm = None
        self.last_update_skipped = False
        self._ema = None
        self._active_data_iterator = None
        self.rngs = Rngs(self.training_config.seed)
        self._loss_accepts_rng = self._callable_accepts_rng(loss_fn)
        self._checkpoint_executor = None
        self._pending_checkpoint = None


    def _callable_accepts_rng(function: Callable[..., Any]) -> bool:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return False

        parameter = signature.parameters.get('rng')
        if parameter is not None:
            return parameter.kind is not inspect.Parameter.POSITIONAL_ONLY

        return any(
            value.kind is inspect.Parameter.VAR_KEYWORD
            for value in signature.parameters.values()
        )


    def add_callback(self, callback: Any) -> Any:
        """Append a callback and return it."""
        self._validate_callback(callback)
        self.callbacks.append(callback)
        return callback


    def remove_callback(self, callback: Any) -> None:
        """Remove a previously registered callback."""
        self.callbacks.remove(callback)


    def _call_event(self, event: str, **kwargs: Any) -> None:
        for callback in tuple(self.callbacks):
            method = getattr(callback, event, None)
            if callable(method):
                method(self, **kwargs)


    def _after_optimizer_step(self, params: Any, logs: Any) -> None:
        """Run subclass bookkeeping after a successful optimizer update."""


    def _before_train_end(self) -> None:
        """Run subclass finalization before train-end callbacks."""


    def _set_dataloader_epoch(dataloader: Any, epoch: int) -> bool:
        candidates = (
            dataloader,
            getattr(dataloader, 'sampler', None),
            getattr(dataloader, 'dataset', None),
        )

        for candidate in candidates:
            set_epoch = getattr(candidate, 'set_epoch', None)
            if callable(set_epoch):
                set_epoch(epoch)
                return True

        return False


    def _validate_callback(callback: Any) -> None:
        events = (
            'on_train_begin',
            'on_step_end',
            'on_log',
            'on_save',
            'on_evaluate',
            'on_train_end',
        )

        if not any(
            callable(getattr(callback, event, None))
            for event in events
        ):
            raise TypeError(
                'Each callback should implement at least one Trainer event'
            )


    def _initial_loss_scale(self) -> float:
        loss_scale = self.training_config.loss_scale
        if loss_scale == 'dynamic':
            return float(self.training_config.initial_loss_scale)

        if loss_scale is None:
            return 1.0

        return float(loss_scale)


    def _diagnose_model_type(self, model: Any) -> str:
        # Detect Taktiny models
        if isinstance(model, Module):
            return "taktiny"

        # Detect Flax NNX models
        if hasattr(model, "__module__") and "flax.nnx" in model.__module__:
            return "nnx"

        # Detect classic Flax Linen models
        if hasattr(model, "__module__") and "flax.linen" in model.__module__:
            return "flax_linen"

        # Detect Equinox models
        if hasattr(model, "__module__") and "equinox" in model.__module__:
            return "equinox"

        return "unknown"


    def extract_params(self) -> PyTree:
        """Extract params based on the diagnosed model type."""
        if self.model_type == "taktiny":
            # Taktiny models are fully registered PyTrees
            return self.model

        elif self.model_type == "nnx":
            from flax import nnx
            _, params = nnx.split(self.model)
            return params

        elif self.model_type == "flax_linen":
            # Assume self.model is a dict of params for Flax Linen in this simplified design
            # (In reality, Flax Trainer would need model.init or params passed in)
            return self.model

        elif self.model_type == "equinox":
            import equinox as eqx
            return eqx.filter(self.model, eqx.is_array)

        else:
            raise ValueError("Unsupported model type")


    def ema(self) -> Module:
        """A fresh model holding the EMA weights, without touching ``self.model``.

        Only available when ``TrainingConfig.ema_decay`` is set. Each access
        returns an independent copy, so callers can evaluate or save it freely.
        """
        if self._ema is None:
            raise RuntimeError(
                'EMA is disabled; set TrainingConfig.ema_decay to enable it'
            )
        return _copy_tree(self._ema)


    def _ema_snapshot(self) -> dict[str, Any] | None:
        """A host copy of the EMA leaves, or ``None`` when EMA is disabled."""
        if self._ema is None:
            return None
        return {
            name: np.array(jax.device_get(value), copy=True)
            for name, value in self._ema.flat_state_dict().items()
        }


    def _setup_optimizer(self, params: PyTree) -> optax.GradientTransformation:
        """Configure an optimizer for the trainable parameter partition."""
        base_opt = self.training_config.optimizer
        if base_opt is None:
            learning_rate = (
                self.training_config.schedule
                if self.training_config.schedule is not None
                else self.training_config.learning_rate
            )
            base_opt = optax.adamw(
                learning_rate,
                weight_decay=self.training_config.weight_decay,
            )
        return base_opt


    def _learning_rate_at_step(self, step: int) -> float | None:
        """Return the rate used by a completed optimizer update."""
        schedule = self.training_config.schedule
        if schedule is None:
            if self.training_config.optimizer is not None:
                return None

            return float(self.training_config.learning_rate)

        value = schedule(max(0, step - 1))
        return float(jax.device_get(value))


    def _place_batch(self, batch: Batch) -> Batch:
        sharding = self.dataset_config.batch_sharding
        if sharding is None:
            if self._mesh is None:
                return jax.tree.map(jax.device_put, batch)

            sharding = jax.sharding.NamedSharding(
                self._mesh,
                jax.sharding.PartitionSpec(),
            )

        if isinstance(sharding, jax.sharding.Sharding):
            return jax.tree.map(
                lambda value: jax.device_put(value, sharding),
                batch,
            )

        return jax.tree.map(
            lambda value, value_sharding: jax.device_put(
                value,
                value_sharding,
            ),
            batch,
            sharding,
        )


    def train(self, resume_from_checkpoint: PathLike | None = None) -> None:
        """Train the configured model, optionally resuming a checkpoint.

        Args:
            resume_from_checkpoint: A ``checkpoint-<step>`` directory or
                ``"latest"`` to select the highest numbered checkpoint in
                ``output_dir``. Resuming restores model or adapter weights,
                optimizer state, Trainer RNG, history, and the saved epoch and
                batch position. The dataloader must reproduce the same
                per-epoch ordering so consumed batches can be skipped
                deterministically.
        """
        console = Console()
        console.print(
            f'[bold green]Starting training for a '
            f'[cyan]{self.model_type.upper()}[/cyan] model[/bold green]'
        )
        console.print(
            f'Max Steps: [bold]{self.training_config.max_steps}[/bold]'
        )

        resume_state = None
        resume_checkpoint = None
        if resume_from_checkpoint is not None:
            resume_checkpoint = self._resolve_resume_checkpoint(
                resume_from_checkpoint
            )
            resume_state = self._load_resume_state(resume_checkpoint)
            self._load_checkpoint_model(resume_checkpoint)
            self.global_step = resume_state['global_step']
            self.log_history = list(
                resume_state.get('log_history', [])
            )
            self.best_metric = resume_state.get('best_metric')
            self.best_model_checkpoint = resume_state.get(
                'best_model_checkpoint'
            )
            self.loss_scale = float(
                resume_state.get('loss_scale', self._initial_loss_scale())
            )
            self.loss_scale_good_steps = resume_state.get(
                'loss_scale_good_steps',
                0,
            )
            self.skipped_updates = resume_state.get('skipped_updates', 0)
            self.micro_step = resume_state.get('micro_step', 0)
            self._restore_rng_state(resume_checkpoint)
            if self.best_model_checkpoint is not None:
                match = re.search(
                    r'checkpoint-(\d+)$',
                    self.best_model_checkpoint,
                )
                if match is not None:
                    self._best_step = int(match.group(1))
            self.saved_checkpoints = [
                path
                for _, path in self._checkpoint_paths()
            ]
            console.print(
                f'[dim]Resuming from {resume_checkpoint} at step '
                f'{self.global_step}[/dim]'
            )

        saving_enabled = (
            self.training_config.save_steps is not None
            or self.training_config.save_at_end
            or self.training_config.load_best_model_at_end
        )
        if (
            self.training_config.eval_strategy != 'no'
            and self._validation_dataloader is None
        ):
            raise ValueError(
                'validation_dataloader is required when evaluation is enabled'
            )
        supports_checkpoint = (
            callable(getattr(self.model, 'save_pretrained', None))
            or (
                jax.process_count() > 1
                and isinstance(self.model, Module)
            )
        )
        if saving_enabled and not supports_checkpoint:
            raise TypeError(
                f'{type(self.model).__name__} does not support '
                'save_pretrained checkpoints'
            )
        if saving_enabled:
            os.makedirs(self.training_config.output_dir, exist_ok=True)
        if (
            saving_enabled
            and self.training_config.save_async
            and jax.process_count() > 1
            and jax.process_index() == 0
        ):
            console.print(
                '[dim]save_async uses coordinated synchronous writes on '
                'multi-host jobs[/dim]'
            )

        self._call_event('on_train_begin')

        # 1. Initialize Optimizer
        params = self.extract_params()
        parameter_mesh = _parameter_mesh(params)
        batch_mesh = _sharding_mesh(self.dataset_config.batch_sharding)
        _validate_parameter_placement(params, batch_mesh)
        self._mesh = parameter_mesh or batch_mesh
        labels = _parameter_labels(params)
        trainable_params, frozen_params = _partition_params(params, labels)
        del labels, params

        trainable_params = _place_trainable_params(
            trainable_params,
            self._mesh,
        )
        frozen_params = _place_trainable_params(
            frozen_params,
            self._mesh,
        )
        if self.model_type == 'taktiny':
            initial_params = _combine_params(
                trainable_params,
                frozen_params,
            )
            self._inject_params(initial_params)
            if (
                self.training_config.ema_decay is not None
                and self._ema is None
            ):
                self._ema = _copy_tree(initial_params)

        optimizer = self._setup_optimizer(trainable_params)
        opt_state = optimizer.init(trainable_params)
        opt_state = _place_optimizer_state(
            opt_state,
            self._mesh,
        )
        if resume_checkpoint is not None:
            import orbax.checkpoint as ocp

            optimizer_path = os.path.join(
                resume_checkpoint,
                'optimizer_state',
            )
            if not os.path.isdir(optimizer_path):
                raise FileNotFoundError(
                    f'Optimizer state was not found: {optimizer_path}'
                )
            checkpointer = ocp.StandardCheckpointer()
            try:
                opt_state = checkpointer.restore(
                    optimizer_path,
                    target=opt_state,
                )
            finally:
                checkpointer.close()

        # 2. Define independently compilable gradient and optimizer phases.
        def calculate_loss(
            candidate_trainable: Any,
            current_frozen: Any,
            batch: Any,
            rng: Any,
        ) -> Any:
            current_params = _combine_params(
                candidate_trainable,
                current_frozen,
            )
            if self._loss_accepts_rng:
                return self.loss_fn(
                    current_params,
                    batch,
                    rng=rng,
                )
            return self.loss_fn(current_params, batch)

        use_loss_scaling = self.training_config.loss_scale is not None

        def scaled_loss(
            candidate_trainable: Any,
            current_frozen: Any,
            batch: Any,
            current_loss_scale: Any,
            rng: Any,
        ) -> tuple[Any, ...]:
            result = calculate_loss(
                candidate_trainable,
                current_frozen,
                batch,
                rng,
            )
            if self.loss_has_aux:
                loss, metrics = result
                return loss * current_loss_scale, (loss, metrics)
            else:
                loss = result
                return loss * current_loss_scale, loss

        loss_and_grad = jax.value_and_grad(scaled_loss, has_aux=True)

        def gradient_step(
            current_trainable: Any,
            current_frozen: Any,
            batch: Any,
            current_loss_scale: Any,
            rng: Any,
        ) -> tuple[Any, ...]:
            (_, aux_data), grads = loss_and_grad(
                current_trainable,
                current_frozen,
                batch,
                current_loss_scale,
                rng,
            )
            if use_loss_scaling:
                grads = jax.tree.map(
                    lambda grad: (
                        grad / current_loss_scale.astype(grad.dtype)
                    ),
                    grads,
                )
            return aux_data, grads

        def optimizer_step(current_trainable: Any, current_opt_state: Any, grads: Any) -> tuple[Any, ...]:
            updates, new_opt_state = optimizer.update(
                grads,
                current_opt_state,
                current_trainable,
            )
            new_trainable = optax.apply_updates(
                current_trainable,
                updates,
            )
            return new_trainable, new_opt_state

        compiled_gradient_step = None
        compiled_optimizer_step = None

        # 3. Training Loop
        import time

        step = self.global_step
        should_stop = (
            self.training_config.max_steps is not None
            and step >= self.training_config.max_steps
        )
        start_time = time.time()
        steps_since_log = 0
        steps_run_this_call = 0
        microbatches_run_this_call = 0
        loss = next(
            (
                record['loss']
                for record in reversed(self.log_history)
                if 'loss' in record
            ),
            None,
        )
        loss_window = deque(
            maxlen=self.training_config.log_interval,
        )

        def moving_average_loss() -> float | None:
            finite_losses = [
                value
                for value in loss_window
                if value is not None and math.isfinite(value)
            ]
            if not finite_losses:
                return None
            return sum(finite_losses) / len(finite_losses)

        grad_norm = None
        update_skipped = False
        resume_step_in_epoch = (
            resume_state['step_in_epoch']
            if resume_state
            else 0
        )
        epoch = 0
        step_in_epoch = resume_step_in_epoch
        accumulation_steps = (
            self.training_config.gradient_accumulation_steps
        )
        accumulated_grads = None
        accumulated_loss = None
        accumulated_metrics = None
        accumulated_microbatches = 0

        # Try to guess total optimizer updates if dataloader has __len__.
        total_steps = None
        if hasattr(self._train_dataloader, '__len__'):
            try:
                dataloader_length = len(self._train_dataloader)
            except TypeError:
                dataloader_length = None
            if dataloader_length is not None:
                total_steps = math.ceil(
                    dataloader_length / accumulation_steps
                )
        if self.training_config.max_steps is not None:
            total_steps = self.training_config.max_steps

        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
        ]
        if total_steps is not None:
            progress_columns.append(
                TextColumn(
                    "[progress.percentage]{task.percentage:>3.0f}%"
                )
            )
        progress_columns.append(TimeElapsedColumn())
        if total_steps is not None:
            progress_columns.append(TimeRemainingColumn())
        progress_columns.append(
            TextColumn(
                "• [dim]Loss:[/dim] "
                "[bold white]{task.fields[loss]:.4f}[/bold white]"
            )
        )

        step_metrics = {}
        with Progress(
            *progress_columns,
            console=console,
        ) as progress:

            task_id = progress.add_task(
                "[cyan]Training...",
                total=total_steps,
                completed=step,
                loss=float(loss) if loss is not None else 0.0,
            )

            def finish_accumulation(current_epoch: Any, current_step_in_epoch: Any) -> None:
                nonlocal accumulated_grads
                nonlocal accumulated_loss
                nonlocal accumulated_metrics
                nonlocal step_metrics
                nonlocal accumulated_microbatches
                nonlocal compiled_optimizer_step
                nonlocal grad_norm
                nonlocal loss
                nonlocal opt_state
                nonlocal should_stop
                nonlocal start_time
                nonlocal step
                nonlocal steps_run_this_call
                nonlocal steps_since_log
                nonlocal trainable_params
                nonlocal update_skipped

                divisor = jnp.asarray(
                    accumulated_microbatches,
                    dtype=jnp.float32,
                )
                # Divide in each gradient's own dtype: the previous float32
                # divisor promoted every bf16 gradient to a full float32 copy.
                # Integer microbatch counts are exact in bf16, so averaging
                # stays exact for power-of-two counts and adds at most one
                # bf16 rounding step otherwise.
                averaged_grads = jax.tree.map(
                    lambda value: (
                        None
                        if value is None
                        else value / divisor.astype(value.dtype)
                    ),
                    accumulated_grads,
                    is_leaf=lambda value: value is None,
                )
                averaged_loss = accumulated_loss / divisor
                if accumulated_metrics is not None:
                    step_metrics = jax.tree.map(lambda v: v / divisor, accumulated_metrics)
                else:
                    step_metrics = {}
                # The grad norm is only computed when it is needed: to clip
                # with max_grad_norm, or to track/report it. Computing it reads
                # every gradient leaf and keeps the averaged tree alive while
                # the reduction runs, so disabling it saves that pass entirely
                # (a further ~2.7% of the gradient tree).
                track_grad_norm = (
                    self.training_config.compute_grad_norm
                    or self.training_config.max_grad_norm is not None
                )
                if track_grad_norm:
                    current_grad_norm = _global_grad_norm(averaged_grads)
                    finite = (
                        jnp.isfinite(averaged_loss)
                        & jnp.isfinite(current_grad_norm)
                    )
                else:
                    current_grad_norm = None
                    finite = jnp.isfinite(averaged_loss)

                if self.training_config.max_grad_norm is not None:
                    clip_scale = jnp.minimum(
                        jnp.asarray(1.0, dtype=jnp.float32),
                        (
                            self.training_config.max_grad_norm
                            / (current_grad_norm + 1e-6)
                        ),
                    )
                    averaged_grads = jax.tree.map(
                        lambda value: (
                            None
                            if value is None
                            else value * clip_scale.astype(value.dtype)
                        ),
                        averaged_grads,
                        is_leaf=lambda value: value is None,
                    )

                loss_value, grad_norm_value, finite_value = jax.device_get(
                    (averaged_loss, current_grad_norm, finite)
                )
                loss_value = float(loss_value)
                grad_norm_value = (
                    None
                    if grad_norm_value is None
                    else float(grad_norm_value)
                )
                finite_value = bool(finite_value)
                update_skipped = (
                    not finite_value
                    and self.training_config.skip_non_finite
                )

                if not update_skipped:
                    if (
                        compiled_optimizer_step is None
                        and self.training_config.jit_compile
                    ):
                        compiled_optimizer_step = jax.jit(
                            optimizer_step,
                            in_shardings=(
                                _tree_shardings(trainable_params),
                                _tree_shardings(opt_state),
                                _tree_shardings(averaged_grads),
                            ),
                            out_shardings=(
                                _tree_shardings(trainable_params),
                                _tree_shardings(opt_state),
                            ),
                            # Only inputs whose storage is overwritten by an
                            # output can be recycled; params and opt_state map
                            # 1:1 onto the outputs, while averaged_grads is
                            # merely read, so donating it would only produce
                            # "Some donated buffers were not usable" warnings.
                            donate_argnums=(0, 1),
                        )
                    update_fn = (
                        compiled_optimizer_step or optimizer_step
                    )
                    trainable_params, opt_state = update_fn(
                        trainable_params,
                        opt_state,
                        averaged_grads,
                    )
                    if (
                        self._ema is not None
                        and self.training_config.ema_decay is not None
                    ):
                        self._ema = _ema_update(
                            self._ema,
                            _combine_params(
                                trainable_params,
                                frozen_params,
                            ),
                            self.training_config.ema_decay,
                        )
                else:
                    self.skipped_updates += 1

                if self.training_config.loss_scale == 'dynamic':
                    if finite_value:
                        self.loss_scale_good_steps += 1
                        if (
                            self.loss_scale_good_steps
                            >= self.training_config.loss_scale_growth_interval
                        ):
                            self.loss_scale *= 2.0
                            self.loss_scale_good_steps = 0
                    else:
                        self.loss_scale = max(1.0, self.loss_scale / 2.0)
                        self.loss_scale_good_steps = 0

                step += 1
                self.global_step = step
                self.last_grad_norm = (
                    grad_norm_value
                    if grad_norm_value is not None
                    and math.isfinite(grad_norm_value)
                    else None
                )
                self.last_update_skipped = update_skipped
                loss = loss_value if math.isfinite(loss_value) else None
                loss_window.append(loss)
                smoothed_loss = moving_average_loss()
                grad_norm = self.last_grad_norm
                steps_run_this_call += 1
                steps_since_log += 1
                learning_rate = self._learning_rate_at_step(step)
                step_logs = {
                    'step': step,
                    'loss': loss,
                    'learning_rate': learning_rate,
                    'grad_norm': grad_norm,
                    'loss_scale': self.loss_scale,
                    'skipped_update': update_skipped,
                }
                for k, v in step_metrics.items():
                    step_logs[k] = float(v)
                if not update_skipped:
                    self._after_optimizer_step(
                        _combine_params(
                            trainable_params,
                            frozen_params,
                        ),
                        dict(step_logs),
                    )
                self._call_event(
                    'on_step_end',
                    logs=dict(step_logs),
                )
                progress.update(
                    task_id,
                    advance=1,
                    loss=(
                        smoothed_loss
                        if smoothed_loss is not None
                        else float('nan')
                    ),
                )

                accumulated_grads = None
                accumulated_loss = None
                accumulated_metrics = None
                accumulated_microbatches = 0

                if step % self.training_config.log_interval == 0:
                    elapsed = time.time() - start_time
                    seconds_per_step = elapsed / max(1, steps_since_log)
                    iteration_time = _format_iteration_time(
                        seconds_per_step
                    )
                    record = {
                        **step_logs,
                        'loss': smoothed_loss,
                        'seconds_per_step': seconds_per_step,
                    }
                    self.log_history.append(record)
                    self._call_event('on_log', logs=dict(record))
                    loss_text = (
                        f'{smoothed_loss:<7.4f}'
                        if smoothed_loss is not None
                        else 'non-finite'
                    )
                    learning_rate_text = (
                        f' [dim]┃ LR: {learning_rate:.3e}[/dim]'
                        if learning_rate is not None
                        else ''
                    )
                    custom_text = ""
                    for k, v in step_metrics.items():
                        custom_text += f" [dim]┃ {k}:[/dim] [yellow]{float(v):.4f}[/yellow]"
                        
                    progress.console.print(
                        f"[bold cyan]Step {step:<6}[/bold cyan] "
                        f"[dim]┃ Loss:[/dim] "
                        f"[bold white]{loss_text}[/bold white]"
                        f"{learning_rate_text}{custom_text} [dim]┃ "
                        f"{iteration_time:>11}[/dim]"
                    )
                    start_time = time.time()
                    steps_since_log = 0

                should_evaluate = (
                    self.training_config.eval_strategy == 'steps'
                    and step % self.training_config.eval_steps == 0
                )
                if should_evaluate:
                    metrics, is_best = self._record_evaluation(
                        _combine_params(
                            trainable_params,
                            frozen_params,
                        ),
                        step=step,
                        epoch=current_epoch,
                    )
                    progress.console.print(
                        f"[bold cyan]Evaluation[/bold cyan] ┃ "
                        f"[bold cyan]Step {step:<6}[/bold cyan] "
                        f"[dim]┃ Loss:[/dim] "
                        f"[bold white]{metrics['eval_loss']:.4f}"
                        f"[/bold white]"
                    )
                    if (
                        is_best
                        and self.training_config.load_best_model_at_end
                    ):
                        self.best_model_checkpoint = (
                            self._checkpoint_directory(step)
                        )
                        self._ensure_checkpoint(
                            step,
                            trainable_params,
                            frozen_params,
                            opt_state,
                            epoch=current_epoch,
                            step_in_epoch=current_step_in_epoch,
                        )

                if (
                    self.training_config.save_steps is not None
                    and step % self.training_config.save_steps == 0
                ):
                    if self._best_step == step:
                        self.best_model_checkpoint = (
                            self._checkpoint_directory(step)
                        )
                    checkpoint_path = self._ensure_checkpoint(
                        step,
                        trainable_params,
                        frozen_params,
                        opt_state,
                        epoch=current_epoch,
                        step_in_epoch=current_step_in_epoch,
                    )
                    progress.console.print(
                        f'[dim]Saved checkpoint to '
                        f'{checkpoint_path}[/dim]'
                    )

                if (
                    self.training_config.max_steps is not None
                    and step >= self.training_config.max_steps
                ):
                    should_stop = True

            while not should_stop:
                if should_stop:
                    break

                epoch_updates_run = 0
                skip_batches = resume_step_in_epoch
                dataloader = self._train_dataloader
                data_iterator = iter(dataloader)
                self._active_data_iterator = data_iterator
                restored_iterator = (
                    resume_checkpoint is not None
                    and self._restore_dataloader_state(
                        data_iterator,
                        resume_checkpoint,
                    )
                )
                if restored_iterator:
                    epoch_batches = data_iterator
                    enumerate_start = skip_batches + 1
                elif skip_batches:
                    epoch_batches = islice(
                        data_iterator,
                        skip_batches,
                        None,
                    )
                    enumerate_start = skip_batches + 1
                else:
                    epoch_batches = data_iterator
                    enumerate_start = 1
                prefetch_size = (
                    0
                    if self._has_iterator_state(data_iterator)
                    else self.dataset_config.prefetch_size
                )
                batches = _prefetch(
                    epoch_batches,
                    self._place_batch,
                    prefetch_size,
                )
                use_fused_accumulation = (
                    self.training_config.jit_compile
                    and accumulation_steps > 1
                    and not self.loss_has_aux
                )
                if use_fused_accumulation:
                    # Fused microbatch loop: all microbatches of one optimizer
                    # step run inside a single jitted ``lax.scan``. Gradient
                    # trees never round-trip through Python and accumulate in
                    # place inside the loop body, removing the per-microbatch
                    # copies of the eager path. Shape-compatible runs are
                    # scanned separately so a smaller final dataloader batch
                    # can still participate in the same optimizer update.
                    fused_cache: dict[int, Any] = {}

                    def batch_signature(batch: Any) -> tuple[Any, ...]:
                        leaves, structure = jax.tree.flatten(batch)
                        return (
                            structure,
                            tuple(
                                (
                                    tuple(value.shape),
                                    value.dtype,
                                )
                                for value in leaves
                            ),
                        )

                    def compatible_runs(
                        chunk: list[Any],
                    ) -> list[list[Any]]:
                        runs: list[list[Any]] = []
                        signature = None
                        for batch in chunk:
                            current_signature = batch_signature(batch)
                            if signature != current_signature:
                                runs.append([])
                                signature = current_signature
                            runs[-1].append(batch)
                        return runs

                    def make_fused_step(num_batches: int) -> Any:
                        def fused_step(
                            init_grads: Any,
                            current_trainable: Any,
                            current_frozen: Any,
                            stacked_batch: Any,
                            keys: Any,
                            current_loss_scale: Any,
                        ) -> tuple[Any, Any]:
                            def accumulate_body(
                                carry: Any,
                                xs: Any,
                            ) -> tuple[Any, Any]:
                                acc_grads, acc_loss = carry
                                batch, key = xs
                                (_, loss), grads = loss_and_grad(
                                    current_trainable,
                                    current_frozen,
                                    batch,
                                    current_loss_scale,
                                    key,
                                )
                                if use_loss_scaling:
                                    grads = jax.tree.map(
                                        lambda grad: (
                                            grad
                                            / current_loss_scale.astype(
                                                grad.dtype
                                            )
                                        ),
                                        grads,
                                    )
                                return (
                                    _accumulate_grads(acc_grads, grads),
                                    acc_loss + loss.astype(jnp.float32),
                                ), None

                            (acc_grads, acc_loss), _ = jax.lax.scan(
                                accumulate_body,
                                (
                                    init_grads,
                                    jnp.asarray(
                                        0.0,
                                        dtype=jnp.float32,
                                    ),
                                ),
                                (stacked_batch, keys),
                                length=num_batches,
                            )
                            return acc_grads, acc_loss

                        return fused_step

                    def get_fused_step(num_batches: int) -> Any:
                        compiled = fused_cache.get(num_batches)
                        if compiled is not None:
                            return compiled
                        compiled = jax.jit(
                            make_fused_step(num_batches),
                            in_shardings=(
                                _tree_shardings(trainable_params),
                                _tree_shardings(trainable_params),
                                _tree_shardings(frozen_params),
                                None,
                                None,
                                None,
                            ),
                            out_shardings=(
                                _tree_shardings(trainable_params),
                                None,
                            ),
                            # The accumulated-grads accumulator is overwritten
                            # by the output; batch tensors are only read, so
                            # donating them can never recycle their storage.
                            donate_argnums=(0,),
                        )
                        fused_cache[num_batches] = compiled
                        return compiled

                    step_in_epoch = enumerate_start - 1
                    while True:
                        if should_stop:
                            break
                        chunk = list(
                            islice(batches, accumulation_steps)
                        )
                        if not chunk:
                            break
                        num_batches = len(chunk)
                        accumulated_grads = _zeros_like_grads(
                            trainable_params
                        )
                        accumulated_loss = jnp.asarray(
                            0.0,
                            dtype=jnp.float32,
                        )
                        for run in compatible_runs(chunk):
                            run_size = len(run)
                            stacked_batch = jax.tree.map(
                                lambda *values: jnp.stack(values),
                                *run,
                            )
                            keys = jnp.stack([
                                jax.random.fold_in(
                                    self.rngs(),
                                    jax.process_index(),
                                )
                                for _ in range(run_size)
                            ])
                            accumulated_grads, run_loss = (
                                get_fused_step(run_size)(
                                    accumulated_grads,
                                    trainable_params,
                                    frozen_params,
                                    stacked_batch,
                                    keys,
                                    jnp.asarray(
                                        self.loss_scale,
                                        dtype=jnp.float32,
                                    ),
                                )
                            )
                            accumulated_loss = (
                                accumulated_loss + run_loss
                            )
                        accumulated_microbatches = num_batches
                        step_in_epoch += num_batches
                        self.micro_step += num_batches
                        microbatches_run_this_call += num_batches

                        if accumulated_microbatches == accumulation_steps:
                            finish_accumulation(epoch, step_in_epoch)
                            epoch_updates_run += 1
                    batches.close()

                else:
                    for step_in_epoch, batch in enumerate(
                        batches,
                        start=enumerate_start,
                    ):
                        if (
                            compiled_gradient_step is None
                            and self.training_config.jit_compile
                        ):
                            compiled_gradient_step = jax.jit(
                                gradient_step,
                                in_shardings=(
                                    _tree_shardings(trainable_params),
                                    _tree_shardings(frozen_params),
                                    _tree_shardings(batch),
                                    None,
                                    None,
                                ),
                                out_shardings=(
                                    None,
                                    _tree_shardings(trainable_params),
                                ),
                                # Batch tensors are only read by the forward
                                # pass, so no donated batch storage could
                                # ever be recycled by an output.
                                donate_argnums=(),
                            )
                        current_gradient_step = (
                            compiled_gradient_step or gradient_step
                        )
                        microbatch_aux, microbatch_grads = (
                            current_gradient_step(
                                trainable_params,
                                frozen_params,
                                batch,
                                jnp.asarray(
                                    self.loss_scale,
                                    dtype=jnp.float32,
                                ),
                                jax.random.fold_in(
                                    self.rngs(),
                                    jax.process_index(),
                                ),
                            )
                        )
                        if self.loss_has_aux:
                            microbatch_loss, microbatch_metrics = microbatch_aux
                        else:
                            microbatch_loss = microbatch_aux
                            microbatch_metrics = {}

                        if accumulated_grads is None:
                            accumulated_grads = microbatch_grads
                            accumulated_loss = microbatch_loss.astype(jnp.float32)
                            accumulated_metrics = jax.tree.map(lambda x: x.astype(jnp.float32), microbatch_metrics)
                        else:
                            accumulated_grads = _accumulate_grads(
                                accumulated_grads,
                                microbatch_grads,
                            )
                            accumulated_loss = (
                                accumulated_loss
                                + microbatch_loss.astype(jnp.float32)
                            )
                            if accumulated_metrics is not None:
                                accumulated_metrics = jax.tree.map(
                                    lambda a, b: a + b.astype(jnp.float32), 
                                    accumulated_metrics, 
                                    microbatch_metrics
                                )
                        accumulated_microbatches += 1
                        self.micro_step += 1
                        microbatches_run_this_call += 1

                        if accumulated_microbatches == accumulation_steps:
                            finish_accumulation(epoch, step_in_epoch)
                            epoch_updates_run += 1
                        if should_stop:
                            break
                    batches.close()

                if accumulated_microbatches and not should_stop:
                    finish_accumulation(epoch, step_in_epoch)
                    epoch_updates_run += 1

                if (
                    self.training_config.eval_strategy == 'epoch'
                    and epoch_updates_run > 0
                ):
                    metrics, is_best = self._record_evaluation(
                        _combine_params(
                            trainable_params,
                            frozen_params,
                        ),
                        step=step,
                        epoch=epoch,
                    )
                    progress.console.print(
                        f"[bold cyan]Evaluation[/bold cyan] ┃ "
                        f"[dim]┃ Loss:[/dim] "
                        f"[bold white]{metrics['eval_loss']:.4f}"
                        f"[/bold white]"
                    )
                    if (
                        is_best
                        and self.training_config.load_best_model_at_end
                    ):
                        self.best_model_checkpoint = (
                            self._checkpoint_directory(step)
                        )
                        self._ensure_checkpoint(
                            step,
                            trainable_params,
                            frozen_params,
                            opt_state,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                        )
                resume_step_in_epoch = 0

                # With no max_steps the dataloader is consumed once; otherwise
                # cycle it (the dataloader owns its own shuffling) until
                # max_steps is reached.
                if self.training_config.max_steps is None:
                    break
                if epoch_updates_run == 0:
                    # The dataloader yielded nothing this pass (e.g. a
                    # one-shot iterator that can no longer be re-iterated).
                    break
                if should_stop:
                    # max_steps was reached on this pass; do not start another.
                    break

            if microbatches_run_this_call == 0 and step == 0:
                raise ValueError('dataloader produced no training batches')

            has_current_training_log = any(
                record.get('step') == step and 'loss' in record
                for record in reversed(self.log_history)
            )
            if steps_run_this_call > 0 and not has_current_training_log:
                seconds_per_step = (
                    (time.time() - start_time)
                    / max(1, steps_since_log)
                )
                learning_rate = self._learning_rate_at_step(step)
                smoothed_loss = moving_average_loss()
                self.log_history.append({
                    'step': step,
                    'loss': smoothed_loss,
                    'seconds_per_step': seconds_per_step,
                    'learning_rate': learning_rate,
                    'grad_norm': grad_norm,
                    'loss_scale': self.loss_scale,
                    'skipped_update': update_skipped,
                })
                self._call_event(
                    'on_log',
                    logs=dict(self.log_history[-1]),
                )
                loss_text = (
                    f'{smoothed_loss:<7.4f}'
                    if smoothed_loss is not None
                    else 'non-finite'
                )
                learning_rate_text = (
                    f' [dim]┃ LR: {learning_rate:.3e}[/dim]'
                    if learning_rate is not None
                    else ''
                )
                progress.console.print(
                    f"[bold cyan]Step {step:<6}[/bold cyan] "
                    f"[dim]┃ Loss:[/dim] "
                    f"[bold white]{loss_text}[/bold white]"
                    f"{learning_rate_text} [dim]┃ "
                    f"{_format_iteration_time(seconds_per_step):>11}"
                    f"[/dim]"
                )
                if saving_enabled:
                    final_checkpoint_path = self._checkpoint_directory(step)
                    if (
                        self._pending_checkpoint is not None
                        and self._pending_checkpoint[0]
                        == final_checkpoint_path
                    ):
                        self._drain_pending_checkpoint()
                    if final_checkpoint_path in self.saved_checkpoints:
                        self._write_trainer_state(
                            final_checkpoint_path,
                            step=step,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                        )

            progress.update(
                task_id,
                completed=step,
                loss=float(loss) if loss is not None else float('nan'),
            )

        # 4. Inject back into the object if needed
        params = _combine_params(trainable_params, frozen_params)
        self._inject_params(params)
        if (
            self.training_config.save_at_end
            and steps_run_this_call > 0
            and (
                self.training_config.save_steps is None
                or step % self.training_config.save_steps != 0
            )
        ):
            if self._best_step == step:
                self.best_model_checkpoint = self._checkpoint_directory(step)
            checkpoint_path = self._ensure_checkpoint(
                step,
                trainable_params,
                frozen_params,
                opt_state,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
            )
            console.print(
                f'[dim]Saved final checkpoint to {checkpoint_path}[/dim]'
            )
        self._drain_pending_checkpoint()
        if self._checkpoint_executor is not None:
            self._checkpoint_executor.shutdown(wait=True)
            self._checkpoint_executor = None
        if self.training_config.load_best_model_at_end:
            if self.best_model_checkpoint is None:
                raise ValueError(
                    'No best checkpoint was produced during evaluation'
                )
            self._load_checkpoint_model(self.best_model_checkpoint)
            console.print(
                f'[dim]Loaded best checkpoint from '
                f'{self.best_model_checkpoint}[/dim]'
            )
        self._before_train_end()
        self._call_event('on_train_end')
        console.print("[bold green]✨ Training complete![/bold green]")


    def _inject_params(self, params: PyTree) -> None:
        if self.model_type == "taktiny":
            # The returned PyTree is a new taktiny Module. We can update self.model in-place.
            self.model.load_state_dict(params.state_dict())
        elif self.model_type == "nnx":
            from flax import nnx
            # params is the state dict, we merge it back into the graph
            nnx.update(self.model, params)
        elif self.model_type == "flax_linen":
            self.params = params
        elif self.model_type == "equinox":
            self.model = params


__all__ = ['Trainer']