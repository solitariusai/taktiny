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

import base64
import copy
import json
import os
import re
import shutil
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from taktiny.nn import Rngs
from taktiny.utils.trainer import _combine_params, _copy_tree

if TYPE_CHECKING:
    from taktiny.trainer.config import TrainingConfig


def _leaves(tree: Any) -> dict[str, Any]:
    """Serialize dynamic leaves only; the caller supplies the model structure."""
    values = {str(i): value for i, value in enumerate(jax.tree.leaves(tree))}
    # Orbax rejects zero-leaf items, but optimizers such as plain SGD have no
    # array state. Preserve their empty structure with an explicit sentinel.
    return values or {'_empty': np.asarray(0, dtype=np.uint8)}


def _restore_tree(path: str, target: Any) -> Any:
    leaves, structure = jax.tree.flatten(target)
    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(path, target=_leaves(target))
    if set(restored) != set(_leaves(target)):
        raise ValueError('Checkpoint does not match the target tree')
    return jax.tree.unflatten(structure, [restored[str(i)] for i in range(len(leaves))])


class TrainerCheckpointMixin:
    """Native Orbax checkpoints; model construction remains the caller's job.

    Each checkpoint contains model_state, optional optimizer_state and ema_state,
    plus Orbax JSON trainer metadata including per-process RNG/iterator state.
    Static model configuration is not serialized. Restore with the same model
    architecture, optimizer, parameter ordering, and data pipeline. Legacy
    safetensors Trainer checkpoints are intentionally not supported.
    """

    training_config: TrainingConfig
    rngs: Rngs
    _active_data_iterator: Any
    log_history: list[Any]
    best_metric: Any
    best_model_checkpoint: str | None
    loss_scale: float
    loss_scale_good_steps: int
    skipped_updates: int
    micro_step: int
    _ema: Any
    saved_checkpoints: list[str]
    _pending_checkpoint: tuple[str, ocp.AsyncCheckpointer] | None

    if TYPE_CHECKING:
        def extract_params(self) -> Any: ...

        def _inject_params(self, params: Any) -> None: ...

        def _call_event(self, event: str, **kwargs: Any) -> None: ...

    @staticmethod
    def _has_iterator_state(iterator: Any) -> bool:
        return (callable(getattr(iterator, 'get_state', None))
                and callable(getattr(iterator, 'set_state', None)))

    def _capture_rng_state(self) -> dict[str, Any]:
        return {
            'impl': str(jax.random.key_impl(self.rngs.key)),
            'key_data': np.asarray(jax.device_get(jax.random.key_data(self.rngs.key))).tolist(),
        }

    def _capture_dataloader_state(self) -> dict[str, Any] | None:
        iterator = self._active_data_iterator
        if iterator is None or not self._has_iterator_state(iterator):
            return None
        state = iterator.get_state()
        if isinstance(state, (bytes, bytearray, memoryview)):
            return {'format': 'bytes', 'value': base64.b64encode(bytes(state)).decode('ascii')}
        # Do not pickle arbitrary user objects. JSON or bytes are explicit contracts.
        return {'format': 'json', 'value': json.loads(json.dumps(state))}

    def _runtime_states(self) -> list[dict[str, Any]]:
        state = {'rng': self._capture_rng_state(), 'data': self._capture_dataloader_state()}
        if jax.process_count() == 1:
            return [state]
        from jax.experimental import multihost_utils
        encoded = np.frombuffer(json.dumps(state).encode('utf-8'), dtype=np.uint8)
        sizes = np.asarray(multihost_utils.process_allgather(np.asarray(len(encoded), np.int32))).reshape(-1)
        padded = np.pad(encoded, (0, int(sizes.max()) - len(encoded)))
        gathered = np.asarray(multihost_utils.process_allgather(padded, tiled=False))
        return [json.loads(row[:int(size)].tobytes()) for row, size in zip(gathered, sizes)]

    def _trainer_state(self, *, step: int, epoch: int, step_in_epoch: int) -> dict[str, Any]:
        return {
            'format_version': 1,
            'global_step': step, 'epoch': epoch, 'step_in_epoch': step_in_epoch,
            'log_history': copy.deepcopy(self.log_history),
            'best_metric': self.best_metric,
            'best_model_checkpoint': self.best_model_checkpoint,
            'gradient_accumulation_steps': self.training_config.gradient_accumulation_steps,
            'loss_scale': self.loss_scale,
            'loss_scale_good_steps': self.loss_scale_good_steps,
            'skipped_updates': self.skipped_updates, 'micro_step': self.micro_step,
            'process_count': jax.process_count(),
            'runtime': self._runtime_states(),
        }

    def _load_resume_state(self, checkpoint_path: str) -> dict[str, Any]:
        with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as checkpointer:
            state = checkpointer.restore(os.path.join(checkpoint_path, 'trainer_state'))
        if state.get('format_version') != 1:
            raise ValueError('Unsupported Trainer checkpoint format')
        for name in ('global_step', 'epoch', 'step_in_epoch', 'micro_step',
                     'loss_scale_good_steps', 'skipped_updates'):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'Invalid checkpoint {name}')
        if state['gradient_accumulation_steps'] != self.training_config.gradient_accumulation_steps:
            raise ValueError('Cannot resume with a different gradient_accumulation_steps value')
        if state['process_count'] != jax.process_count():
            raise ValueError('Exact resume requires the same process_count')
        return state

    def _restore_rng_state(self, checkpoint_path: str) -> bool:
        state = self._load_resume_state(checkpoint_path)['runtime'][jax.process_index()]['rng']
        self.rngs = Rngs(jax.random.wrap_key_data(jnp.asarray(state['key_data'], jnp.uint32),
                                               impl=state['impl']))
        return True

    def _restore_dataloader_state(self, iterator: Any, checkpoint_path: str) -> bool:
        state = self._load_resume_state(checkpoint_path)['runtime'][jax.process_index()]['data']
        if state is None:
            return False
        if not self._has_iterator_state(iterator):
            raise TypeError('Checkpoint requires a dataloader iterator with set_state()')
        value = state['value']
        if state['format'] == 'bytes':
            value = base64.b64decode(value)
        iterator.set_state(value)
        return True

    def _load_checkpoint_model(self, checkpoint_path: str) -> None:
        restored = _restore_tree(os.path.join(checkpoint_path, 'model_state'), self.extract_params())
        self._inject_params(restored)
        if self.training_config.ema_decay is not None:
            self._ema = _copy_tree(self.extract_params())
            ema_path = os.path.join(checkpoint_path, 'ema_state')
            if os.path.isdir(ema_path):
                self._ema = _restore_tree(ema_path, self._ema)

    def _restore_optimizer_state(self, checkpoint_path: str, target: Any) -> Any:
        path = os.path.join(checkpoint_path, 'optimizer_state')
        if not os.path.isdir(path):
            raise FileNotFoundError('Optimizer state was not saved; exact training resume is unavailable')
        return _restore_tree(path, target)

    def _checkpoint_directory(self, step: int) -> str:
        directory = self.training_config.output_dir
        if directory is None:
            raise ValueError('output_dir is required to save checkpoints')
        return os.path.abspath(os.path.join(os.fspath(directory), f'checkpoint-{step}'))

    def _checkpoint_paths(self) -> list[tuple[int, str]]:
        directory = self.training_config.output_dir
        if directory is None or not os.path.isdir(directory):
            return []
        paths = []
        for entry in os.scandir(directory):
            match = re.fullmatch(r'checkpoint-(\d+)', entry.name)
            if (match and entry.is_dir() and not entry.is_symlink()
                    and os.path.isfile(os.path.join(entry.path, '_CHECKPOINT_METADATA'))):
                paths.append((int(match.group(1)), os.path.abspath(entry.path)))
        return sorted(paths)

    def _resolve_resume_checkpoint(self, checkpoint: Any) -> str:
        self._drain_pending_checkpoint()
        if checkpoint != 'latest':
            path = os.path.abspath(os.fspath(checkpoint))
            if not os.path.isdir(path):
                raise FileNotFoundError(f'Resume checkpoint was not found: {path}')
            return path
        paths = self._checkpoint_paths()
        if not paths:
            raise FileNotFoundError('No completed Orbax checkpoints were found in output_dir')
        return paths[-1][1]

    def _rotate_checkpoints(self) -> None:
        limit = self.training_config.save_total_limit
        if limit is None:
            return
        paths = self._checkpoint_paths()
        retained = {self.best_model_checkpoint} if self.best_model_checkpoint else set()
        # Keep latest resumable state as well as best weights, even if limit=1.
        if paths:
            retained.add(paths[-1][1])
        for _, path in reversed(paths):
            if len(retained) >= limit:
                break
            retained.add(path)
        if jax.process_index() == 0:
            for _, path in paths:
                if path not in retained:
                    shutil.rmtree(path)
        self.saved_checkpoints = [path for _, path in paths if path in retained]

    def _finalize_checkpoint(self, checkpoint_path: str) -> None:
        if checkpoint_path not in self.saved_checkpoints:
            self.saved_checkpoints.append(checkpoint_path)
        self._rotate_checkpoints()
        if jax.process_index() == 0:
            self._call_event('on_save', checkpoint_path=checkpoint_path)

    def _drain_pending_checkpoint(self) -> str | None:
        pending = self._pending_checkpoint
        if pending is None:
            return None
        self._pending_checkpoint = None
        try:
            pending[1].wait_until_finished()
        finally:
            pending[1].close()
        self._finalize_checkpoint(pending[0])
        return pending[0]

    def _save_checkpoint(self, step: int, trainable_params: Any, frozen_params: Any,
                         opt_state: Any, *, epoch: int, step_in_epoch: int) -> str:
        self._drain_pending_checkpoint()
        self._inject_params(_combine_params(trainable_params, frozen_params))
        path = self._checkpoint_directory(step)
        if os.path.exists(path):
            raise FileExistsError(f'Checkpoint already exists: {path}')
        state = self._trainer_state(step=step, epoch=epoch, step_in_epoch=step_in_epoch)
        items = {
            'model_state': ocp.args.StandardSave(_leaves(self.extract_params())),
            'trainer_state': ocp.args.JsonSave(state),
        }
        registry = ocp.DefaultCheckpointHandlerRegistry()
        registry.add('model_state', ocp.args.StandardSave, ocp.StandardCheckpointHandler())
        registry.add('trainer_state', ocp.args.JsonSave, ocp.JsonCheckpointHandler())
        if self.training_config.save_optimizer_state:
            items['optimizer_state'] = ocp.args.StandardSave(_leaves(opt_state))
            registry.add('optimizer_state', ocp.args.StandardSave, ocp.StandardCheckpointHandler())
        if self._ema is not None:
            items['ema_state'] = ocp.args.StandardSave(_leaves(self._ema))
            registry.add('ema_state', ocp.args.StandardSave, ocp.StandardCheckpointHandler())
        handler = ocp.CompositeCheckpointHandler(handler_registry=registry)
        async_checkpointer = (
            ocp.AsyncCheckpointer(handler) if self.training_config.save_async else None
        )
        checkpointer = async_checkpointer or ocp.Checkpointer(handler)
        try:
            # Orbax owns atomic publication and async host snapshots. Never overwrite.
            checkpointer.save(path, args=ocp.args.Composite(**items))
        except BaseException:
            checkpointer.close()
            raise
        if async_checkpointer is not None:
            self._pending_checkpoint = (path, async_checkpointer)
        else:
            checkpointer.close()
            self._finalize_checkpoint(path)
        return path

    def _write_trainer_state(self, checkpoint_path: str, *, step: int,
                             epoch: int, step_in_epoch: int) -> None:
        self._drain_pending_checkpoint()
        # Update bookkeeping only; keep the original iterator/RNG snapshot paired
        # with the stored model, even after the loop has consumed a StopIteration.
        state = self._load_resume_state(checkpoint_path)
        state.update(log_history=copy.deepcopy(self.log_history),
                     best_metric=self.best_metric,
                     best_model_checkpoint=self.best_model_checkpoint)
        with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as checkpointer:
            checkpointer.save(os.path.join(checkpoint_path, 'trainer_state'),
                              args=ocp.args.JsonSave(state), force=True)

    def _ensure_checkpoint(self, step: int, trainable_params: Any, frozen_params: Any,
                           opt_state: Any, *, epoch: int, step_in_epoch: int) -> str:
        path = self._checkpoint_directory(step)
        if self._pending_checkpoint is not None and self._pending_checkpoint[0] == path:
            self._drain_pending_checkpoint()
        if path in self.saved_checkpoints and os.path.isdir(path):
            self._write_trainer_state(path, step=step, epoch=epoch, step_in_epoch=step_in_epoch)
            return path
        return self._save_checkpoint(step, trainable_params, frozen_params, opt_state,
                                     epoch=epoch, step_in_epoch=step_in_epoch)


__all__ = ['TrainerCheckpointMixin']
