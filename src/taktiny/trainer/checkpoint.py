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

import copy
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from taktiny.nn import Rngs
from taktiny.nn.base import Module
from taktiny.utils.trainer import _combine_params, _copy_tree
from taktiny.utils.typing import PathLike, PyTree


class TrainerCheckpointMixin:
    def _rng_state_path(checkpoint_path: str) -> str:
        if jax.process_count() == 1:
            filename = 'rng_state.json'
        else:
            filename = f'rng_state-{jax.process_index():05d}.json'
        return os.path.join(checkpoint_path, filename)


    def _capture_rng_state(self) -> dict[str, Any]:
        return {
            'impl': str(jax.random.key_impl(self.rngs.key)),
            'key_data': np.asarray(
                jax.device_get(jax.random.key_data(self.rngs.key))
            ).tolist(),
        }


    def _save_rng_state(
        self,
        checkpoint_path: str,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        if state is None:
            state = self._capture_rng_state()

        state_path = self._rng_state_path(checkpoint_path)
        with open(state_path, 'w') as state_file:
            json.dump(state, state_file, indent=2)

        return state_path


    def _restore_rng_state(self, checkpoint_path: str) -> bool:
        state_path = self._rng_state_path(checkpoint_path)
        if not os.path.isfile(state_path):
            return False

        with open(state_path) as state_file:
            state = json.load(state_file)

        impl = state.get('impl')
        key_data = state.get('key_data')
        if not isinstance(impl, str) or not isinstance(key_data, list):
            raise TypeError('Checkpoint RNG state is invalid')

        key = jax.random.wrap_key_data(
            jnp.asarray(key_data, dtype=jnp.uint32),
            impl=impl,
        )
        self.rngs = Rngs(key)
        return True


    def _has_iterator_state(iterator: Any) -> bool:
        return (
            callable(getattr(iterator, 'get_state', None))
            and callable(getattr(iterator, 'set_state', None))
        )


    def _dataloader_state_paths(checkpoint_path: str) -> tuple[str, str]:
        suffix = (
            ''
            if jax.process_count() == 1
            else f'-{jax.process_index():05d}'
        )

        return (
            os.path.join(
                checkpoint_path,
                f'dataloader_state{suffix}.bin',
            ),
            os.path.join(
                checkpoint_path,
                f'dataloader_state{suffix}.json',
            ),
        )


    def _capture_dataloader_state(self) -> tuple[str, Any] | None:
        iterator = self._active_data_iterator
        if iterator is None or not self._has_iterator_state(iterator):
            return None

        state = iterator.get_state()
        if isinstance(state, (bytes, bytearray, memoryview)):
            return ('bytes', bytes(state))

        try:
            json.dumps(state)
        except (TypeError, ValueError) as error:
            raise TypeError(
                'Dataloader iterator get_state() should return bytes or '
                'JSON-serializable data'
            ) from error

        return ('json', state)


    def _save_dataloader_state(
        self,
        checkpoint_path: str,
        snapshot: tuple[str, Any] | None = None,
    ) -> str | None:
        if snapshot is None:
            snapshot = self._capture_dataloader_state()

        if snapshot is None:
            return None

        state_format, state = snapshot
        binary_path, json_path = self._dataloader_state_paths(
            checkpoint_path
        )

        if state_format == 'bytes':
            with open(binary_path, 'wb') as state_file:
                state_file.write(state)

            if os.path.isfile(json_path):
                os.remove(json_path)

            return binary_path

        with open(json_path, 'w') as state_file:
            json.dump(state, state_file)

        if os.path.isfile(binary_path):
            os.remove(binary_path)

        return json_path


    def _restore_dataloader_state(self, iterator: Any, checkpoint_path: str) -> bool:
        binary_path, json_path = self._dataloader_state_paths(
            checkpoint_path
        )

        existing_paths = [
            path for path in (binary_path, json_path) \
                if os.path.isfile(path)
        ]

        if not existing_paths:
            return False

        if len(existing_paths) != 1:
            raise ValueError(
                'Resume checkpoint contains multiple dataloader states'
            )

        if not self._has_iterator_state(iterator):
            return False

        state_path = existing_paths[0]
        if state_path == binary_path:
            with open(state_path, 'rb') as state_file:
                state = state_file.read()

        else:
            with open(state_path) as state_file:
                state = json.load(state_file)

        iterator.set_state(state)
        return True


    def _checkpoint_directory(self, step: int) -> str:
        return os.path.join(
            os.fspath(self.training_config.output_dir),
            f'checkpoint-{step}',
        )


    def _checkpoint_paths(self) -> list[tuple[int, str]]:
        output_dir = self.training_config.output_dir
        if output_dir is None or not os.path.isdir(output_dir):
            return []

        checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
        checkpoints = []
        for entry in os.scandir(output_dir):
            match = checkpoint_pattern.fullmatch(entry.name)
            if entry.is_dir() and match is not None:
                checkpoints.append((int(match.group(1)), entry.path))

        checkpoints.sort()
        return checkpoints


    def _rotate_checkpoints(self) -> None:
        limit = self.training_config.save_total_limit
        if limit is None:
            return

        checkpoints = self._checkpoint_paths()
        available_paths = {
            checkpoint_path
            for _, checkpoint_path in checkpoints
        }
        retained = set()
        if self.best_model_checkpoint in available_paths:
            retained.add(self.best_model_checkpoint)

        remaining = max(0, limit - len(retained))
        for _, checkpoint_path in reversed(checkpoints):
            if checkpoint_path in retained:
                continue

            if remaining == 0:
                break

            retained.add(checkpoint_path)
            remaining -= 1

        for _, checkpoint_path in checkpoints:
            if checkpoint_path in retained:
                continue

            shutil.rmtree(checkpoint_path)
        self.saved_checkpoints = [
            path
            for path in self.saved_checkpoints
            if path in retained
        ]


    def _resolve_resume_checkpoint(self, checkpoint: PathLike) -> str:
        if checkpoint != 'latest':
            checkpoint_path = os.fspath(checkpoint)
            if not os.path.isdir(checkpoint_path):
                raise FileNotFoundError(
                    f'Resume checkpoint was not found: {checkpoint_path}'
                )

            return checkpoint_path

        if self.training_config.output_dir is None:
            raise ValueError(
                'output_dir is required when resuming from "latest"'
            )
        checkpoints = self._checkpoint_paths()
        if not checkpoints:
            raise FileNotFoundError(
                'No checkpoint-* directories were found in output_dir'
            )

        return checkpoints[-1][1]


    def _load_resume_state(self, checkpoint_path: str) -> dict[str, Any]:
        trainer_state_path = os.path.join(
            checkpoint_path,
            'trainer_state.json',
        )

        if not os.path.isfile(trainer_state_path):
            raise FileNotFoundError(
                f'Trainer state was not found: {trainer_state_path}'
            )

        with open(trainer_state_path) as trainer_state_file:
            state = json.load(trainer_state_file)

        for key in ('global_step', 'epoch', 'step_in_epoch'):
            value = state.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f'trainer_state.json has invalid {key}: {value!r}'
                )

        history = state.get('log_history', [])
        if not isinstance(history, list):
            raise TypeError(
                'trainer_state.json log_history must be a list'
            )

        accumulation_steps = state.get('gradient_accumulation_steps', 1)
        if accumulation_steps != (
            self.training_config.gradient_accumulation_steps
        ):
            raise ValueError(
                'Cannot resume with a different '
                'gradient_accumulation_steps value'
            )
        for key in ('loss_scale_good_steps', 'skipped_updates', 'micro_step'):
            value = state.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f'trainer_state.json has invalid {key}: {value!r}'
                )

        loss_scale = state.get('loss_scale', self._initial_loss_scale())
        if not isinstance(loss_scale, (int, float)) or loss_scale <= 0:
            raise ValueError(
                'trainer_state.json has invalid loss_scale: '
                f'{loss_scale!r}'
            )

        return state


    def _load_checkpoint_model(self, checkpoint_path: str) -> None:
        model_state_path = os.path.join(
            checkpoint_path,
            'model_state',
        )
        if os.path.isdir(model_state_path):
            import orbax.checkpoint as ocp

            if not isinstance(self.model, Module):
                raise TypeError(
                    'Distributed model-state checkpoints currently require '
                    'a Taktiny Module'
                )

            target = self.model.flat_state_dict()
            checkpointer = ocp.StandardCheckpointer()
            try:
                restored = checkpointer.restore(
                    model_state_path,
                    target=target,
                )

            finally:
                checkpointer.close()

            self.model.load_flat_state_dict(restored)
            self._load_ema(checkpoint_path)
            return

        adapter_config = os.path.join(
            checkpoint_path,
            'adapter_config.json',
        )
        has_adapter = (
            os.path.isfile(adapter_config)
            and (
                os.path.isfile(os.path.join(
                    checkpoint_path,
                    'adapter_model.safetensors',
                ))
                or os.path.isfile(os.path.join(
                    checkpoint_path,
                    'adapter_model.safetensors.index.json',
                ))
            )
        )
        has_model = (
            os.path.isfile(os.path.join(
                checkpoint_path,
                'model.safetensors',
            ))
            or os.path.isfile(os.path.join(
                checkpoint_path,
                'model.safetensors.index.json',
            ))
        )
        if has_adapter and has_model:
            raise ValueError(
                'Resume checkpoint contains both full-model and adapter '
                'weights'
            )

        if has_adapter:
            from taktiny.takt import Takt

            Takt.load_peft(
                self.model,
                checkpoint_path,
                local=True,
            )
            self._load_ema(checkpoint_path)
            return

        if not has_model:
            raise FileNotFoundError(
                'Resume checkpoint contains neither model nor adapter '
                'Safetensors'
            )

        load_pretrained = getattr(self.model, 'load_pretrained', None)
        if not callable(load_pretrained):
            raise TypeError(
                f'{type(self.model).__name__} cannot load full model '
                'checkpoints in place'
            )

        load_pretrained(checkpoint_path)
        self._load_ema(checkpoint_path)


    def _load_ema(self, checkpoint_path: str) -> None:
        """Restore the EMA tree from a checkpoint, if present and enabled."""
        if self.training_config.ema_decay is None:
            return
        if self.model_type != 'taktiny':
            raise TypeError('EMA checkpoints require a Taktiny Module')

        self._ema = _copy_tree(self.extract_params())

        single_path = os.path.join(
            checkpoint_path,
            'model-ema.safetensors',
        )
        index_path = os.path.join(
            checkpoint_path,
            'model-ema.safetensors.index.json',
        )

        from safetensors.numpy import load_file

        if os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map = json.load(f).get('weight_map', {})
            flat: dict[str, Any] = {}
            for shard in sorted(set(weight_map.values())):
                shard_path = os.path.join(checkpoint_path, shard)
                if not os.path.isfile(shard_path):
                    raise FileNotFoundError(
                        f'EMA shard not found: {shard_path}'
                    )
                flat.update(load_file(shard_path))
        elif os.path.isfile(single_path):
            flat = load_file(single_path)
        else:
            # Checkpoint predates EMA support; start the EMA from the
            # restored weights.
            return

        self._ema.load_flat_state_dict({
            name: jnp.asarray(value)
            for name, value in flat.items()
        })


    def _write_ema_checkpoint(
        self,
        temporary_path: str,
        ema_snapshot: dict[str, Any],
    ) -> None:
        """Write the EMA weights using the model's sharded checkpoint layout.

        The EMA files mirror the model's own ``model*.safetensors`` naming
        with an ``-ema`` suffix, so a sharded model produces e.g.
        ``model-00001-of-00002-ema.safetensors`` plus a
        ``model-ema.safetensors.index.json`` index. A single-shard model
        writes ``model-ema.safetensors``.
        """
        if self.model_type != 'taktiny':
            raise TypeError('EMA checkpoints require a Taktiny Module')

        staging = os.path.join(temporary_path, '_ema_staging')
        ema_model = _copy_tree(self.model)
        ema_model.load_flat_state_dict({
            name: jnp.asarray(value)
            for name, value in ema_snapshot.items()
        })
        ema_model.save_pretrained(
            staging,
            max_shard_size=self.training_config.max_shard_size,
        )

        # Move only the model weight files, renamed with an -ema suffix; the
        # config and other files are already written by the main model save.
        for name in os.listdir(staging):
            if not name.startswith('model'):
                continue
            renamed = name
            if name == 'model.safetensors':
                renamed = 'model-ema.safetensors'
            elif name.startswith('model-') and name.endswith('.safetensors'):
                renamed = name[:-len('.safetensors')] + '-ema.safetensors'
            elif name == 'model.safetensors.index.json':
                renamed = 'model-ema.safetensors.index.json'
            else:
                continue
            os.replace(
                os.path.join(staging, name),
                os.path.join(temporary_path, renamed),
            )
        shutil.rmtree(staging, ignore_errors=True)

        # Point the index's weight_map at the -ema shard filenames.
        index_path = os.path.join(
            temporary_path,
            'model-ema.safetensors.index.json',
        )
        if os.path.isfile(index_path):
            with open(index_path) as f:
                index = json.load(f)
            index['weight_map'] = {
                key: value[:-len('.safetensors')] + '-ema.safetensors'
                for key, value in index.get('weight_map', {}).items()
            }
            with open(index_path, 'w') as f:
                json.dump(index, f)


    def _write_trainer_state(
        self,
        checkpoint_path: str,
        *,
        step: int,
        epoch: int,
        step_in_epoch: int,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        if state is None:
            state = self._trainer_state(
                step=step,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
            )

        trainer_state_path = os.path.join(
            checkpoint_path,
            'trainer_state.json',
        )
        temporary_path = (
            f'{trainer_state_path}.tmp-{uuid.uuid4().hex}'
        )

        try:
            with open(temporary_path, 'w') as trainer_state_file:
                json.dump(state, trainer_state_file, indent=2)
                trainer_state_file.flush()
                os.fsync(trainer_state_file.fileno())

            os.replace(temporary_path, trainer_state_path)
        finally:
            if os.path.isfile(temporary_path):
                os.remove(temporary_path)


    def _trainer_state(
        self,
        *,
        step: int,
        epoch: int,
        step_in_epoch: int,
    ) -> dict[str, Any]:
        return {
            'global_step': step,
            'epoch': epoch,
            'step_in_epoch': step_in_epoch,
            'log_history': copy.deepcopy(self.log_history),
            'best_metric': self.best_metric,
            'best_model_checkpoint': self.best_model_checkpoint,
            'gradient_accumulation_steps': (
                self.training_config.gradient_accumulation_steps
            ),
            'loss_scale': self.loss_scale,
            'loss_scale_good_steps': self.loss_scale_good_steps,
            'skipped_updates': self.skipped_updates,
            'micro_step': self.micro_step,
        }


    def _host_snapshot(tree: PyTree) -> PyTree:
        def copy_leaf(value: Any) -> Any:
            value = jax.device_get(value)
            if isinstance(value, np.ndarray):
                return np.array(value, copy=True)

            return copy.deepcopy(value)

        return jax.tree.map(copy_leaf, tree)


    def _sync_hosts(name: str) -> None:
        if jax.process_count() <= 1:
            return

        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


    def _finalize_checkpoint(self, checkpoint_path: str) -> None:
        if jax.process_index() == 0:
            if checkpoint_path not in self.saved_checkpoints:
                self.saved_checkpoints.append(checkpoint_path)

            self._rotate_checkpoints()
            self._call_event(
                'on_save',
                checkpoint_path=checkpoint_path,
            )

        self._sync_hosts(
            f'taktiny-checkpoint-finalize-{os.path.basename(checkpoint_path)}'
        )
        if jax.process_count() > 1:
            self.saved_checkpoints = [
                path for _, path in self._checkpoint_paths()
            ]


    def _write_checkpoint_directory(
        self,
        temporary_path: str,
        checkpoint_path: str,
        *,
        model_snapshot: Any,
        optimizer_state: PyTree,
        ema_snapshot: dict[str, Any] | None = None,
        dataloader_state: tuple[str, Any] | None,
        rng_state: Mapping[str, Any],
        trainer_state: Mapping[str, Any],
    ) -> str:
        is_primary = jax.process_index() == 0
        is_multihost = jax.process_count() > 1
        barrier_name = os.path.basename(checkpoint_path)

        try:
            if is_primary:
                if os.path.exists(temporary_path):
                    shutil.rmtree(temporary_path)
                os.makedirs(temporary_path)
            self._sync_hosts(f'taktiny-checkpoint-open-{barrier_name}')

            if is_multihost:
                if not isinstance(self.model, Module):
                    raise TypeError(
                        'Multi-host checkpoints currently require a '
                        'Taktiny Module'
                    )
                import orbax.checkpoint as ocp

                model_state_path = os.path.join(
                    temporary_path,
                    'model_state',
                )
                checkpointer = ocp.StandardCheckpointer()
                try:
                    checkpointer.save(
                        model_state_path,
                        self.model.flat_state_dict(),
                        force=True,
                    )
                    checkpointer.wait_until_finished()
                finally:
                    checkpointer.close()

                if is_primary:
                    save_config = getattr(self.model, '_save_config', None)
                    if callable(save_config):
                        save_config(temporary_path)

            elif is_primary:
                if model_snapshot is None:
                    self.model.save_pretrained(
                        temporary_path,
                        max_shard_size=(
                            self.training_config.max_shard_size
                        ),
                    )

                else:
                    self.model._save_pretrained_snapshot(
                        model_snapshot,
                        temporary_path,
                        max_shard_size=(
                            self.training_config.max_shard_size
                        ),
                    )

                if ema_snapshot is not None:
                    self._write_ema_checkpoint(
                        temporary_path,
                        ema_snapshot,
                    )

            self._sync_hosts(f'taktiny-checkpoint-model-{barrier_name}')

            if dataloader_state is not None:
                self._save_dataloader_state(
                    temporary_path,
                    dataloader_state,
                )

            self._save_rng_state(temporary_path, rng_state)

            if self.training_config.save_optimizer_state:
                import orbax.checkpoint as ocp

                optimizer_path = os.path.join(
                    temporary_path,
                    'optimizer_state',
                )
                checkpointer = ocp.StandardCheckpointer()
                try:
                    checkpointer.save(
                        optimizer_path,
                        optimizer_state,
                        force=True,
                    )
                    checkpointer.wait_until_finished()
                finally:
                    checkpointer.close()

            self._sync_hosts(f'taktiny-checkpoint-data-{barrier_name}')
            if is_primary:
                self._write_trainer_state(
                    temporary_path,
                    step=trainer_state['global_step'],
                    epoch=trainer_state['epoch'],
                    step_in_epoch=trainer_state['step_in_epoch'],
                    state=trainer_state,
                )

            self._sync_hosts(f'taktiny-checkpoint-close-{barrier_name}')

            if is_primary:
                if os.path.exists(checkpoint_path):
                    raise FileExistsError(
                        f'Checkpoint already exists: {checkpoint_path}'
                    )
                os.replace(temporary_path, checkpoint_path)

            self._sync_hosts(f'taktiny-checkpoint-publish-{barrier_name}')
            return checkpoint_path
        except BaseException:
            if is_primary and os.path.isdir(temporary_path):
                shutil.rmtree(temporary_path)
            if not is_multihost:
                raise
            # Other hosts may already be waiting at a collective. Preserve the
            # original exception on the failing host rather than masking it.
            raise


    def _drain_pending_checkpoint(self) -> str | None:
        if self._pending_checkpoint is None:
            return None

        checkpoint_path, future = self._pending_checkpoint
        self._pending_checkpoint = None
        try:
            future.result()
        except BaseException:
            if self._checkpoint_executor is not None:
                self._checkpoint_executor.shutdown(wait=True)
                self._checkpoint_executor = None
            raise

        self._finalize_checkpoint(checkpoint_path)
        return checkpoint_path


    def _save_checkpoint(
        self,
        step: int,
        trainable_params: PyTree,
        frozen_params: PyTree,
        opt_state: PyTree,
        *,
        epoch: int,
        step_in_epoch: int,
    ) -> str:
        supports_checkpoint = (
            callable(getattr(self.model, 'save_pretrained', None))
            or (
                jax.process_count() > 1
                and isinstance(self.model, Module)
            )
        )
        if not supports_checkpoint:
            raise TypeError(
                f'{type(self.model).__name__} does not support '
                'save_pretrained checkpoints'
            )

        self._drain_pending_checkpoint()
        self._inject_params(
            _combine_params(trainable_params, frozen_params)
        )
        checkpoint_path = self._checkpoint_directory(step)
        if jax.process_count() > 1:
            temporary_path = f'{checkpoint_path}.tmp'
        else:
            temporary_path = (
                f'{checkpoint_path}.tmp-{uuid.uuid4().hex}'
            )

        dataloader_state = self._capture_dataloader_state()
        rng_state = self._capture_rng_state()
        trainer_state = self._trainer_state(
            step=step,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
        )

        use_async = (
            self.training_config.save_async
            and jax.process_count() == 1
        )
        if use_async:
            snapshot = getattr(
                self.model,
                '_checkpoint_snapshot',
                None,
            )
            save_snapshot = getattr(
                self.model,
                '_save_pretrained_snapshot',
                None,
            )
            if not callable(snapshot) or not callable(save_snapshot):
                raise TypeError(
                    'save_async requires a model with checkpoint snapshot '
                    'support'
                )
            model_snapshot = snapshot()
            optimizer_state = self._host_snapshot(opt_state)
            ema_snapshot = self._ema_snapshot()
            if self._checkpoint_executor is None:
                self._checkpoint_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix='taktiny-checkpoint',
                )
            future = self._checkpoint_executor.submit(
                self._write_checkpoint_directory,
                temporary_path,
                checkpoint_path,
                model_snapshot=model_snapshot,
                optimizer_state=optimizer_state,
                ema_snapshot=ema_snapshot,
                dataloader_state=dataloader_state,
                rng_state=rng_state,
                trainer_state=trainer_state,
            )
            self._pending_checkpoint = (checkpoint_path, future)
            return checkpoint_path

        self._write_checkpoint_directory(
            temporary_path,
            checkpoint_path,
            model_snapshot=None,
            optimizer_state=opt_state,
            ema_snapshot=self._ema_snapshot(),
            dataloader_state=dataloader_state,
            rng_state=rng_state,
            trainer_state=trainer_state,
        )
        self._finalize_checkpoint(checkpoint_path)
        return checkpoint_path


    def _ensure_checkpoint(
        self,
        step: int,
        trainable_params: PyTree,
        frozen_params: PyTree,
        opt_state: PyTree,
        *,
        epoch: int,
        step_in_epoch: int,
    ) -> str:
        checkpoint_path = self._checkpoint_directory(step)
        if (
            self._pending_checkpoint is not None
            and self._pending_checkpoint[0] == checkpoint_path
        ):
            self._drain_pending_checkpoint()
        if (
            checkpoint_path in self.saved_checkpoints
            and os.path.isdir(checkpoint_path)
        ):
            self._write_trainer_state(
                checkpoint_path,
                step=step,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
            )
            return checkpoint_path
        return self._save_checkpoint(
            step,
            trainable_params,
            frozen_params,
            opt_state,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
        )


__all__ = ['TrainerCheckpointMixin']