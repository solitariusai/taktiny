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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
from itertools import islice
import inspect
import json
import math
import os
import re
import shutil
import uuid
from typing import Any, TypeVar

import jax
import numpy as np
import qwix
from taktiny.trainer.config import TrainingConfig, DatasetConfig
from taktiny.nn import Rngs
from taktiny.nn.base import Module, Parameter
from taktiny.utils.typing import Batch, LossFn, PathLike, PyTree

import jax.numpy as jnp
import optax

T = TypeVar('T')

def _is_trainable_value(value: Any) -> bool:
    if isinstance(value, qwix.QArray):
        return False

    if not hasattr(value, 'dtype'):
        return False

    if not jnp.issubdtype(value.dtype, jnp.inexact):
        return False

    return value.dtype != getattr(jnp, 'float8_e4m3fn', None)


def _parameter_labels(params: PyTree) -> PyTree:
    """Build Optax labels from explicit Taktiny parameter metadata."""
    if not isinstance(params, Module):
        return jax.tree.map(
            lambda value: (
                'trainable'
                if _is_trainable_value(value)
                else 'frozen'
            ),
            params,
        )

    def label_parameter(parameter: Parameter) -> PyTree:
        trainable = (
            getattr(parameter, 'trainable', True)
            and _is_trainable_value(parameter.value)
        )
        label = 'trainable' if trainable else 'frozen'
        return jax.tree.map(lambda _: label, parameter)

    return jax.tree.map(
        label_parameter,
        params,
        is_leaf=lambda value: isinstance(value, Parameter),
    )


def _partition_params(params: PyTree, labels: PyTree) -> tuple[PyTree, PyTree]:
    def label_value(label: PyTree) -> PyTree:
        return label.value if isinstance(label, Parameter) else label

    def empty_value(value: PyTree) -> PyTree:
        if isinstance(value, Parameter):
            empty = object.__new__(type(value))
            empty.__dict__.update(value.__dict__)
            empty.value = None
            return empty
        return None

    trainable = jax.tree.map(
        lambda value, label: (
            value
            if label_value(label) == 'trainable'
            else empty_value(value)
        ),
        params,
        labels,
        is_leaf=lambda value: isinstance(value, Parameter),
    )
    frozen = jax.tree.map(
        lambda value, label: (
            empty_value(value)
            if label_value(label) == 'trainable'
            else value
        ),
        params,
        labels,
        is_leaf=lambda value: isinstance(value, Parameter),
    )
    return trainable, frozen


def _combine_params(trainable: PyTree, frozen: PyTree) -> PyTree:
    def combine_value(
        trainable_value: PyTree,
        frozen_value: PyTree,
    ) -> PyTree:
        if isinstance(trainable_value, Parameter):
            if trainable_value.value is None:
                return frozen_value
            return trainable_value
        return frozen_value if trainable_value is None else trainable_value

    return jax.tree.map(
        combine_value,
        trainable,
        frozen,
        is_leaf=lambda value: value is None or isinstance(value, Parameter),
    )


def _global_grad_norm(grads: PyTree) -> jax.Array:
    """Compute the L2 norm of a gradient tree without full-size copies.

    Each leaf is contracted with itself through ``jnp.vdot``, which XLA lowers
    to a fused dot with float32 accumulation; only a per-leaf scalar is
    produced. No squared or cast copy of the gradient tree is ever allocated,
    so the peak memory of the norm is negligible even for huge models.
    Frozen leaves (``None``) are skipped.
    """
    norm_sq = jax.tree.reduce(
        lambda acc, grad: (
            acc
            if grad is None
            else acc + jnp.vdot(grad, grad).astype(jnp.float32)
        ),
        grads,
        initializer=jnp.asarray(0.0, dtype=jnp.float32),
        is_leaf=lambda value: value is None,
    )
    return jnp.sqrt(norm_sq)


def _zeros_like_grads(params: PyTree) -> PyTree:
    """Build a zero gradient accumulator matching a trainable parameter tree.

    Frozen (``None``) leaves are preserved so the accumulator mirrors the
    gradient tree returned by ``jax.value_and_grad``.
    """
    return jax.tree.map(
        lambda value: (
            None if value is None else jnp.zeros_like(value)
        ),
        params,
        is_leaf=lambda value: value is None,
    )


def _accumulate_grads(total: PyTree, value: PyTree) -> PyTree:
    """Sum two gradient trees, preserving frozen (``None``) leaves."""
    return jax.tree.map(
        lambda a, b: (
            None if a is None or b is None else a + b
        ),
        total,
        value,
        is_leaf=lambda value: value is None,
    )


def _copy_tree(tree: PyTree) -> PyTree:
    """Return a fresh structure with independent array leaves."""
    return jax.tree.map(
        lambda value: (
            value.copy()
            if hasattr(value, 'dtype')
            else copy.deepcopy(value)
        ),
        tree,
    )


def _ema_update(
    ema: PyTree,
    params: PyTree,
    decay: float,
) -> PyTree:
    """Blend the EMA tree toward the current weights, in the weights' dtype.

    Frozen (``None``) leaves are passed through unchanged.
    """
    def blend(ema_value: Any, param_value: Any) -> Any:
        if ema_value is None or param_value is None:
            return param_value
        return ema_value * decay + param_value * (1.0 - decay)

    return jax.tree.map(
        blend,
        ema,
        params,
        is_leaf=lambda value: value is None,
    )


def _tree_shardings(tree: PyTree) -> PyTree:
    return jax.tree.map(
        lambda value: (
            value.sharding if isinstance(value, jax.Array) else None
        ),
        tree,
    )


def _parameter_mesh(params: PyTree) -> jax.sharding.Mesh | None:
    for value in jax.tree.leaves(params):
        sharding = getattr(value, 'sharding', None)
        if isinstance(sharding, jax.sharding.NamedSharding):
            return sharding.mesh
    return None


def _sharding_mesh(sharding: PyTree) -> jax.sharding.Mesh | None:
    for value in jax.tree.leaves(sharding):
        if isinstance(value, jax.sharding.NamedSharding):
            return value.mesh
    return None


def _uses_multiple_devices(tree: PyTree) -> bool:
    for value in jax.tree.leaves(tree):
        sharding = getattr(value, 'sharding', None)
        if sharding is not None and len(sharding.device_set) > 1:
            return True
    return False


def _validate_parameter_placement(
    params: PyTree,
    batch_mesh: jax.sharding.Mesh | None,
) -> None:
    parameter_mesh = _parameter_mesh(params)
    if (
        parameter_mesh is not None
        and batch_mesh is not None
        and parameter_mesh != batch_mesh
    ):
        raise ValueError(
            'Model parameters and batches must use the same device mesh.'
        )
    if batch_mesh is None or batch_mesh.size <= 1:
        return
    if parameter_mesh is not None or _uses_multiple_devices(params):
        return
    raise ValueError(
        'Multi-device batch sharding requires pre-sharded model parameters. '
        'Load the model with the same mesh and parameter sharding rules; '
        'Trainer will not replicate an unsharded model automatically.'
    )


def _place_trainable_params(
    tree: PyTree,
    mesh: jax.sharding.Mesh | None,
) -> PyTree:
    if mesh is None:
        return tree

    replicated = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )

    def place(value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        if isinstance(value.sharding, jax.sharding.NamedSharding):
            return value
        return jax.device_put(value, replicated)

    return jax.tree.map(place, tree)


def _place_optimizer_state(
    tree: PyTree,
    mesh: jax.sharding.Mesh | None,
) -> PyTree:
    if mesh is None or mesh.size <= 1:
        return tree

    replicated = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )

    def place(value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        if isinstance(value.sharding, jax.sharding.NamedSharding):
            return value
        if value.ndim == 0:
            return jax.device_put(value, replicated)
        raise ValueError(
            'A non-scalar optimizer state did not inherit parameter sharding. '
            'Provide an optimizer whose parameter-shaped state preserves '
            'input shardings.'
        )

    return jax.tree.map(place, tree)


def _prefetch(
    iterable: Iterable[T],
    place: Callable[[T], T],
    size: int,
) -> Iterator[T]:
    """Place a bounded number of batches ahead of consumption."""
    iterator = iter(iterable)
    if size == 0:
        for value in iterator:
            yield place(value)
        return

    queue = deque()
    for _ in range(size):
        try:
            queue.append(place(next(iterator)))
        except StopIteration:
            break

    while queue:
        yield queue.popleft()
        try:
            queue.append(place(next(iterator)))
        except StopIteration:
            pass


def _format_iteration_time(seconds: float) -> str:
    if seconds < 1:
        return f'{seconds * 1000:.1f} ms/it'

    if seconds < 60:
        return f'{seconds:.1f} s/it'

    return f'{seconds / 60:.1f} min/it'


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


class Trainer:
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

    @staticmethod
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

    @staticmethod
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
            raise ValueError('Checkpoint RNG state is invalid')

        key = jax.random.wrap_key_data(
            jnp.asarray(key_data, dtype=jnp.uint32),
            impl=impl,
        )
        self.rngs = Rngs(key)
        return True

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

    @staticmethod
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

    @staticmethod
    def _has_iterator_state(iterator: Any) -> bool:
        return (
            callable(getattr(iterator, 'get_state', None))
            and callable(getattr(iterator, 'set_state', None))
        )

    @staticmethod
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

    @staticmethod
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

    @property
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
            raise ValueError(
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

    @staticmethod
    def _host_snapshot(tree: PyTree) -> PyTree:
        def copy_leaf(value: Any) -> Any:
            value = jax.device_get(value)
            if isinstance(value, np.ndarray):
                return np.array(value, copy=True)

            return copy.deepcopy(value)

        return jax.tree.map(copy_leaf, tree)

    @staticmethod
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
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

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
