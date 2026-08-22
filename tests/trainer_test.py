from types import SimpleNamespace
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import taktiny
from taktiny import nn
from taktiny import Takt
from taktiny.cosettes.overture import PretrainedModel
from taktiny.peft import LoraConfig
from taktiny.trainer import (
    DatasetConfig,
    TensorBoardCallback,
    Trainer,
    TrainerCallback,
    TrainingConfig,
    WandbCallback,
)
from taktiny.trainer.trainer import (
    _format_iteration_time,
    _global_grad_norm,
    _parameter_labels,
    _partition_params,
    _place_trainable_params,
    _prefetch,
    _validate_parameter_placement,
)


class TinyModel(nn.Module):
    def __init__(self):
        self.weight = nn.Parameter(jnp.asarray(0.0))
        self.frozen = nn.Parameter(jnp.asarray(3.0), trainable=False)


class SavingTinyModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.save_calls = []

    def save_pretrained(self, path, *, max_shard_size):
        os.makedirs(path, exist_ok=True)
        saved_file = os.path.join(path, 'weight.txt')
        with open(saved_file, 'w') as checkpoint:
            checkpoint.write(str(float(self.weight.value)))
        self.save_calls.append(
            (os.fspath(path), max_shard_size, float(self.weight.value))
        )
        return (saved_file,)


class CheckpointTinyModel(PretrainedModel):
    def __init__(self):
        self.config = {}
        self.weight = nn.Parameter(jnp.asarray(0.0))
        self.frozen = nn.Parameter(jnp.asarray(3.0), trainable=False)


class SlowCheckpointTinyModel(CheckpointTinyModel):
    @classmethod
    def _save_pretrained_snapshot(cls, snapshot, path, **kwargs):
        time.sleep(0.05)
        return super()._save_pretrained_snapshot(
            snapshot,
            path,
            **kwargs,
        )


class FailingSavingTinyModel(TinyModel):
    def save_pretrained(self, path, *, max_shard_size):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'partial.txt'), 'w') as partial:
            partial.write('incomplete')
        raise RuntimeError('checkpoint write failed')


class AdapterTrainingModel(PretrainedModel):
    def __init__(self):
        self.config = {}
        self.proj = nn.Linear(
            1,
            1,
            bias=False,
            rngs=nn.Rngs(0),
        )

    def __call__(self, x):
        return self.proj(x)


class RecordingCallback(TrainerCallback):
    def __init__(self):
        self.events = []

    def on_train_begin(self, trainer):
        self.events.append(('train_begin', trainer.global_step))

    def on_step_end(self, trainer, logs):
        self.events.append(('step_end', dict(logs)))

    def on_log(self, trainer, logs):
        self.events.append(('log', dict(logs)))

    def on_save(self, trainer, checkpoint_path):
        assert os.path.isdir(checkpoint_path)
        self.events.append(('save', os.fspath(checkpoint_path)))

    def on_evaluate(self, trainer, metrics):
        self.events.append((
            'evaluate',
            dict(metrics),
            trainer.best_metric,
        ))

    def on_train_end(self, trainer):
        self.events.append(('train_end', trainer.global_step))


class FakeSummaryWriter:
    def __init__(self):
        self.scalars = []
        self.flushes = 0
        self.closed = False

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), step))

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


class FakeWandbRun:
    def __init__(self):
        self.logs = []
        self.finished = False

    def log(self, values, *, step):
        self.logs.append((dict(values), step))

    def finish(self):
        self.finished = True


class StatefulIterator:
    def __init__(self, batches, *, state_format):
        self.batches = batches
        self.state_format = state_format
        self.position = 0
        self.restored_position = None
        self.next_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.position]
        self.position += 1
        self.next_count += 1
        return batch

    def get_state(self):
        if self.state_format == 'bytes':
            return self.position.to_bytes(8, byteorder='little')
        return {'position': self.position}

    def set_state(self, state):
        if self.state_format == 'bytes':
            position = int.from_bytes(state, byteorder='little')
        else:
            position = state['position']
        self.position = position
        self.restored_position = position


class StatefulLoader:
    def __init__(self, batches, *, state_format='bytes'):
        self.batches = batches
        self.state_format = state_format
        self.iterators = []

    def __iter__(self):
        iterator = StatefulIterator(
            self.batches,
            state_format=self.state_format,
        )
        self.iterators.append(iterator)
        return iterator

    def __len__(self):
        return len(self.batches)


class EpochAwareLoader:
    def __init__(self, batches):
        self.batches = batches
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def squared_error(model, batch):
    prediction = model.weight.value * batch['x']
    return jnp.mean((prediction - batch['y']) ** 2)


def projection_error(model, batch):
    prediction = model(batch['x'])
    return jnp.mean((prediction - batch['y']) ** 2)


def absolute_error_metrics(model, batch):
    prediction = model.weight.value * batch['x']
    return {
        'mae': jnp.mean(jnp.abs(prediction - batch['y'])),
        'eval_bias': jnp.mean(prediction - batch['y']),
    }


@pytest.mark.parametrize(
    ('seconds', 'expected'),
    [
        (0.4812, '481.2 ms/it'),
        (12.7297, '12.7 s/it'),
        (90.0, '1.5 min/it'),
    ],
)
def test_iteration_time_format(seconds, expected):
    assert _format_iteration_time(seconds) == expected


@pytest.mark.parametrize('jit_compile', [False, True])
def test_trainer_updates_only_trainable_parameters(jit_compile):
    model = TinyModel()
    batches = [
        {
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }
        for _ in range(2)
    ]
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=2,
            learning_rate=0.1,
            log_interval=2,
            jit_compile=jit_compile,
        ),
        DatasetConfig(batches, prefetch_size=2),
        loss_fn=squared_error,
    )

    trainer.train()

    assert float(model.weight.value) != 0.0
    assert float(model.frozen.value) == 3.0


def test_gradient_accumulation_fused_scan_matches_eager_path():
    def run(jit_compile):
        model = TinyModel()
        batches = [
            {
                'x': np.asarray([1.0], dtype=np.float32),
                'y': np.asarray([2.0], dtype=np.float32),
            },
            {
                'x': np.asarray([3.0], dtype=np.float32),
                'y': np.asarray([1.0], dtype=np.float32),
            },
        ]
        trainer = Trainer(
            model,
            TrainingConfig(
                max_steps=1,
                learning_rate=0.1,
                log_interval=1,
                jit_compile=jit_compile,
                gradient_accumulation_steps=2,
            ),
            DatasetConfig(batches, prefetch_size=2),
            loss_fn=squared_error,
        )
        trainer.train()
        return float(model.weight.value), trainer.log_history

    eager_weight, eager_history = run(jit_compile=False)
    fused_weight, fused_history = run(jit_compile=True)

    assert eager_weight != 0.0
    assert fused_weight == pytest.approx(eager_weight)
    assert fused_history[-1]['loss'] == pytest.approx(
        eager_history[-1]['loss']
    )


def test_trainer_rejects_empty_dataloader():
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(),
        DatasetConfig([], prefetch_size=1),
        loss_fn=squared_error,
    )

    with pytest.raises(
        ValueError,
        match='dataloader produced no training batches',
    ):
        trainer.train()


def test_trainer_saves_by_step_and_rotates_checkpoints(tmp_path):
    model = SavingTinyModel()
    batches = [
        {
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }
        for _ in range(5)
    ]
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=5,
            learning_rate=0.1,
            log_interval=5,
            output_dir=tmp_path,
            save_steps=2,
            save_total_limit=2,
            save_at_end=True,
            max_shard_size='1GB',
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert [
        os.path.basename(call[0]).split('.tmp-', 1)[0]
        for call in model.save_calls
    ] == ['checkpoint-2', 'checkpoint-4', 'checkpoint-5']
    assert all(call[1] == '1GB' for call in model.save_calls)
    assert all(call[2] != 0.0 for call in model.save_calls)
    assert trainer.global_step == 5
    assert [
        os.path.basename(path)
        for path in trainer.saved_checkpoints
    ] == ['checkpoint-4', 'checkpoint-5']
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        'checkpoint-4',
        'checkpoint-5',
    ]
    final_checkpoint = tmp_path / 'checkpoint-5'
    assert (final_checkpoint / 'optimizer_state').is_dir()
    with (final_checkpoint / 'trainer_state.json').open() as state_file:
        trainer_state = json.load(state_file)
    assert trainer_state == {
        'global_step': 5,
        'epoch': 0,
        'step_in_epoch': 5,
        'log_history': trainer.log_history,
        'best_metric': None,
        'best_model_checkpoint': None,
        'gradient_accumulation_steps': 1,
        'loss_scale': 1.0,
        'loss_scale_good_steps': 0,
        'skipped_updates': 0,
        'micro_step': 5,
    }
    assert [record['step'] for record in trainer.log_history] == [5]
    assert np.isfinite(trainer.log_history[0]['loss'])
    assert trainer.log_history[0]['seconds_per_step'] >= 0
    labels = _parameter_labels(model)
    trainable_params, _ = _partition_params(model, labels)
    optimizer_target = optax.adamw(
        0.1,
        weight_decay=0.0,
    ).init(trainable_params)
    checkpointer = ocp.StandardCheckpointer()
    try:
        restored_optimizer = checkpointer.restore(
            final_checkpoint / 'optimizer_state',
            target=optimizer_target,
        )
    finally:
        checkpointer.close()
    assert int(restored_optimizer[0].count) == 5


def test_trainer_does_not_duplicate_scheduled_final_checkpoint(tmp_path):
    model = SavingTinyModel()
    batches = [
        {
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }
        for _ in range(4)
    ]
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=4,
            output_dir=tmp_path,
            save_steps=2,
            save_at_end=True,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert [
        os.path.basename(call[0]).split('.tmp-', 1)[0]
        for call in model.save_calls
    ] == ['checkpoint-2', 'checkpoint-4']
    with (
        tmp_path / 'checkpoint-4' / 'trainer_state.json'
    ).open() as state_file:
        state = json.load(state_file)
    assert state['log_history'][-1]['step'] == 4


def test_rng_state_round_trips_without_advancing_on_save(tmp_path):
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(seed=123),
        DatasetConfig([]),
        loss_fn=squared_error,
    )
    trainer.rngs()
    trainer._save_rng_state(tmp_path)
    expected = trainer.rngs()

    restored = Trainer(
        TinyModel(),
        TrainingConfig(seed=999),
        DatasetConfig([]),
        loss_fn=squared_error,
    )
    assert restored._restore_rng_state(tmp_path)

    np.testing.assert_array_equal(
        jax.random.key_data(restored.rngs()),
        jax.random.key_data(expected),
    )


def test_loss_function_can_receive_trainer_rng():
    received = {}

    def stochastic_loss(model, batch, *, rng):
        received['rng'] = rng
        return squared_error(model, batch)

    trainer = Trainer(
        TinyModel(),
        TrainingConfig(max_steps=1, jit_compile=False),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }]),
        loss_fn=stochastic_loss,
    )
    trainer.train()

    assert 'rng' in received


@pytest.mark.parametrize('jit_compile', [False, True])
def test_evaluation_uses_separate_rng_for_stochastic_loss(jit_compile):
    def stochastic_loss(model, batch, *, rng):
        multiplier = jax.random.uniform(rng, ())
        prediction = model.weight.value * batch['x'] * multiplier
        return jnp.mean((prediction - batch['y']) ** 2)

    batch = {
        'x': np.asarray([1.0], dtype=np.float32),
        'y': np.asarray([2.0], dtype=np.float32),
    }
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(jit_compile=jit_compile, seed=7),
        DatasetConfig([], validation_dataloader=[batch]),
        loss_fn=stochastic_loss,
    )
    training_key_before = jax.random.key_data(trainer.rngs.key)

    first = trainer.evaluate()
    second = trainer.evaluate()

    assert np.isfinite(first['eval_loss'])
    assert first == second
    np.testing.assert_array_equal(
        jax.random.key_data(trainer.rngs.key),
        training_key_before,
    )


def test_async_checkpoint_uses_stable_snapshot_and_publishes_atomically(
    tmp_path,
):
    model = SlowCheckpointTinyModel()
    batches = [{
        'x': np.asarray([1.0], dtype=np.float32),
        'y': np.asarray([2.0], dtype=np.float32),
    } for _ in range(2)]
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=2,
            learning_rate=0.1,
            output_dir=tmp_path,
            save_steps=1,
            save_async=True,
            save_optimizer_state=False,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert (tmp_path / 'checkpoint-1' / 'rng_state.json').is_file()
    assert (tmp_path / 'checkpoint-2' / 'rng_state.json').is_file()
    assert not list(tmp_path.glob('checkpoint-*.tmp-*'))
    first = CheckpointTinyModel().load_pretrained(
        tmp_path / 'checkpoint-1'
    )
    second = CheckpointTinyModel().load_pretrained(
        tmp_path / 'checkpoint-2'
    )
    assert float(first.weight.value) != float(second.weight.value)


def test_failed_checkpoint_never_publishes_partial_directory(tmp_path):
    trainer = Trainer(
        FailingSavingTinyModel(),
        TrainingConfig(
            max_steps=1,
            output_dir=tmp_path,
            save_steps=1,
            save_optimizer_state=False,
        ),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }]),
        loss_fn=squared_error,
    )

    with pytest.raises(RuntimeError, match='checkpoint write failed'):
        trainer.train()

    assert not (tmp_path / 'checkpoint-1').exists()
    assert not list(tmp_path.glob('checkpoint-1.tmp-*'))


def test_multihost_state_files_are_process_local(monkeypatch, tmp_path):
    monkeypatch.setattr(jax, 'process_count', lambda: 4)
    monkeypatch.setattr(jax, 'process_index', lambda: 2)
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(),
        DatasetConfig([]),
        loss_fn=squared_error,
    )

    assert trainer._rng_state_path(tmp_path).endswith(
        'rng_state-00002.json'
    )
    assert trainer._dataloader_state_paths(tmp_path) == (
        str(tmp_path / 'dataloader_state-00002.bin'),
        str(tmp_path / 'dataloader_state-00002.json'),
    )


def test_multihost_checkpoint_coordinates_publication(
    monkeypatch,
    tmp_path,
):
    class FakeCheckpointer:
        def save(self, path, item, *, force):
            assert force
            assert item
            os.makedirs(path)

        def wait_until_finished(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(jax, 'process_count', lambda: 2)
    monkeypatch.setattr(jax, 'process_index', lambda: 0)
    monkeypatch.setattr(ocp, 'StandardCheckpointer', FakeCheckpointer)
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(
            output_dir=tmp_path,
            save_optimizer_state=False,
        ),
        DatasetConfig([]),
        loss_fn=squared_error,
    )
    barriers = []
    trainer._sync_hosts = barriers.append
    temporary = tmp_path / 'checkpoint-1.tmp'
    checkpoint = tmp_path / 'checkpoint-1'
    trainer_state = trainer._trainer_state(
        step=1,
        epoch=0,
        step_in_epoch=1,
    )

    trainer._write_checkpoint_directory(
        temporary,
        checkpoint,
        model_snapshot=None,
        optimizer_state=None,
        dataloader_state=None,
        rng_state=trainer._capture_rng_state(),
        trainer_state=trainer_state,
    )

    assert checkpoint.is_dir()
    assert (checkpoint / 'model_state').is_dir()
    assert (checkpoint / 'rng_state-00000.json').is_file()
    assert (checkpoint / 'trainer_state.json').is_file()
    assert barriers == [
        'taktiny-checkpoint-open-checkpoint-1',
        'taktiny-checkpoint-model-checkpoint-1',
        'taktiny-checkpoint-data-checkpoint-1',
        'taktiny-checkpoint-close-checkpoint-1',
        'taktiny-checkpoint-publish-checkpoint-1',
    ]


def test_trainer_records_log_interval_and_final_history():
    model = TinyModel()
    batches = [
        {
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }
        for _ in range(5)
    ]
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=5,
            learning_rate=0.1,
            log_interval=2,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert [record['step'] for record in trainer.log_history] == [2, 4, 5]
    assert all('epoch' not in record for record in trainer.log_history)
    assert all(
        isinstance(record['loss'], float)
        for record in trainer.log_history
    )
    assert all(
        record['seconds_per_step'] >= 0
        for record in trainer.log_history
    )


def test_trainer_logs_rolling_average_loss():
    def supplied_loss(model, batch):
        return model.weight.value * 0.0 + batch['loss']

    losses = [1.0, 3.0, 5.0, 7.0, 9.0]
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(
            max_steps=len(losses),
            log_interval=3,
        ),
        DatasetConfig(
            [
                {'loss': np.asarray(value, dtype=np.float32)}
                for value in losses
            ],
            prefetch_size=0,
        ),
        loss_fn=supplied_loss,
    )

    trainer.train()

    assert [record['step'] for record in trainer.log_history] == [3, 5]
    assert trainer.log_history[0]['loss'] == pytest.approx(3.0)
    assert trainer.log_history[1]['loss'] == pytest.approx(7.0)


def test_default_optimizer_uses_and_logs_schedule():
    schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=0.2,
        transition_steps=2,
    )
    model = TinyModel()
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=3,
            learning_rate=10.0,
            schedule=schedule,
            log_interval=1,
        ),
        DatasetConfig([
            {
                'x': np.asarray([1.0], dtype=np.float32),
                'y': np.asarray([2.0], dtype=np.float32),
            }
            for _ in range(3)
        ], prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert [
        record['learning_rate']
        for record in trainer.log_history
    ] == pytest.approx([0.0, 0.1, 0.2])
    assert 0.0 < float(model.weight.value) < 1.0


def test_custom_optimizer_schedule_is_logged():
    schedule = optax.linear_schedule(
        init_value=0.2,
        end_value=0.0,
        transition_steps=2,
    )
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(
            max_steps=2,
            optimizer=optax.sgd(schedule),
            schedule=schedule,
            log_interval=1,
        ),
        DatasetConfig([
            {
                'x': np.asarray([1.0], dtype=np.float32),
                'y': np.asarray([2.0], dtype=np.float32),
            }
            for _ in range(2)
        ], prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert [
        record['learning_rate']
        for record in trainer.log_history
    ] == pytest.approx([0.2, 0.1])


def test_custom_optimizer_without_schedule_logs_unknown_rate():
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(
            max_steps=1,
            optimizer=optax.sgd(0.1),
            log_interval=1,
        ),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }], prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert trainer.log_history[-1]['learning_rate'] is None


def test_trainer_dispatches_callback_events_in_order(tmp_path):
    callback = RecordingCallback()
    batches = [{
        'x': np.asarray([1.0], dtype=np.float32),
        'y': np.asarray([2.0], dtype=np.float32),
    }]
    trainer = Trainer(
        CheckpointTinyModel(),
        TrainingConfig(
            max_steps=1,
            log_interval=1,
            output_dir=tmp_path,
            save_steps=1,
            eval_strategy='steps',
            eval_steps=1,
        ),
        DatasetConfig(
            batches,
            validation_dataloader=batches,
            prefetch_size=0,
        ),
        loss_fn=squared_error,
        callbacks=callback,
    )

    trainer.train()

    assert [event[0] for event in callback.events] == [
        'train_begin',
        'step_end',
        'log',
        'log',
        'evaluate',
        'save',
        'train_end',
    ]
    step_logs = callback.events[1][1]
    assert step_logs['step'] == 1
    assert step_logs['learning_rate'] == pytest.approx(1e-3)
    assert callback.events[2][1]['loss'] is not None
    assert callback.events[3][1]['eval_loss'] >= 0
    assert callback.events[4][2] == pytest.approx(
        callback.events[4][1]['eval_loss']
    )
    assert callback.events[-1] == ('train_end', 1)


def test_partial_callback_and_callback_registration():
    received = []

    class LogCallback:
        def on_log(self, trainer, logs):
            values = dict(logs)
            values['model_type'] = trainer.model_type
            received.append(values)

    callback = LogCallback()
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(max_steps=1, log_interval=1),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }], prefetch_size=0),
        loss_fn=squared_error,
    )
    assert trainer.add_callback(callback) is callback

    trainer.train()
    trainer.remove_callback(callback)

    assert len(received) == 1
    values = received[0]
    assert values['step'] == 1
    assert values['model_type'] == 'taktiny'
    assert trainer.callbacks == []


def test_custom_metrics_are_averaged_and_prefixed():
    model = TinyModel()
    batches = [
        {
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        },
        {
            'x': np.asarray([2.0], dtype=np.float32),
            'y': np.asarray([4.0], dtype=np.float32),
        },
    ]
    trainer = Trainer(
        model,
        TrainingConfig(),
        DatasetConfig(
            [],
            validation_dataloader=batches,
            prefetch_size=0,
        ),
        loss_fn=squared_error,
        compute_metrics=absolute_error_metrics,
    )

    metrics = trainer.evaluate()

    assert metrics == pytest.approx({
        'eval_loss': 10.0,
        'eval_mae': 3.0,
        'eval_bias': -3.0,
    })
    assert trainer.log_history[-1] == {
        'step': 0,
        **metrics,
    }


@pytest.mark.parametrize(
    ('compute_metrics', 'error', 'message'),
    [
        (lambda model, batch: 1.0, TypeError, 'return a mapping'),
        (
            lambda model, batch: {'values': jnp.ones((2,))},
            ValueError,
            'must be scalar',
        ),
        (
            lambda model, batch: {'loss': 1.0},
            ValueError,
            'cannot replace eval_loss',
        ),
    ],
)
def test_custom_metrics_validate_results(compute_metrics, error, message):
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(),
        DatasetConfig(
            [],
            validation_dataloader=[{
                'x': np.asarray([1.0], dtype=np.float32),
                'y': np.asarray([2.0], dtype=np.float32),
            }],
            prefetch_size=0,
        ),
        loss_fn=squared_error,
        compute_metrics=compute_metrics,
    )

    with pytest.raises(error, match=message):
        trainer.evaluate()


def test_custom_metrics_require_consistent_names():
    def inconsistent_metrics(model, batch):
        if float(batch['x'][0]) == 1.0:
            return {'first': 1.0}
        return {'second': 2.0}

    trainer = Trainer(
        TinyModel(),
        TrainingConfig(),
        DatasetConfig(
            [],
            validation_dataloader=[
                {
                    'x': np.asarray([1.0], dtype=np.float32),
                    'y': np.asarray([2.0], dtype=np.float32),
                },
                {
                    'x': np.asarray([2.0], dtype=np.float32),
                    'y': np.asarray([4.0], dtype=np.float32),
                },
            ],
            prefetch_size=0,
        ),
        loss_fn=squared_error,
        compute_metrics=inconsistent_metrics,
    )

    with pytest.raises(ValueError, match='same metric names'):
        trainer.evaluate()


def test_tensorboard_callback_reports_training_and_evaluation(tmp_path):
    writer = FakeSummaryWriter()
    callback = TensorBoardCallback(writer=writer)
    batches = [{
        'x': np.asarray([1.0], dtype=np.float32),
        'y': np.asarray([2.0], dtype=np.float32),
    }]
    trainer = Trainer(
        CheckpointTinyModel(),
        TrainingConfig(
            max_steps=1,
            log_interval=1,
            output_dir=tmp_path,
            save_steps=1,
            eval_strategy='steps',
            eval_steps=1,
        ),
        DatasetConfig(
            batches,
            validation_dataloader=batches,
            prefetch_size=0,
        ),
        loss_fn=squared_error,
        callbacks=[callback],
        compute_metrics=absolute_error_metrics,
    )

    trainer.train()

    scalar_names = {
        name
        for name, _, _ in writer.scalars
    }
    assert 'train/loss' in scalar_names
    assert 'train/learning_rate' in scalar_names
    assert 'eval/loss' in scalar_names
    assert 'eval/mae' in scalar_names
    assert writer.flushes == 2
    assert writer.closed is False


def test_tensorboard_callback_lazily_owns_writer(monkeypatch, tmp_path):
    writer = FakeSummaryWriter()
    writer_factory_calls = []

    def writer_factory(*, log_dir):
        writer_factory_calls.append(log_dir)
        return writer

    monkeypatch.setitem(
        sys.modules,
        'tensorboardX',
        SimpleNamespace(SummaryWriter=writer_factory),
    )
    callback = TensorBoardCallback(log_dir=tmp_path)
    trainer = SimpleNamespace(
        global_step=3,
        training_config=SimpleNamespace(output_dir=None),
    )

    callback.on_log(trainer, {'step': 3, 'loss': 1.5})
    callback.on_train_end(trainer)

    assert writer_factory_calls == [tmp_path]
    assert writer.closed is True


def test_wandb_callback_reports_logs():
    run = FakeWandbRun()
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(max_steps=1, log_interval=1),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }], prefetch_size=0),
        loss_fn=squared_error,
        callbacks=[WandbCallback(run=run)],
    )

    trainer.train()

    assert len(run.logs) == 1
    values, step = run.logs[0]
    assert step == 1
    assert values['loss'] >= 0
    assert values['learning_rate'] == pytest.approx(1e-3)
    assert run.finished is False


def test_wandb_callback_lazily_owns_run(monkeypatch):
    run = FakeWandbRun()
    init_calls = []

    def init(**kwargs):
        init_calls.append(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules,
        'wandb',
        SimpleNamespace(init=init),
    )
    callback = WandbCallback(
        project='project',
        name='run',
        config={'layers': 2},
        mode='offline',
    )
    trainer = SimpleNamespace(global_step=4)

    callback.on_log(trainer, {'step': 4, 'loss': 0.5})
    callback.on_train_end(trainer)

    assert init_calls == [{
        'project': 'project',
        'name': 'run',
        'config': {'layers': 2},
        'mode': 'offline',
    }]
    assert run.finished is True


def test_callback_api_is_exported_at_package_root():
    assert taktiny.TrainerCallback is TrainerCallback
    assert taktiny.TensorBoardCallback is TensorBoardCallback
    assert taktiny.WandbCallback is WandbCallback


@pytest.mark.parametrize(
    'kwargs',
    [
        {'callbacks': [object()]},
        {'compute_metrics': 1.0},
    ],
)
def test_trainer_validates_reporting_hooks(kwargs):
    with pytest.raises(TypeError):
        Trainer(
            TinyModel(),
            TrainingConfig(),
            DatasetConfig([]),
            loss_fn=squared_error,
            **kwargs,
        )


@pytest.mark.parametrize('jit_compile', [False, True])
def test_gradient_accumulation_matches_larger_batch(jit_compile):
    combined_model = TinyModel()
    Trainer(
        combined_model,
        TrainingConfig(
            max_steps=1,
            optimizer=optax.sgd(0.1),
            log_interval=1,
            jit_compile=jit_compile,
        ),
        DatasetConfig([{
            'x': np.asarray([1.0, 2.0], dtype=np.float32),
            'y': np.asarray([2.0, 4.0], dtype=np.float32),
        }], prefetch_size=0),
        loss_fn=squared_error,
    ).train()

    accumulated_model = TinyModel()
    trainer = Trainer(
        accumulated_model,
        TrainingConfig(
            max_steps=1,
            optimizer=optax.sgd(0.1),
            log_interval=1,
            jit_compile=jit_compile,
            gradient_accumulation_steps=2,
        ),
        DatasetConfig([
            {
                'x': np.asarray([1.0], dtype=np.float32),
                'y': np.asarray([2.0], dtype=np.float32),
            },
            {
                'x': np.asarray([2.0], dtype=np.float32),
                'y': np.asarray([4.0], dtype=np.float32),
            },
        ], prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert trainer.global_step == 1
    assert trainer.micro_step == 2
    assert float(accumulated_model.weight.value) == pytest.approx(
        float(combined_model.weight.value),
        rel=1e-6,
        abs=1e-6,
    )


def test_gradient_accumulation_flushes_partial_epoch_window():
    model = TinyModel()
    trainer = Trainer(
        model,
        TrainingConfig(
            optimizer=optax.sgd(0.1),
            log_interval=10,
            gradient_accumulation_steps=2,
        ),
        DatasetConfig([
            {
                'x': np.asarray([1.0], dtype=np.float32),
                'y': np.asarray([2.0], dtype=np.float32),
            }
            for _ in range(3)
        ], prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert trainer.global_step == 2
    assert trainer.micro_step == 3
    assert trainer.log_history[-1]['step'] == 2


def test_global_gradient_clipping_limits_update_norm():
    model = TinyModel()
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=1,
            optimizer=optax.sgd(1.0),
            log_interval=1,
            max_grad_norm=1.0,
        ),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([100.0], dtype=np.float32),
        }], prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert float(model.weight.value) == pytest.approx(1.0, abs=1e-5)
    assert trainer.log_history[-1]['grad_norm'] == pytest.approx(200.0)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
def test_non_finite_gradient_skips_update(bad_value):
    model = TinyModel()
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=1,
            optimizer=optax.sgd(1.0),
            log_interval=1,
        ),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([bad_value], dtype=np.float32),
        }], prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert float(model.weight.value) == 0.0
    assert trainer.global_step == 1
    assert trainer.skipped_updates == 1
    assert trainer.log_history[-1]['loss'] is None
    assert trainer.log_history[-1]['grad_norm'] is None
    assert trainer.log_history[-1]['skipped_update'] is True


def test_dynamic_loss_scaling_recovers_after_non_finite_gradient():
    model = TinyModel()
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=2,
            optimizer=optax.sgd(0.1),
            log_interval=1,
            loss_scale='dynamic',
            initial_loss_scale=8.0,
            loss_scale_growth_interval=1,
        ),
        DatasetConfig([
            {
                'x': np.asarray([1.0], dtype=np.float32),
                'y': np.asarray([np.nan], dtype=np.float32),
            },
            {
                'x': np.asarray([1.0], dtype=np.float32),
                'y': np.asarray([2.0], dtype=np.float32),
            },
        ], prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert trainer.skipped_updates == 1
    assert trainer.loss_scale == 8.0
    assert trainer.log_history[0]['loss_scale'] == 4.0
    assert trainer.log_history[1]['loss_scale'] == 8.0
    assert float(model.weight.value) != 0.0


def test_fixed_loss_scaling_updates_fp16_parameter():
    model = TinyModel()
    model.weight.value = jnp.asarray(0.0, dtype=jnp.float16)
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=1,
            optimizer=optax.sgd(0.1),
            log_interval=1,
            loss_scale=128.0,
        ),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float16),
            'y': np.asarray([2.0], dtype=np.float16),
        }], prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert np.isfinite(float(model.weight.value))
    assert float(model.weight.value) != 0.0
    assert trainer.loss_scale == 128.0


def test_trainer_can_disable_optimizer_state_saving(tmp_path):
    model = SavingTinyModel()
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=1,
            output_dir=tmp_path,
            save_steps=1,
            save_optimizer_state=False,
        ),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }]),
        loss_fn=squared_error,
    )

    trainer.train()

    checkpoint = tmp_path / 'checkpoint-1'
    assert not (checkpoint / 'optimizer_state').exists()
    assert (checkpoint / 'trainer_state.json').exists()


def test_trainer_resume_matches_uninterrupted_training(tmp_path):
    batches = [
        {
            'x': np.asarray([value], dtype=np.float32),
            'y': np.asarray([2 * value], dtype=np.float32),
        }
        for value in range(1, 5)
    ]
    control = CheckpointTinyModel()
    Trainer(
        control,
        TrainingConfig(
            max_steps=4,
            learning_rate=0.1,
            log_interval=1,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    ).train()

    first_model = CheckpointTinyModel()
    first_trainer = Trainer(
        first_model,
        TrainingConfig(
            max_steps=2,
            learning_rate=0.1,
            log_interval=1,
            output_dir=tmp_path,
            save_steps=2,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )
    first_trainer.train()

    resumed_model = CheckpointTinyModel()
    resumed_trainer = Trainer(
        resumed_model,
        TrainingConfig(
            max_steps=4,
            learning_rate=0.1,
            log_interval=1,
            output_dir=tmp_path,
            save_steps=2,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )
    resumed_trainer.train(resume_from_checkpoint='latest')

    assert resumed_trainer.global_step == 4
    assert [record['step'] for record in resumed_trainer.log_history] == [
        1,
        2,
        3,
        4,
    ]
    assert float(resumed_model.weight.value) == pytest.approx(
        float(control.weight.value),
        rel=1e-6,
        abs=1e-6,
    )
    assert float(resumed_model.frozen.value) == 3.0
    assert (tmp_path / 'checkpoint-4').is_dir()


def test_trainer_resume_preserves_accumulation_boundaries(tmp_path):
    batches = [
        {
            'x': np.asarray([value], dtype=np.float32),
            'y': np.asarray([2 * value], dtype=np.float32),
        }
        for value in range(1, 5)
    ]
    control = CheckpointTinyModel()
    Trainer(
        control,
        TrainingConfig(
            max_steps=2,
            optimizer=optax.sgd(0.1),
            log_interval=1,
            gradient_accumulation_steps=2,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    ).train()

    first_model = CheckpointTinyModel()
    Trainer(
        first_model,
        TrainingConfig(
            max_steps=1,
            optimizer=optax.sgd(0.1),
            log_interval=1,
            output_dir=tmp_path,
            save_steps=1,
            gradient_accumulation_steps=2,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    ).train()

    resumed_model = CheckpointTinyModel()
    resumed_trainer = Trainer(
        resumed_model,
        TrainingConfig(
            max_steps=2,
            optimizer=optax.sgd(0.1),
            log_interval=1,
            output_dir=tmp_path,
            save_steps=1,
            gradient_accumulation_steps=2,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )
    resumed_trainer.train(resume_from_checkpoint='latest')

    assert resumed_trainer.global_step == 2
    assert resumed_trainer.micro_step == 4
    assert float(resumed_model.weight.value) == pytest.approx(
        float(control.weight.value),
        rel=1e-6,
        abs=1e-6,
    )
    with (
        tmp_path / 'checkpoint-2' / 'trainer_state.json'
    ).open() as state_file:
        state = json.load(state_file)
    assert state['step_in_epoch'] == 4
    assert state['micro_step'] == 4
    assert state['gradient_accumulation_steps'] == 2


def test_trainer_resume_rejects_changed_accumulation_steps(tmp_path):
    batches = [{
        'x': np.asarray([1.0], dtype=np.float32),
        'y': np.asarray([2.0], dtype=np.float32),
    }]
    Trainer(
        CheckpointTinyModel(),
        TrainingConfig(
            max_steps=1,
            output_dir=tmp_path,
            save_steps=1,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    ).train()
    trainer = Trainer(
        CheckpointTinyModel(),
        TrainingConfig(
            max_steps=2,
            output_dir=tmp_path,
            save_steps=1,
            gradient_accumulation_steps=2,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )

    with pytest.raises(ValueError, match='gradient_accumulation_steps'):
        trainer.train(resume_from_checkpoint='latest')


@pytest.mark.parametrize('state_format', ['bytes', 'json'])
def test_trainer_checkpoints_and_restores_iterator_state(
    tmp_path,
    state_format,
):
    batches = [
        {
            'x': np.asarray([value], dtype=np.float32),
            'y': np.asarray([2 * value], dtype=np.float32),
        }
        for value in range(1, 5)
    ]
    control = CheckpointTinyModel()
    Trainer(
        control,
        TrainingConfig(
            max_steps=4,
            optimizer=optax.sgd(0.1),
            log_interval=1,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    ).train()

    first_loader = StatefulLoader(
        batches,
        state_format=state_format,
    )
    Trainer(
        CheckpointTinyModel(),
        TrainingConfig(
            max_steps=2,
            optimizer=optax.sgd(0.1),
            log_interval=1,
            output_dir=tmp_path,
            save_steps=2,
        ),
        DatasetConfig(first_loader, prefetch_size=3),
        loss_fn=squared_error,
    ).train()

    state_suffix = 'bin' if state_format == 'bytes' else 'json'
    state_path = (
        tmp_path
        / 'checkpoint-2'
        / f'dataloader_state.{state_suffix}'
    )
    assert state_path.is_file()
    assert first_loader.iterators[0].position == 2
    assert first_loader.iterators[0].next_count == 2

    resumed_model = CheckpointTinyModel()
    resumed_loader = StatefulLoader(
        batches,
        state_format=state_format,
    )
    resumed_trainer = Trainer(
        resumed_model,
        TrainingConfig(
            max_steps=4,
            optimizer=optax.sgd(0.1),
            log_interval=1,
            output_dir=tmp_path,
            save_steps=2,
        ),
        DatasetConfig(resumed_loader, prefetch_size=3),
        loss_fn=squared_error,
    )
    resumed_trainer.train(resume_from_checkpoint='latest')

    restored_iterator = resumed_loader.iterators[0]
    assert restored_iterator.restored_position == 2
    assert restored_iterator.next_count == 2
    assert resumed_trainer.global_step == 4
    assert float(resumed_model.weight.value) == pytest.approx(
        float(control.weight.value),
        rel=1e-6,
        abs=1e-6,
    )


def test_trainer_does_not_reshuffle_passed_dataloader():
    # The dataloader owns its own shuffling; the trainer must not call
    # set_epoch to force a reshuffle.
    batches = [{
        'x': np.asarray([1.0], dtype=np.float32),
        'y': np.asarray([2.0], dtype=np.float32),
    }]
    loader = EpochAwareLoader(batches)
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(
            optimizer=optax.sgd(0.1),
            log_interval=10,
        ),
        DatasetConfig(loader, prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert loader.epochs == []
    assert trainer.global_step == 1


def test_epoch_hook_supports_nested_sampler_and_dataset():
    class EpochTarget:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    sampler = EpochTarget()
    dataset = EpochTarget()

    assert Trainer._set_dataloader_epoch(
        SimpleNamespace(sampler=sampler),
        3,
    )
    assert Trainer._set_dataloader_epoch(
        SimpleNamespace(dataset=dataset),
        4,
    )
    assert sampler.epochs == [3]
    assert dataset.epochs == [4]
    assert not Trainer._set_dataloader_epoch([], 5)
    assert not Trainer._has_iterator_state(iter([]))


def test_trainer_resume_applies_saved_adapter_to_base_model(tmp_path):
    batches = [
        {
            'x': np.asarray([[1.0]], dtype=np.float32),
            'y': np.asarray([[2.0]], dtype=np.float32),
        },
        {
            'x': np.asarray([[2.0]], dtype=np.float32),
            'y': np.asarray([[4.0]], dtype=np.float32),
        },
    ]
    adapted_model = Takt.apply_peft(
        AdapterTrainingModel(),
        LoraConfig(
            target_modules='proj',
            rank=1,
            alpha=1,
            rngs=nn.Rngs(1),
        ),
    )
    Trainer(
        adapted_model,
        TrainingConfig(
            max_steps=1,
            learning_rate=0.1,
            log_interval=1,
            output_dir=tmp_path,
            save_steps=1,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=projection_error,
    ).train()

    resumed_model = AdapterTrainingModel()
    resumed_trainer = Trainer(
        resumed_model,
        TrainingConfig(
            max_steps=2,
            learning_rate=0.1,
            log_interval=1,
            output_dir=tmp_path,
            save_steps=1,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=projection_error,
    )
    resumed_trainer.train(resume_from_checkpoint='latest')

    assert isinstance(resumed_model.proj, nn.LoRALinear)
    assert resumed_trainer.global_step == 2
    assert [record['step'] for record in resumed_trainer.log_history] == [
        1,
        2,
    ]
    assert (tmp_path / 'checkpoint-2' / 'adapter_config.json').is_file()


def test_trainer_resume_latest_requires_checkpoint(tmp_path):
    trainer = Trainer(
        CheckpointTinyModel(),
        TrainingConfig(output_dir=tmp_path),
        DatasetConfig([]),
        loss_fn=squared_error,
    )

    with pytest.raises(FileNotFoundError, match='No checkpoint'):
        trainer.train(resume_from_checkpoint='latest')


def test_step_evaluation_loads_and_preserves_best_checkpoint(tmp_path):
    model = CheckpointTinyModel()
    batches = [{
        'x': np.asarray([1.0], dtype=np.float32),
        'y': np.asarray([1.0], dtype=np.float32),
    } for _ in range(2)]
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=2,
            optimizer=optax.sgd(1.5),
            log_interval=1,
            output_dir=tmp_path,
            save_total_limit=1,
            eval_strategy='steps',
            eval_steps=1,
            load_best_model_at_end=True,
        ),
        DatasetConfig(
            batches,
            validation_dataloader=batches,
            prefetch_size=0,
        ),
        loss_fn=squared_error,
    )

    trainer.train()

    evaluations = [
        record
        for record in trainer.log_history
        if 'eval_loss' in record
    ]
    assert [record['step'] for record in evaluations] == [1, 2]
    assert [record['eval_loss'] for record in evaluations] == pytest.approx(
        [4.0, 16.0]
    )
    assert trainer.best_metric == pytest.approx(4.0)
    assert trainer.best_model_checkpoint == str(
        tmp_path / 'checkpoint-1'
    )
    assert float(model.weight.value) == pytest.approx(3.0)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        'checkpoint-1',
    ]
    assert trainer.evaluate()['eval_loss'] == pytest.approx(4.0)


def test_epoch_evaluation_records_at_end_of_dataloader():
    model = CheckpointTinyModel()
    batches = [{
        'x': np.asarray([1.0], dtype=np.float32),
        'y': np.asarray([2.0], dtype=np.float32),
    }]
    trainer = Trainer(
        model,
        TrainingConfig(
            learning_rate=0.1,
            log_interval=10,
            eval_strategy='epoch',
        ),
        DatasetConfig(
            batches,
            validation_dataloader=batches,
            prefetch_size=0,
        ),
        loss_fn=squared_error,
    )

    trainer.train()

    evaluations = [
        record
        for record in trainer.log_history
        if 'eval_loss' in record
    ]
    assert [record['step'] for record in evaluations] == [1]
    assert all('epoch' not in record for record in evaluations)
    assert trainer.best_metric == min(
        record['eval_loss']
        for record in evaluations
    )
    assert trainer.best_model_checkpoint is None


def test_trainer_requires_validation_data_for_eval():
    trainer = Trainer(
        CheckpointTinyModel(),
        TrainingConfig(
            eval_strategy='steps',
            eval_steps=1,
        ),
        DatasetConfig([]),
        loss_fn=squared_error,
    )

    with pytest.raises(ValueError, match='validation_dataloader'):
        trainer.train()


def test_trainer_rejects_saving_for_unsupported_model(tmp_path):
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(
            output_dir=tmp_path,
            save_steps=1,
        ),
        DatasetConfig([]),
        loss_fn=squared_error,
    )

    with pytest.raises(TypeError, match='does not support save_pretrained'):
        trainer.train()


def test_prefetch_preserves_order_and_stays_bounded():
    placed = []
    batches = _prefetch(
        range(4),
        lambda value: placed.append(value) or value,
        size=2,
    )

    assert next(batches) == 0
    assert placed == [0, 1]
    assert next(batches) == 1
    assert placed == [0, 1, 2]
    assert list(batches) == [2, 3]


def test_multi_device_batches_require_pre_sharded_parameters():
    batch_mesh = SimpleNamespace(size=2)

    with pytest.raises(ValueError, match='pre-sharded model parameters'):
        _validate_parameter_placement(
            {'weight': jnp.ones((2, 2))},
            batch_mesh,
        )


def test_parameter_and_batch_meshes_must_match():
    devices = np.asarray(jax.devices())
    parameter_mesh = Mesh(devices, ('parameters',))
    batch_mesh = Mesh(devices, ('batch',))
    params = {
        'weight': jnp.asarray(
            np.ones((1,), dtype=np.float32),
            device=NamedSharding(parameter_mesh, P()),
        )
    }

    with pytest.raises(ValueError, match='same device mesh'):
        _validate_parameter_placement(params, batch_mesh)


def test_trainable_placement_preserves_existing_named_sharding():
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ('data',))
    sharding = NamedSharding(mesh, P())
    value = jnp.asarray(
        np.ones((1,), dtype=np.float32),
        device=sharding,
    )

    placed = _place_trainable_params({'weight': value}, mesh)

    assert placed['weight'] is value


def test_parameter_placement_uses_single_device_mesh():
    devices = np.asarray([jax.devices()[0]])
    mesh = Mesh(devices, ('data',))
    value = jnp.asarray(np.ones((1,), dtype=np.float32))

    placed = _place_trainable_params({'weight': value}, mesh)

    assert isinstance(placed['weight'].sharding, NamedSharding)
    assert placed['weight'].sharding.mesh == mesh


@pytest.mark.parametrize(
    'factory',
    [
        lambda: TrainingConfig(max_steps=0),
        lambda: TrainingConfig(log_interval=0),
        lambda: TrainingConfig(schedule=0.1),
        lambda: TrainingConfig(save_steps=0, output_dir='output'),
        lambda: TrainingConfig(save_total_limit=0),
        lambda: TrainingConfig(save_at_end='yes', output_dir='output'),
        lambda: TrainingConfig(save_optimizer_state='yes'),
        lambda: TrainingConfig(save_async='yes'),
        lambda: TrainingConfig(save_steps=1),
        lambda: TrainingConfig(eval_strategy='sometimes'),
        lambda: TrainingConfig(eval_strategy='steps'),
        lambda: TrainingConfig(eval_steps=0),
        lambda: TrainingConfig(
            eval_strategy='steps',
            eval_steps=1,
            load_best_model_at_end=True,
        ),
        lambda: TrainingConfig(gradient_accumulation_steps=0),
        lambda: TrainingConfig(gradient_accumulation_steps=True),
        lambda: TrainingConfig(max_grad_norm=0),
        lambda: TrainingConfig(skip_non_finite='yes'),
        lambda: TrainingConfig(loss_scale='fixed'),
        lambda: TrainingConfig(loss_scale=0),
        lambda: TrainingConfig(initial_loss_scale=0),
        lambda: TrainingConfig(loss_scale_growth_interval=0),
        lambda: DatasetConfig([], prefetch_size=-1),
    ],
)
def test_training_configuration_validation(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_dataset_config_requires_train_dataloader():
    with pytest.raises(
        TypeError,
        match='train_dataloader is required',
    ):
        DatasetConfig()


def test_global_grad_norm_matches_optax_for_float32():
    key = jax.random.key(7)
    grads = {
        'w': jax.random.normal(key, (4, 8)),
        'b': jax.random.normal(jax.random.fold_in(key, 1), (8,)),
    }
    expected = optax.tree.norm(
        jax.tree.map(lambda value: value.astype(jnp.float32), grads)
    )

    actual = _global_grad_norm(grads)

    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_global_grad_norm_matches_optax_for_bfloat16():
    key = jax.random.key(8)
    grads = {
        'w': jax.random.normal(key, (16, 16), dtype=jnp.bfloat16),
        'b': jax.random.normal(
            jax.random.fold_in(key, 1),
            (16,),
            dtype=jnp.bfloat16,
        ),
    }
    expected = optax.tree.norm(
        jax.tree.map(lambda value: value.astype(jnp.float32), grads)
    )

    actual = _global_grad_norm(grads)

    # bf16 leaves are accumulated by XLA in f32; allow bf16 rounding error.
    assert jnp.allclose(actual, expected, rtol=5e-2, atol=5e-2)


def test_global_grad_norm_handles_mixed_dtypes_and_zero_leaves():
    grads = {
        'f32': jnp.asarray([[1.0, -2.0], [3.0, -4.0]], dtype=jnp.float32),
        'bf16': jnp.asarray([2.0, 2.0], dtype=jnp.bfloat16),
        'zero': jnp.zeros((3, 3), dtype=jnp.float32),
    }
    expected = optax.tree.norm(
        jax.tree.map(lambda value: value.astype(jnp.float32), grads)
    )

    actual = _global_grad_norm(grads)

    assert jnp.allclose(actual, expected, rtol=5e-2, atol=5e-2)


def test_global_grad_norm_of_empty_tree_is_zero():
    assert float(_global_grad_norm({})) == 0.0


def test_trainer_can_skip_grad_norm_tracking():
    model = TinyModel()
    batches = [
        {
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        },
        {
            'x': np.asarray([3.0], dtype=np.float32),
            'y': np.asarray([1.0], dtype=np.float32),
        },
    ]
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=2,
            learning_rate=0.1,
            log_interval=1,
            compute_grad_norm=False,
        ),
        DatasetConfig(batches, prefetch_size=2),
        loss_fn=squared_error,
    )

    trainer.train()

    assert float(model.weight.value) != 0.0
    assert float(model.frozen.value) == 3.0
    assert all(
        record.get('grad_norm') is None
        for record in trainer.log_history
    )


def test_grad_norm_still_computed_when_clipping_enabled():
    model = TinyModel()
    batches = [
        {
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }
    ]
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=1,
            learning_rate=0.1,
            log_interval=1,
            max_grad_norm=1.0,
            compute_grad_norm=False,
        ),
        DatasetConfig(batches, prefetch_size=1),
        loss_fn=squared_error,
    )

    trainer.train()

    assert trainer.log_history[-1].get('grad_norm') is not None
    assert float(model.weight.value) != 0.0


def test_ema_update_blends_and_preserves_frozen():
    from taktiny.trainer.trainer import _ema_update

    ema = {'w': jnp.asarray(0.0), 'f': None}
    params = {'w': jnp.asarray(10.0), 'f': None}

    out = _ema_update(ema, params, decay=0.9)

    assert float(out['w']) == pytest.approx(0.9 * 0.0 + 0.1 * 10.0)
    assert out['f'] is None


def test_ema_property_returns_independent_model():
    model = TinyModel()
    batches = [
        {
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        },
        {
            'x': np.asarray([3.0], dtype=np.float32),
            'y': np.asarray([1.0], dtype=np.float32),
        },
    ]
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=2,
            learning_rate=0.1,
            log_interval=1,
            ema_decay=0.9,
        ),
        DatasetConfig(batches, prefetch_size=2),
        loss_fn=squared_error,
    )

    trainer.train()

    ema_model = trainer.ema
    assert ema_model is not model
    assert float(ema_model.weight.value) != 0.0
    assert bool(jnp.isfinite(jnp.asarray(ema_model.weight.value)))

    # The EMA copy is independent of the trained model.
    model.weight.value = jnp.asarray(999.0)
    assert float(ema_model.weight.value) != 999.0


def test_ema_disabled_property_raises():
    model = TinyModel()
    trainer = Trainer(
        model,
        TrainingConfig(max_steps=1),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }], prefetch_size=1),
        loss_fn=squared_error,
    )

    trainer.train()

    with pytest.raises(RuntimeError, match='ema_decay'):
        trainer.ema


def test_ema_saved_and_restored_in_checkpoint(tmp_path):
    def make_trainer():
        model = CheckpointTinyModel()
        batches = [
            {
                'x': np.asarray([1.0], dtype=np.float32),
                'y': np.asarray([2.0], dtype=np.float32),
            },
            {
                'x': np.asarray([3.0], dtype=np.float32),
                'y': np.asarray([1.0], dtype=np.float32),
            },
        ]
        trainer = Trainer(
            model,
            TrainingConfig(
                max_steps=2,
                learning_rate=0.1,
                log_interval=1,
                ema_decay=0.9,
                output_dir=str(tmp_path),
                save_steps=1,
            ),
            DatasetConfig(batches, prefetch_size=2),
            loss_fn=squared_error,
        )
        return trainer

    trainer = make_trainer()
    trainer.train()

    checkpoint_dir = os.path.join(str(tmp_path), 'checkpoint-2')
    ema_file = os.path.join(checkpoint_dir, 'model-ema.safetensors')
    assert os.path.isfile(ema_file)
    ema_before = float(trainer.ema.weight.value)

    # Restore the EMA into a fresh trainer (the same helper the resume path
    # uses), and verify the weights round-trip.
    fresh = Trainer(
        TinyModel(),
        TrainingConfig(
            max_steps=2,
            learning_rate=0.1,
            log_interval=1,
            ema_decay=0.9,
        ),
        DatasetConfig([], prefetch_size=1),
        loss_fn=squared_error,
    )
    fresh._load_ema(checkpoint_dir)
    ema_after = float(fresh.ema.weight.value)

    assert ema_after == pytest.approx(ema_before)


def test_ema_decay_configuration_validation():
    with pytest.raises(ValueError, match='ema_decay'):
        TrainingConfig(ema_decay=0)
    with pytest.raises(ValueError, match='ema_decay'):
        TrainingConfig(ema_decay=1.0)
    assert TrainingConfig(ema_decay=None).ema_decay is None
    assert TrainingConfig(ema_decay=0.9999).ema_decay == 0.9999


class ShardedStubModel(CheckpointTinyModel):
    def save_pretrained(self, path, *, max_shard_size):
        os.makedirs(path, exist_ok=True)
        from safetensors.numpy import save_file
        save_file(
            {'weight': np.asarray([1.0], dtype=np.float32)},
            os.path.join(path, 'model-00001-of-00002.safetensors'),
        )
        save_file(
            {'frozen': np.asarray([2.0], dtype=np.float32)},
            os.path.join(path, 'model-00002-of-00002.safetensors'),
        )
        with open(os.path.join(path, 'model.safetensors.index.json'), 'w') as f:
            json.dump({
                'weight_map': {
                    'weight': 'model-00001-of-00002.safetensors',
                    'frozen': 'model-00002-of-00002.safetensors',
                },
            }, f)


def test_ema_checkpoint_shards_with_ema_suffix(tmp_path):
    model = ShardedStubModel()
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=1,
            learning_rate=0.1,
            log_interval=1,
            ema_decay=0.9,
        ),
        DatasetConfig([{
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        }], prefetch_size=1),
        loss_fn=squared_error,
    )
    trainer.train()

    out = str(tmp_path)
    trainer._write_ema_checkpoint(out, trainer._ema_snapshot())

    assert os.path.isfile(
        os.path.join(out, 'model-00001-of-00002-ema.safetensors')
    )
    assert os.path.isfile(
        os.path.join(out, 'model-00002-of-00002-ema.safetensors')
    )
    assert os.path.isfile(
        os.path.join(out, 'model-ema.safetensors.index.json')
    )
    with open(os.path.join(out, 'model-ema.safetensors.index.json')) as f:
        index = json.load(f)
    assert all(
        'ema' in shard for shard in index['weight_map'].values()
    )


def test_trainer_cycles_dataloader_until_max_steps():
    loader = EpochAwareLoader([
        {
            'x': np.asarray([1.0], dtype=np.float32),
            'y': np.asarray([2.0], dtype=np.float32),
        },
        {
            'x': np.asarray([3.0], dtype=np.float32),
            'y': np.asarray([1.0], dtype=np.float32),
        },
    ])
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(
            max_steps=5,
            learning_rate=0.1,
            log_interval=1,
        ),
        DatasetConfig(loader, prefetch_size=0),
        loss_fn=squared_error,
    )

    trainer.train()

    assert trainer.global_step == 5
    # Two batches per cycle: steps 0-2, 2-4, then one more on the third pass.
    assert len(trainer.log_history) > 0
    assert trainer.log_history[-1]['step'] == 5
