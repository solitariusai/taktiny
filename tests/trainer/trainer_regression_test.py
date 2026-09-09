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
        TinyModel(),
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
    assert taktiny.trainer.TrainerCallback is TrainerCallback
    assert taktiny.trainer.TensorBoardCallback is TensorBoardCallback
    assert taktiny.trainer.WandbCallback is WandbCallback



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



def test_jitted_accumulation_accepts_smaller_final_batch():
    batches = [
        {
            'x': np.asarray(values, dtype=np.float32),
            'y': np.asarray(values, dtype=np.float32) * 2,
        }
        for values in (
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0],
        )
    ]

    eager_model = TinyModel()
    eager_trainer = Trainer(
        eager_model,
        TrainingConfig(
            optimizer=optax.sgd(0.1),
            gradient_accumulation_steps=4,
            jit_compile=False,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )
    eager_trainer.train()

    fused_model = TinyModel()
    fused_trainer = Trainer(
        fused_model,
        TrainingConfig(
            optimizer=optax.sgd(0.1),
            gradient_accumulation_steps=4,
            jit_compile=True,
        ),
        DatasetConfig(batches, prefetch_size=0),
        loss_fn=squared_error,
    )
    fused_trainer.train()

    assert fused_trainer.global_step == 1
    assert fused_trainer.micro_step == 4
    assert float(fused_model.weight.value) == pytest.approx(
        float(eager_model.weight.value),
        rel=1e-6,
        abs=1e-6,
    )



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
    model.weight = nn.Parameter(jnp.asarray(0.0, dtype=jnp.float16))
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



def test_trainer_resume_latest_requires_checkpoint(tmp_path):
    trainer = Trainer(
        TinyModel(),
        TrainingConfig(output_dir=tmp_path),
        DatasetConfig([]),
        loss_fn=squared_error,
    )

    with pytest.raises(FileNotFoundError, match='No completed Orbax checkpoints'):
        trainer.train(resume_from_checkpoint='latest')



def test_step_evaluation_loads_and_preserves_best_checkpoint(tmp_path):
    model = TinyModel()
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
    model = TinyModel()
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
        TinyModel(),
        TrainingConfig(
            eval_strategy='steps',
            eval_steps=1,
        ),
        DatasetConfig([]),
        loss_fn=squared_error,
    )

    with pytest.raises(ValueError, match='validation_dataloader'):
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
    model.weight = nn.Parameter(jnp.asarray(999.0))
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



def test_ema_decay_configuration_validation():
    with pytest.raises(ValueError, match='ema_decay'):
        TrainingConfig(ema_decay=0)
    with pytest.raises(ValueError, match='ema_decay'):
        TrainingConfig(ema_decay=1.0)
    assert TrainingConfig(ema_decay=None).ema_decay is None
    assert TrainingConfig(ema_decay=0.9999).ema_decay == 0.9999



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
