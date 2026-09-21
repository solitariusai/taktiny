"""CPU regression tests for Trainer and native Orbax checkpoints."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.trainer import DatasetConfig, Trainer, TrainerCallback, TrainingConfig


class TinyModel(nn.Module):
    def __init__(self):
        self.weight = nn.Parameter(jnp.asarray(0.0))
        self.frozen = nn.Parameter(jnp.asarray(3.0), trainable=False)
        self.rngs = nn.Rngs(17)


def loss_fn(model, batch, *, rng=None):
    noise = 0 if rng is None else .01 * jax.random.normal(rng)
    return jnp.mean((model.weight.value - batch['target'] + noise) ** 2)


def make_trainer(*, model=None, batches=None, callbacks=None, loss=loss_fn, **kwargs):
    if batches is None:
        batches = [{'target': jnp.asarray([1., 2.])},
                   {'target': jnp.asarray([2., 3.])},
                   {'target': jnp.asarray([3., 4.])}]
    return Trainer(
        TinyModel() if model is None else model,
        TrainingConfig(max_steps=kwargs.pop('max_steps', 4), log_interval=1, **kwargs),
        DatasetConfig(train_dataloader=batches, prefetch_size=0),
        loss_fn=loss, callbacks=callbacks,
    )


def assert_trees_equal(left, right):
    a, b = jax.tree.leaves(left), jax.tree.leaves(right)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('jit', [False, True])
def test_training_updates_only_trainable_parameters(jit):
    trainer = make_trainer(jit_compile=jit)
    original_key = jax.random.key_data(trainer.model.rngs.key)
    trainer.train()
    assert trainer.global_step == 4
    assert trainer.micro_step == 4
    assert float(trainer.model.weight.value) > 0
    assert float(trainer.model.frozen.value) == 3
    np.testing.assert_array_equal(jax.random.key_data(trainer.model.rngs.key), original_key)


@pytest.mark.parametrize('jit', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('accumulation', [1, 2])
def test_orbax_resume_matches_uninterrupted(tmp_path, jit, asynchronous, accumulation):
    config = dict(jit_compile=jit, gradient_accumulation_steps=accumulation, ema_decay=.8)
    reference = make_trainer(max_steps=5, **config)
    reference.train()
    partial = make_trainer(max_steps=2, output_dir=tmp_path, save_at_end=True,
                           save_async=asynchronous, **config)
    partial.train()
    checkpoint = tmp_path / 'checkpoint-2'
    assert (checkpoint / 'model_state').is_dir()
    assert (checkpoint / 'optimizer_state').is_dir()
    assert (checkpoint / 'ema_state').is_dir()
    assert not list(checkpoint.rglob('*.safetensors'))
    assert partial._pending_checkpoint is None
    resumed = make_trainer(max_steps=5, output_dir=tmp_path, **config)
    resumed.train(resume_from_checkpoint='latest')
    assert_trees_equal(reference.model, resumed.model)
    assert_trees_equal(reference.ema, resumed.ema)
    np.testing.assert_array_equal(jax.random.key_data(reference.rngs.key),
                                  jax.random.key_data(resumed.rngs.key))
    assert resumed.micro_step == reference.micro_step
    assert resumed.global_step == 5


def test_orbax_checkpoint_is_readable_without_trainer(tmp_path):
    trainer = make_trainer(max_steps=1, output_dir=tmp_path, save_at_end=True)
    trainer.train()
    with ocp.StandardCheckpointer() as checkpointer:
        state = checkpointer.restore(str(tmp_path / 'checkpoint-1' / 'model_state'))
    assert len(state) == len(jax.tree.leaves(trainer.model))
    with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as checkpointer:
        state = checkpointer.restore(str(tmp_path / 'checkpoint-1' / 'trainer_state'))
    assert state['global_step'] == 1
    assert state['process_count'] == 1


@pytest.mark.parametrize('split_step', [2, 3])
@pytest.mark.parametrize('accumulation', [1, 2])
def test_scheduled_resume_matches_every_step_and_optimizer(tmp_path, split_step, accumulation):
    # Keep the schedule horizon identical across interrupted and full runs.
    schedule = optax.warmup_cosine_decay_schedule(0.0, 0.03, 2, 8)
    config = dict(
        schedule=schedule, optimizer=optax.adamw(schedule), jit_compile=True,
        gradient_accumulation_steps=accumulation, ema_decay=0.8,
        save_at_end=True, save_async=True,
    )
    reference = make_trainer(max_steps=8, output_dir=tmp_path / 'full', **config)
    reference.train()
    partial = make_trainer(max_steps=split_step, output_dir=tmp_path / 'split', **config)
    partial.train()
    resumed = make_trainer(max_steps=8, output_dir=tmp_path / 'split', **config)
    resumed.train('latest')

    assert_trees_equal(reference.model, resumed.model)
    assert_trees_equal(reference.ema, resumed.ema)
    np.testing.assert_array_equal(jax.random.key_data(reference.rngs.key),
                                  jax.random.key_data(resumed.rngs.key))
    assert resumed.micro_step == reference.micro_step
    assert len(reference.log_history) == len(resumed.log_history) == 8
    for expected, actual in zip(reference.log_history, resumed.log_history):
        assert actual['step'] == expected['step']
        for field in ('loss', 'learning_rate', 'grad_norm'):
            assert actual[field] == pytest.approx(expected[field], rel=1e-6, abs=1e-7)

    with ocp.StandardCheckpointer() as checkpointer:
        full_state = checkpointer.restore(str(tmp_path / 'full/checkpoint-8/optimizer_state'))
        split_state = checkpointer.restore(str(tmp_path / 'split/checkpoint-8/optimizer_state'))
    assert_trees_equal(full_state, split_state)


class StatefulLoader:
    def __init__(self, binary=False):
        self.binary = binary
        self.restored = []

    def __iter__(self):
        parent = self

        class Iterator:
            index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index == 5:
                    raise StopIteration
                self.index += 1
                return {'target': jnp.asarray(float(self.index))}

            def get_state(self):
                state = {'index': self.index}
                return json.dumps(state).encode() if parent.binary else state

            def set_state(self, state):
                if isinstance(state, bytes):
                    state = json.loads(state)
                parent.restored.append(state['index'])
                self.index = state['index']
        return Iterator()


@pytest.mark.parametrize('binary', [False, True])
def test_native_iterator_state_is_restored(tmp_path, binary):
    reference = make_trainer(batches=StatefulLoader(binary), max_steps=7)
    reference.train()
    partial = make_trainer(batches=StatefulLoader(binary), max_steps=2,
                           output_dir=tmp_path, save_at_end=True)
    partial.train()
    data = StatefulLoader(binary)
    resumed = make_trainer(batches=data, max_steps=7, output_dir=tmp_path)
    resumed.train('latest')
    assert data.restored == [2]
    assert_trees_equal(reference.model, resumed.model)


def test_model_only_checkpoint_rejects_exact_resume(tmp_path):
    trainer = make_trainer(max_steps=1, output_dir=tmp_path, save_at_end=True,
                           save_optimizer_state=False)
    trainer.train()
    assert not (tmp_path / 'checkpoint-1' / 'optimizer_state').exists()
    resumed = make_trainer(output_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match='Optimizer state'):
        resumed.train('latest')


def test_stateless_optimizer_resume(tmp_path):
    config = dict(optimizer=optax.sgd(.01), jit_compile=True)
    reference = make_trainer(max_steps=4, **config)
    reference.train()
    make_trainer(max_steps=3, output_dir=tmp_path, save_at_end=True, **config).train()
    resumed = make_trainer(max_steps=4, output_dir=tmp_path, **config)
    resumed.train('latest')
    assert_trees_equal(reference.model, resumed.model)


def test_native_grain_loader_resume(tmp_path):
    from taktiny.data import DataLoader

    def source():
        return DataLoader([{'target': np.asarray([float(i)])} for i in range(5)],
                          shuffle=True, seed=4)

    reference = make_trainer(batches=source(), max_steps=7)
    reference.train()
    make_trainer(batches=source(), max_steps=2, output_dir=tmp_path, save_at_end=True).train()
    resumed = make_trainer(batches=source(), max_steps=7, output_dir=tmp_path)
    resumed.train('latest')
    assert_trees_equal(reference.model, resumed.model)


def test_changed_accumulation_is_rejected(tmp_path):
    make_trainer(max_steps=1, output_dir=tmp_path, save_at_end=True).train()
    trainer = make_trainer(output_dir=tmp_path, gradient_accumulation_steps=2)
    with pytest.raises(ValueError, match='gradient_accumulation_steps'):
        trainer.train('latest')


class Recorder(TrainerCallback):
    def __init__(self):
        self.events = []

    def on_train_begin(self, trainer):
        self.events.append('begin')

    def on_save(self, trainer, checkpoint_path):
        assert Path(checkpoint_path, '_CHECKPOINT_METADATA').is_file()
        self.events.append('save')

    def on_train_end(self, trainer):
        self.events.append('end')


def test_async_save_callbacks_and_retention(tmp_path):
    recorder = Recorder()
    trainer = make_trainer(output_dir=tmp_path, save_steps=1, save_async=True,
                           save_total_limit=2, callbacks=recorder)
    trainer.train()
    assert recorder.events == ['begin', 'save', 'save', 'save', 'save', 'end']
    assert [Path(p).name for p in trainer.saved_checkpoints] == ['checkpoint-3', 'checkpoint-4']


def test_save_failure_does_not_publish_checkpoint(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('save failed')
    monkeypatch.setattr(ocp.Checkpointer, 'save', fail)
    trainer = make_trainer(output_dir=tmp_path, save_at_end=True)
    with pytest.raises(RuntimeError, match='save failed'):
        trainer.train()
    assert trainer._checkpoint_paths() == []
    assert trainer._active_data_iterator is None


def test_latest_ignores_incomplete_and_unrelated_directories(tmp_path):
    (tmp_path / 'checkpoint-999.tmp').mkdir()
    (tmp_path / 'checkpoint-998').mkdir()
    trainer = make_trainer(output_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        trainer.train('latest')


def test_checkpoint_refuses_overwrite(tmp_path):
    make_trainer(max_steps=1, output_dir=tmp_path, save_at_end=True).train()
    trainer = make_trainer(max_steps=1, output_dir=tmp_path, save_at_end=True)
    with pytest.raises(FileExistsError):
        trainer.train()
    assert (tmp_path / 'checkpoint-1' / 'model_state').is_dir()


def test_aux_loss_evaluation_and_best_checkpoint(tmp_path):
    model = TinyModel()
    trainer = Trainer(model, TrainingConfig(
        max_steps=3, log_interval=1, output_dir=tmp_path, eval_strategy='steps',
        eval_steps=1, load_best_model_at_end=True, save_total_limit=1,
    ), DatasetConfig(train_dataloader=[{'target': jnp.asarray(1.)}],
                     validation_dataloader=[{'target': jnp.asarray(1.)}], prefetch_size=0),
        loss_fn=lambda m, b: (loss_fn(m, b), {'metric': jnp.asarray(2.)}), loss_has_aux=True)
    trainer.train()
    assert trainer.best_model_checkpoint is not None
    assert Path(trainer.best_model_checkpoint).is_dir()
    assert np.isfinite(trainer.evaluate()['eval_loss'])
    assert model.training is True


@pytest.mark.parametrize('jit', [False, True])
def test_accumulation_clipping_and_loss_scaling(jit):
    trainer = make_trainer(jit_compile=jit, gradient_accumulation_steps=2,
                           max_grad_norm=.1, loss_scale='dynamic',
                           loss_scale_growth_interval=1, max_steps=2)
    trainer.train()
    assert trainer.loss_scale == 32768 * 4
    assert trainer.micro_step == 3
    assert trainer.last_grad_norm is not None


def test_nonfinite_update_is_skipped():
    trainer = make_trainer(loss=lambda m, b: m.weight.value * jnp.asarray(float('nan')),
                           max_steps=1, loss_scale='dynamic')
    trainer.train()
    assert trainer.skipped_updates == 1
    assert float(trainer.model.weight.value) == 0
    assert trainer.loss_scale == 16384


def test_generic_caller_data_only():
    with pytest.raises(TypeError):
        DatasetConfig(train_dataloader='org/repo')
    with pytest.raises(TypeError):
        DatasetConfig(repo_id='org/repo')
    with pytest.raises(TypeError):
        DatasetConfig()
    assert list(DatasetConfig(train_dataloader=[1, 2]).train_dataloader) == [1, 2]


def test_one_shot_iterator_and_empty_input():
    trainer = make_trainer(batches=iter([{'target': jnp.asarray(1.)}]), max_steps=5)
    trainer.train()
    assert trainer.global_step == 1
    with pytest.raises(ValueError, match='no training batches'):
        make_trainer(batches=[]).train()


def test_sharded_model_checkpoint_restores_layout(tmp_path):
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    layout = NamedSharding(mesh, P('data'))
    model = TinyModel()
    model.weight = nn.Parameter(jax.device_put(jnp.zeros((len(jax.devices()),)), layout))
    model.frozen = nn.Parameter(jax.device_put(jnp.ones((len(jax.devices()),)), layout),
                                trainable=False)
    trainer = make_trainer(model=model, batches=[{'target': jnp.asarray(1.)}],
                           max_steps=1, output_dir=tmp_path, save_at_end=True)
    trainer.train()
    expected = np.asarray(model.weight.value).copy()
    restored_model = TinyModel()
    restored_model.weight = nn.Parameter(jax.device_put(jnp.zeros_like(model.weight.value), layout))
    restored_model.frozen = nn.Parameter(jax.device_put(jnp.ones_like(model.frozen.value), layout),
                                         trainable=False)
    resumed = make_trainer(model=restored_model, batches=[{'target': jnp.asarray(1.)}],
                           max_steps=1, output_dir=tmp_path)
    resumed.train('latest')
    np.testing.assert_allclose(restored_model.weight.value, expected)
    assert restored_model.weight.value.sharding.is_equivalent_to(layout, 1)
