import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from taktiny import nn
from taktiny.ensembles.vllm import (
    GPUWeightSync,
    LocalVLLMEngine,
    VLLM,
)
from taktiny.ensembles.vllm import _local as vllm_local
from taktiny.ensembles.vllm import _sync as vllm_sync
from taktiny.ensembles.vllm._weights import iter_checkpoint_weights
from taktiny.cosettes._overture import PretrainedModel
from taktiny.nn.lora import LoRALinear


class FakeEngine:
    def __init__(self):
        self.start_calls = 0
        self.generate_calls = []
        self.sync_calls = []
        self.close_calls = 0
        self.fail_sync = False

    def start(self):
        self.start_calls += 1

    def generate(self, *args, **kwargs):
        self.generate_calls.append((args, kwargs))
        return {'args': args, 'kwargs': kwargs}

    def sync(self, model, *, policy_version, **kwargs):
        self.sync_calls.append((model, policy_version, kwargs))
        if self.fail_sync:
            raise RuntimeError('synchronization failed')

    def close(self):
        self.close_calls += 1


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.options = kwargs


class FakeEngineCore:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


class FakeLLM:
    instances = []
    next_token_ids = []

    def __init__(self, *, model, **kwargs):
        self.model = model
        self.options = kwargs
        self.generate_calls = []
        self.llm_engine = SimpleNamespace(
            engine_core=FakeEngineCore(),
        )
        self.weight_calls = []
        self.instances.append(self)

    def generate(self, prompts, sampling_params, *, use_tqdm):
        self.generate_calls.append((
            prompts,
            sampling_params,
            use_tqdm,
        ))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(token_ids=token_ids),
                ],
            )
            for token_ids in self.next_token_ids
        ]

    def init_weight_transfer_engine(self, request):
        self.weight_calls.append(('init', request))

    def start_weight_update(self):
        self.weight_calls.append(('start',))

    def update_weights(self, request):
        self.weight_calls.append(('update', request))

    def finish_weight_update(self, weight_version=None):
        self.weight_calls.append(('finish', weight_version))

    def reset_prefix_cache(self):
        self.weight_calls.append(('reset_prefix_cache',))

    def collective_rpc(self, method):
        self.weight_calls.append(('collective_rpc', method))

    def update_weight_version(self, version):
        self.weight_calls.append(('version', version))


class FakeWeightTransferConfig:
    def __init__(self, *, backend):
        self.backend = backend


@pytest.fixture
def fake_vllm(monkeypatch):
    FakeLLM.instances = []
    FakeLLM.next_token_ids = []
    module = SimpleNamespace(
        LLM=FakeLLM,
        SamplingParams=FakeSamplingParams,
    )
    monkeypatch.setitem(sys.modules, 'vllm', module)
    monkeypatch.setitem(
        sys.modules,
        'vllm.config',
        SimpleNamespace(
            WeightTransferConfig=FakeWeightTransferConfig,
        ),
    )
    return module


def tiny_model():
    return SimpleNamespace(
        base_model_name_or_path='example/tiny-model',
        config=SimpleNamespace(
            eos_token_id=[2, 3],
            pad_token_id=99,
        ),
    )


def test_vllm_requires_model():
    with pytest.raises(ValueError, match='model is required'):
        VLLM(None, engine=FakeEngine())


def test_vllm_rejects_multiple_engine_sources():
    with pytest.raises(ValueError, match='either engine or engine_factory'):
        VLLM(
            object(),
            engine=FakeEngine(),
            engine_factory=lambda model: FakeEngine(),
        )


def test_vllm_validates_engine_contract():
    with pytest.raises(TypeError, match='generate, sync, close'):
        VLLM(object(), engine=object())


def test_vllm_starts_existing_engine_and_delegates_generation():
    model = object()
    engine = FakeEngine()
    runtime = VLLM(model, engine=engine)

    result = runtime.generate('prompt', temperature=0.7)

    assert runtime.model is model
    assert runtime.started
    assert engine.start_calls == 1
    assert engine.generate_calls == [
        (('prompt',), {'temperature': 0.7}),
    ]
    assert result == {
        'args': ('prompt',),
        'kwargs': {'temperature': 0.7},
    }


def test_vllm_lazily_builds_engine_with_options():
    model = object()
    engine = FakeEngine()
    factory_calls = []

    def factory(candidate, **kwargs):
        factory_calls.append((candidate, kwargs))
        return engine

    runtime = VLLM(
        model,
        engine_factory=factory,
        auto_start=False,
        tensor_parallel_size=8,
        placement='colocated',
    )

    assert not runtime.started
    assert factory_calls == []
    assert runtime.engine_options == {
        'tensor_parallel_size': 8,
        'placement': 'colocated',
    }

    runtime.generate([1, 2, 3], max_new_tokens=4)

    assert factory_calls == [(
        model,
        {
            'tensor_parallel_size': 8,
            'placement': 'colocated',
        },
    )]
    assert engine.start_calls == 1


def test_vllm_sync_versions_only_successful_updates():
    model = object()
    engine = FakeEngine()
    runtime = VLLM(model, engine=engine)

    assert runtime.sync(chunk_size=16) == 1
    assert runtime.policy_version == 1
    assert engine.sync_calls == [
        (model, 1, {'chunk_size': 16}),
    ]

    engine.fail_sync = True
    with pytest.raises(RuntimeError, match='synchronization failed'):
        runtime.sync()

    assert runtime.policy_version == 1


def test_vllm_sync_accepts_explicit_policy_version():
    model = object()
    engine = FakeEngine()
    runtime = VLLM(model, engine=engine, auto_start=False)

    assert runtime.sync(policy_version=5) == 5
    assert runtime.policy_version == 5
    assert engine.sync_calls == [(model, 5, {})]

    with pytest.raises(
        ValueError,
        match='greater than the current policy version',
    ):
        runtime.sync(policy_version=5)


def test_vllm_close_is_idempotent_and_blocks_use():
    engine = FakeEngine()
    runtime = VLLM(object(), engine=engine, auto_start=False)

    runtime.close()
    runtime.close()

    assert runtime.closed
    assert not runtime.started
    assert engine.close_calls == 1
    with pytest.raises(RuntimeError, match='runtime is closed'):
        runtime.generate('prompt')
    with pytest.raises(RuntimeError, match='runtime is closed'):
        runtime.sync()


def test_vllm_context_manager_starts_and_closes_engine():
    engine = FakeEngine()

    with VLLM(
        object(),
        engine=engine,
        auto_start=False,
    ) as runtime:
        assert runtime.started
        assert not runtime.closed

    assert runtime.closed
    assert engine.start_calls == 1
    assert engine.close_calls == 1


def test_vllm_default_engine_loads_model_for_active_platform(fake_vllm):
    model = tiny_model()

    runtime = VLLM(
        model,
        platform='gpu',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.5,
    )

    assert isinstance(runtime.engine, LocalVLLMEngine)
    assert runtime.engine.platform == 'gpu'
    assert len(FakeLLM.instances) == 1
    llm = FakeLLM.instances[0]
    assert llm.model == 'example/tiny-model'
    assert llm.options == {
        'tensor_parallel_size': 2,
        'gpu_memory_utilization': 0.5,
        'weight_transfer_config': llm.options[
            'weight_transfer_config'
        ],
        'skip_tokenizer_init': True,
        'generation_config': 'vllm',
    }
    assert (
        llm.options['weight_transfer_config'].backend
        == 'ipc'
    )

    core = llm.llm_engine.engine_core
    runtime.close()
    assert core.shutdown_calls == 1


def test_vllm_generate_normalizes_padded_token_batches(fake_vllm):
    FakeLLM.next_token_ids = [
        [30, 31],
        [40],
    ]
    runtime = VLLM(tiny_model(), platform='gpu')
    input_ids = jnp.asarray([
        [0, 10, 11],
        [20, 21, 0],
    ], dtype=jnp.int32)
    attention_mask = jnp.asarray([
        [0, 1, 1],
        [1, 1, 0],
    ])

    result = runtime.generate(
        input_ids,
        max_new_tokens=2,
        temperature=0.7,
        top_k=12,
        top_p=0.9,
        seed=7,
        attention_mask=attention_mask,
        repetition_penalty=1.1,
    )

    np.testing.assert_array_equal(
        result,
        np.asarray([
            [0, 10, 11, 30, 31],
            [20, 21, 0, 40, 99],
        ], dtype=np.int32),
    )
    llm = FakeLLM.instances[0]
    prompts, sampling_params, use_tqdm = llm.generate_calls[0]
    assert prompts == [
        {'prompt_token_ids': [10, 11]},
        {'prompt_token_ids': [20, 21]},
    ]
    assert use_tqdm is False
    assert sampling_params.options == {
        'temperature': 0.7,
        'top_k': 12,
        'top_p': 0.9,
        'seed': 7,
        'repetition_penalty': 1.1,
        'stop_token_ids': [2, 3],
        'max_tokens': 2,
        'detokenize': False,
        'skip_special_tokens': False,
    }


def test_vllm_generate_zero_tokens_skips_engine_request(fake_vllm):
    runtime = VLLM(tiny_model(), platform='gpu')
    input_ids = jnp.asarray([[1, 2, 3]])

    result = runtime.generate(input_ids, max_new_tokens=0)

    np.testing.assert_array_equal(result, input_ids)
    assert FakeLLM.instances[0].generate_calls == []


def test_vllm_generate_rejects_offline_streamer(fake_vllm):
    runtime = VLLM(tiny_model(), platform='gpu')

    with pytest.raises(
        NotImplementedError,
        match='asynchronous vLLM engine',
    ):
        runtime.generate(
            jnp.asarray([[1, 2, 3]]),
            max_new_tokens=1,
            streamer=object(),
        )


def test_local_vllm_rejects_non_taktiny_tpu_backend(monkeypatch):
    monkeypatch.delenv('TPU_BACKEND_TYPE', raising=False)
    monkeypatch.delenv('MODEL_IMPL_TYPE', raising=False)
    with pytest.raises(
        NotImplementedError,
        match='cannot execute a TakTiny nn.Module',
    ):
        LocalVLLMEngine(tiny_model(), platform='tpu')

    assert 'TPU_BACKEND_TYPE' not in vllm_local.os.environ
    assert 'MODEL_IMPL_TYPE' not in vllm_local.os.environ


def test_local_vllm_reports_platform_dependency(monkeypatch):
    def missing_vllm(name):
        raise ModuleNotFoundError(
            "No module named 'vllm'",
            name='vllm',
        )

    monkeypatch.setattr(
        vllm_local.importlib,
        'import_module',
        missing_vllm,
    )
    engine = LocalVLLMEngine(tiny_model(), platform='gpu')

    with pytest.raises(
        ImportError,
        match='vllm is required.*GPU',
    ):
        engine.start()


def test_local_vllm_requires_pretrained_model_source():
    with pytest.raises(
        ValueError,
        match='Unable to determine the vLLM model source',
    ):
        LocalVLLMEngine(
            SimpleNamespace(config={}),
            platform='gpu',
        )


class TinySyncModel(nn.Module):
    def __init__(self):
        rngs = nn.Rngs(0)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(
            3,
            2,
            rngs=rngs,
        )
        base = nn.Linear(
            2,
            3,
            bias=False,
            rngs=rngs,
        )
        self.model.proj = LoRALinear(
            base,
            rank=1,
            alpha=2,
            rngs=rngs,
        )
        self.model.embed_tokens.embedding.value = jnp.asarray([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        self.model.proj.base_layer.weight.value = jnp.asarray([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        self.model.proj.lora_A.value = jnp.asarray([
            [1.0],
            [2.0],
        ])
        self.model.proj.lora_B.value = jnp.asarray([
            [0.5, 1.0, 1.5],
        ])
        self.base_model_name_or_path = 'example/tiny-model'
        self.config = SimpleNamespace()


def test_checkpoint_weight_export_merges_lora_and_restores_layout():
    model = TinySyncModel()

    weights = dict(iter_checkpoint_weights(model))

    assert set(weights) == {
        'model.embed_tokens.weight',
        'model.proj.weight',
    }
    np.testing.assert_array_equal(
        weights['model.embed_tokens.weight'],
        model.model.embed_tokens.embedding.value,
    )
    expected = (
        model.model.proj.base_layer.weight.value
        + (
            model.model.proj.lora_A.value
            @ model.model.proj.lora_B.value
        )
        * model.model.proj.scaling
    ).T
    np.testing.assert_allclose(
        weights['model.proj.weight'],
        expected,
    )


def test_checkpoint_weight_export_supports_custom_mapper():
    model = TinySyncModel()

    weights = dict(iter_checkpoint_weights(
        model,
        lambda name, value: (
            (name.replace('model.', 'transformer.'), value)
            if 'embed_tokens' not in name
            else None
        ),
    ))

    assert set(weights) == {'transformer.proj.weight'}


class TinyStackedSyncModel(nn.Module):
    _expand_stacked_state_dict = staticmethod(
        PretrainedModel._expand_stacked_state_dict
    )

    def __init__(self):
        self.model = nn.Module()
        self.model.layers = nn.Module()
        self.model.layers.stacked = nn.Module()
        parameter = nn.Parameter(jnp.arange(12.0).reshape(2, 2, 3))
        parameter.input_axis_count = 1
        self.model.layers.stacked.proj = nn.Module()
        self.model.layers.stacked.proj.weight = parameter


def test_checkpoint_weight_export_expands_compact_layers_in_order():
    weights = list(iter_checkpoint_weights(TinyStackedSyncModel()))

    assert [name for name, _ in weights] == [
        'model.layers.0.proj.weight',
        'model.layers.1.proj.weight',
    ]
    np.testing.assert_array_equal(
        weights[0][1],
        np.arange(6.0).reshape(2, 3).T,
    )
    np.testing.assert_array_equal(
        weights[1][1],
        np.arange(6.0, 12.0).reshape(2, 3).T,
    )


def test_gpu_sync_uses_vllm_ipc_lifecycle(
    fake_vllm,
    monkeypatch,
):
    class FakeIPCTrainerInitInfo:
        def __init__(self, **kwargs):
            self.options = kwargs

    class FakeTrainerEngine:
        def __init__(self, client, source):
            self.client = client
            self.source = source

        def send_weights(self):
            self.client.start_weight_update()
            names = [name for name, _ in self.source]
            self.client.update_weights({'names': names})
            self.client.finish_weight_update()

    class FakeTrainerFactory:
        @classmethod
        def trainer_init(cls, *, init_info, client, source):
            assert init_info.options['rank'] == 0
            assert init_info.options['packed'] is True
            client.init_weight_transfer_engine({
                'packed': init_info.options['packed'],
            })
            return FakeTrainerEngine(client, source)

    monkeypatch.setitem(
        sys.modules,
        'vllm.distributed.weight_transfer.base',
        SimpleNamespace(
            ParamMeta=lambda name, dtype, shape: (
                name,
                dtype,
                shape,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        'vllm.distributed.weight_transfer.factory',
        SimpleNamespace(
            WeightTransferTrainerFactory=FakeTrainerFactory,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        'vllm.distributed.weight_transfer.ipc_engine',
        SimpleNamespace(
            IPCTrainerInitInfo=FakeIPCTrainerInitInfo,
        ),
    )
    monkeypatch.setattr(
        vllm_sync,
        '_torch_from_jax',
        lambda value, torch, device: np.asarray(value),
    )
    runtime = VLLM(
        TinySyncModel(),
        platform='gpu',
        sync_packed=True,
    )

    assert runtime.sync() == 1

    llm = runtime.engine.llm
    assert llm.weight_calls == [
        ('init', {'init_info': {'packed': True}}),
        ('start',),
        (
            'update',
            {
                'update_info': {
                    'names': [
                        'model.embed_tokens.weight',
                        'model.proj.weight',
                    ],
                },
            },
        ),
        ('finish', '1'),
    ]


def test_gpu_sync_rejects_multi_gpu_local_ipc(fake_vllm):
    runtime = VLLM(
        TinySyncModel(),
        platform='gpu',
        tensor_parallel_size=2,
    )

    with pytest.raises(
        NotImplementedError,
        match='tensor_parallel_size=1',
    ):
        runtime.sync()
