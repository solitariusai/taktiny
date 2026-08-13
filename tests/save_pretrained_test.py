import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import qwix
from safetensors import safe_open
from safetensors.numpy import save_file

from taktiny import Takt, nn
from taktiny.cosettes import _overture as pretrained_base
from taktiny.cosettes._overture import PretrainedModel
from taktiny.maestro.config import ModelConfig
from taktiny.peft import LoraConfig


class TinyPretrainedModel(PretrainedModel):
    def __init__(self):
        self.config = ModelConfig(
            architectures=['TinyPretrainedModel'],
            hidden_size=4,
        )
        self.proj = nn.Linear(
            4,
            3,
            bias=False,
            rngs=nn.Rngs(0),
        )

    def __call__(self, x):
        return self.proj(x)


class TinyProjectionBlock(nn.Module):
    def __init__(self, seed):
        self.proj = nn.Linear(
            4,
            3,
            bias=False,
            rngs=nn.Rngs(seed),
        )


class TinyStackedPretrainedModel(PretrainedModel):
    def __init__(self):
        self.layers = nn.SeqStack(
            TinyProjectionBlock(seed)
            for seed in range(2)
        )


class TinyLoadableModel(PretrainedModel):
    def __init__(
        self,
        config,
        rngs,
        mesh=None,
        sharding_rules=None,
    ):
        self.config = config
        self.proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            quant=getattr(config, 'quant', None),
            rngs=rngs,
        )


class TinyStackedLoadableModel(PretrainedModel):
    def __init__(
        self,
        config,
        rngs,
        mesh=None,
        sharding_rules=None,
    ):
        self.config = config

        def layers():
            for _ in range(config.num_hidden_layers):
                layer = TinyProjectionBlock(0)
                layer.proj.weight.value = jnp.zeros(
                    (4, 3),
                    dtype=config.torch_dtype,
                )
                layer.proj.weight.quantization = getattr(
                    config,
                    'quant',
                    None,
                )
                yield layer

        self.layers = nn.SeqStack(layers())


class TinyGroupedStackedLoadableModel(PretrainedModel):
    def __init__(
        self,
        config,
        rngs,
        mesh=None,
        sharding_rules=None,
    ):
        self.config = config
        modes = ('a', 'a', 'b', 'b', 'a', 'a')
        layers = []
        for mode in modes:
            layer = TinyProjectionBlock(0)
            layer.mode = mode
            layer.proj.weight.value = jnp.zeros(
                (4, 3),
                dtype=config.torch_dtype,
            )
            layers.append(layer)
        self.layers = nn.SeqStack(layers)


class FakeRepoUrl:
    repo_id = 'resolved/model'


class FakeCommitInfo:
    commit_url = 'https://huggingface.co/resolved/model/commit/abc123'


class FakeHfApi:
    calls = {}

    def __init__(self, **kwargs):
        self.calls['init'] = kwargs

    def create_repo(self, **kwargs):
        self.calls['create_repo'] = kwargs
        return FakeRepoUrl()

    def create_branch(self, **kwargs):
        self.calls['create_branch'] = kwargs

    def upload_folder(self, **kwargs):
        self.calls['upload_folder'] = {
            **kwargs,
            'files': sorted(os.listdir(kwargs['folder_path'])),
        }
        self.calls['temporary_directory'] = kwargs['folder_path']
        return FakeCommitInfo()


def test_save_pretrained_writes_only_lora_adapter(tmp_path):
    model = TinyPretrainedModel()
    model.base_model_name_or_path = 'example/base-model'
    model = Takt.apply_peft(
        model,
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
            rngs=nn.Rngs(1),
        ),
    )

    saved_paths = model.save_pretrained(tmp_path)

    adapter_path = tmp_path / 'adapter_model.safetensors'
    config_path = tmp_path / 'adapter_config.json'
    assert adapter_path.exists()
    assert config_path.exists()
    assert (tmp_path / 'config.json').exists()
    assert not (tmp_path / 'model.safetensors').exists()
    assert saved_paths == (
        str(tmp_path / 'config.json'),
        str(tmp_path / 'adapter_config.json'),
        str(tmp_path / 'adapter_model.safetensors'),
    )

    with safe_open(adapter_path, framework='np') as adapter:
        assert set(adapter.keys()) == {
            'proj.lora_A',
            'proj.lora_B',
        }

    with config_path.open() as config_file:
        config = json.load(config_file)
    assert config == {
        'peft_type': 'LORA',
        'target_modules': ['proj'],
        'rank': 2,
        'alpha': 4.0,
        'base_model_name_or_path': 'example/base-model',
    }
    with (tmp_path / 'config.json').open() as config_file:
        model_config = json.load(config_file)
    assert model_config == {
        'architectures': ['TinyPretrainedModel'],
        'hidden_size': 4,
    }

    output = jax.jit(lambda candidate, x: candidate(x))(
        model,
        jnp.ones((1, 4)),
    )
    assert output.shape == (1, 3)


def test_save_pretrained_writes_full_model_after_lora_merge(tmp_path):
    model = Takt.apply_peft(
        TinyPretrainedModel(),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
            rngs=nn.Rngs(1),
        ),
    )
    Takt.merge_peft(model, dtype='bfloat16')

    saved_paths = model.save_pretrained(tmp_path)

    assert saved_paths == (
        str(tmp_path / 'config.json'),
        str(tmp_path / 'model.safetensors'),
        str(tmp_path / 'model.safetensors.index.json'),
    )
    assert not (tmp_path / 'adapter_model.safetensors').exists()
    assert not (tmp_path / 'adapter_config.json').exists()
    with safe_open(
        tmp_path / 'model.safetensors',
        framework='np',
    ) as checkpoint:
        assert checkpoint.get_tensor('proj.weight').dtype == np.dtype(
            'bfloat16'
        )


def test_save_pretrained_without_lora_writes_full_model(tmp_path):
    model = TinyPretrainedModel()

    saved_paths = model.save_pretrained(tmp_path)

    weights_path = tmp_path / 'model.safetensors'
    index_path = tmp_path / 'model.safetensors.index.json'
    assert weights_path.exists()
    assert index_path.exists()
    assert (tmp_path / 'config.json').exists()
    assert not (tmp_path / 'adapter_model.safetensors').exists()
    assert saved_paths == (
        str(tmp_path / 'config.json'),
        str(tmp_path / 'model.safetensors'),
        str(tmp_path / 'model.safetensors.index.json'),
    )

    with safe_open(weights_path, framework='np') as checkpoint:
        assert set(checkpoint.keys()) == {'proj.weight'}

    with index_path.open() as index_file:
        index = json.load(index_file)
    assert index['weight_map'] == {
        'proj.weight': 'model.safetensors',
    }
    with (tmp_path / 'config.json').open() as config_file:
        config = json.load(config_file)
    assert config == {
        'architectures': ['TinyPretrainedModel'],
        'hidden_size': 4,
    }


def test_save_pretrained_round_trips_qwix_model(tmp_path):
    source = TinyPretrainedModel()
    source.proj.weight.value = qwix.quantize(
        source.proj.weight.value,
        'int4',
        channelwise_axes=(1,),
    )

    saved_paths = source.save_pretrained(tmp_path)

    quantization_path = tmp_path / 'quantization_config.json'
    assert quantization_path.is_file()
    assert str(quantization_path) in saved_paths
    with quantization_path.open() as quantization_file:
        metadata = json.load(quantization_file)
    specification = metadata['parameters']['proj.weight']
    assert specification == {
        'qtype': 'int4',
        'qvalue_dtype': 'int4',
        'qvalue': 'proj.weight.__qwix__.qvalue',
        'scale': 'proj.weight.__qwix__.scale',
        'zero_point': None,
    }
    with safe_open(
        tmp_path / 'model.safetensors',
        framework='np',
    ) as checkpoint:
        assert set(checkpoint.keys()) == {
            'proj.weight.__qwix__.qvalue',
            'proj.weight.__qwix__.scale',
        }
        assert checkpoint.get_tensor(
            'proj.weight.__qwix__.qvalue'
        ).dtype == np.int8

    restored = TinyPretrainedModel()
    restored.load_pretrained(tmp_path)

    restored_weight = restored.proj.weight.value
    assert isinstance(restored_weight, qwix.QArray)
    assert restored_weight.qtype == 'int4'
    assert restored_weight.qvalue.dtype == jnp.dtype('int4')
    np.testing.assert_array_equal(
        restored_weight.qvalue,
        source.proj.weight.value.qvalue,
    )
    np.testing.assert_array_equal(
        restored_weight.scale,
        source.proj.weight.value.scale,
    )


def test_save_pretrained_round_trips_stacked_qwix_model(tmp_path):
    source = TinyStackedPretrainedModel()
    source.layers.stacked.proj.weight.value = qwix.quantize(
        source.layers.stacked.proj.weight.value,
        'nf4',
        channelwise_axes=(0, 2),
    )
    source.save_pretrained(tmp_path, max_shard_size=64)

    restored = TinyStackedPretrainedModel()
    restored.load_pretrained(tmp_path)

    restored_weight = restored.layers.stacked.proj.weight.value
    assert isinstance(restored_weight, qwix.QArray)
    assert restored_weight.qtype == 'nf4'
    assert restored_weight.qvalue.dtype == jnp.dtype('uint4')
    np.testing.assert_array_equal(
        restored_weight.qvalue,
        source.layers.stacked.proj.weight.value.qvalue,
    )
    np.testing.assert_array_equal(
        restored_weight.scale,
        source.layers.stacked.proj.weight.value.scale,
    )


def test_dense_save_removes_stale_qwix_metadata(tmp_path):
    model = TinyPretrainedModel()
    dense_weight = model.proj.weight.value
    model.proj.weight.value = qwix.quantize(
        dense_weight,
        'int4',
        channelwise_axes=(1,),
    )
    model.save_pretrained(tmp_path)
    assert (tmp_path / 'quantization_config.json').is_file()

    model.proj.weight.value = dense_weight
    model.save_pretrained(tmp_path)

    assert not (tmp_path / 'quantization_config.json').exists()
    restored = TinyPretrainedModel().load_pretrained(tmp_path)
    assert not isinstance(restored.proj.weight.value, qwix.QArray)


def test_from_pretrained_recognizes_native_qwix_checkpoint(tmp_path):
    config = ModelConfig(
        architectures=['TinyLoadableModel'],
        hidden_size=4,
        torch_dtype='bfloat16',
    )
    source = TinyLoadableModel(config, nn.Rngs(0))
    source.proj.weight.value = qwix.quantize(
        source.proj.weight.value,
        'int4',
        channelwise_axes=(1,),
    )
    source.save_pretrained(tmp_path)

    restored = TinyLoadableModel.from_pretrained(
        tmp_path,
        config,
        local=True,
    )

    assert isinstance(restored.proj.weight.value, qwix.QArray)
    np.testing.assert_array_equal(
        restored.proj.weight.value.qvalue,
        source.proj.weight.value.qvalue,
    )


def test_from_pretrained_places_ptq_weights_on_default_device(tmp_path):
    save_file(
        {
            'proj.weight': np.arange(
                16,
                dtype=np.float32,
            ).reshape(4, 4),
        },
        tmp_path / 'model.safetensors',
    )
    config = ModelConfig(
        architectures=['TinyLoadableModel'],
        hidden_size=4,
        torch_dtype='bfloat16',
    )

    restored = TinyLoadableModel.from_pretrained(
        tmp_path,
        config,
        local=True,
        dtype='int4',
    )

    weight = restored.proj.weight.value
    assert isinstance(weight, qwix.QArray)
    assert weight.qtype == 'int4'
    assert all(
        jax.devices()[0] in component.devices()
        for component in jax.tree.leaves(weight)
    )


def test_from_pretrained_streams_numbered_layers_into_seqstack(tmp_path):
    layer_weights = [
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.arange(12, 24, dtype=np.float32).reshape(3, 4),
    ]
    save_file(
        {
            f'layers.{index}.proj.weight': weight
            for index, weight in enumerate(layer_weights)
        },
        tmp_path / 'model.safetensors',
    )
    config = ModelConfig(
        num_hidden_layers=2,
        torch_dtype='float32',
    )

    restored = TinyStackedLoadableModel.from_pretrained(
        tmp_path,
        config,
        local=True,
    )

    expected = np.stack(layer_weights, axis=0).transpose(0, 2, 1)
    np.testing.assert_array_equal(
        restored.layers.stacked.proj.weight.value,
        expected,
    )


def test_from_pretrained_streams_numbered_layers_into_grouped_seqstack(
    tmp_path,
):
    layer_weights = [
        np.full((3, 4), index, dtype=np.float32)
        for index in range(6)
    ]
    save_file(
        {
            f'layers.{index}.proj.weight': weight
            for index, weight in enumerate(layer_weights)
        },
        tmp_path / 'model.safetensors',
    )
    config = ModelConfig(
        num_hidden_layers=6,
        torch_dtype='float32',
    )

    restored = TinyGroupedStackedLoadableModel.from_pretrained(
        tmp_path,
        config,
        local=True,
    )

    assert restored.layers.group_sizes == (2, 2, 2)
    for group_index, group in enumerate(restored.layers.groups):
        expected = np.stack(
            layer_weights[group_index * 2:(group_index + 1) * 2],
            axis=0,
        ).transpose(0, 2, 1)
        np.testing.assert_array_equal(
            group.stacked.proj.weight.value,
            expected,
        )


def test_from_pretrained_quantizes_seqstack_layers_while_loading(tmp_path):
    save_file(
        {
            f'layers.{index}.proj.weight': (
                np.arange(index * 12, (index + 1) * 12, dtype=np.float32)
                .reshape(3, 4)
            )
            for index in range(2)
        },
        tmp_path / 'model.safetensors',
    )
    config = ModelConfig(
        num_hidden_layers=2,
        torch_dtype='bfloat16',
    )

    restored = TinyStackedLoadableModel.from_pretrained(
        tmp_path,
        config,
        local=True,
        dtype='int4',
    )

    weight = restored.layers.stacked.proj.weight.value
    assert isinstance(weight, qwix.QArray)
    assert weight.shape == (2, 4, 3)
    assert weight.scale.shape[0] == 2
    assert not restored.layers.stacked.proj.weight.trainable


def test_save_pretrained_finds_lora_inside_seqstack(tmp_path):
    model = Takt.apply_peft(
        TinyStackedPretrainedModel(),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
        ),
    )

    model.save_pretrained(tmp_path)

    with safe_open(
        tmp_path / 'adapter_model.safetensors',
        framework='np',
    ) as adapter:
        assert set(adapter.keys()) == {
            'layers.0.proj.lora_A',
            'layers.0.proj.lora_B',
            'layers.1.proj.lora_A',
            'layers.1.proj.lora_B',
        }
        assert adapter.get_tensor('layers.0.proj.lora_A').shape == (4, 2)
        assert adapter.get_tensor('layers.0.proj.lora_B').shape == (2, 3)


def test_expand_stacked_state_dict_orders_parameters_by_layer():
    state = {
        'embed.weight': jnp.zeros((3, 4)),
        'layers.stacked.norm.weight': jnp.zeros((2, 4)),
        'layers.stacked.proj.weight': jnp.zeros((2, 4, 3)),
        'norm.weight': jnp.zeros((4,)),
    }

    expanded = PretrainedModel._expand_stacked_state_dict(state)

    assert list(expanded) == [
        'embed.weight',
        'layers.0.norm.weight',
        'layers.0.proj.weight',
        'layers.1.norm.weight',
        'layers.1.proj.weight',
        'norm.weight',
    ]


def test_expand_grouped_seqstack_uses_global_layer_indices():
    state = {
        'layers.groups.0.stacked.proj.weight': jnp.zeros((2, 4, 3)),
        'layers.groups.1.stacked.proj.weight': jnp.ones((2, 4, 3)),
        'layers.groups.2.stacked.proj.weight': jnp.full((2, 4, 3), 2),
    }

    expanded = PretrainedModel._expand_stacked_state_dict(state)

    assert list(expanded) == [
        f'layers.{index}.proj.weight'
        for index in range(6)
    ]


def test_save_pretrained_expands_full_seqstack_state(tmp_path):
    model = TinyStackedPretrainedModel()

    model.save_pretrained(tmp_path)

    with safe_open(
        tmp_path / 'model.safetensors',
        framework='np',
    ) as checkpoint:
        assert set(checkpoint.keys()) == {
            'layers.0.proj.weight',
            'layers.1.proj.weight',
        }
        assert checkpoint.get_tensor(
            'layers.0.proj.weight'
        ).shape == (4, 3)

    with (tmp_path / 'model.safetensors.index.json').open() as index_file:
        index = json.load(index_file)
    assert index['weight_map'] == {
        'layers.0.proj.weight': 'model.safetensors',
        'layers.1.proj.weight': 'model.safetensors',
    }


def test_save_pretrained_shards_full_model_at_size_limit(tmp_path):
    model = TinyStackedPretrainedModel()

    saved_paths = model.save_pretrained(tmp_path, max_shard_size=48)

    shard_names = {
        'model-00001-of-00002.safetensors',
        'model-00002-of-00002.safetensors',
    }
    assert {path.name for path in tmp_path.glob('model-*.safetensors')} == (
        shard_names
    )
    assert not (tmp_path / 'model.safetensors').exists()

    with (tmp_path / 'model.safetensors.index.json').open() as index_file:
        index = json.load(index_file)
    assert index['metadata']['total_size'] == 96
    assert set(index['weight_map']) == {
        'layers.0.proj.weight',
        'layers.1.proj.weight',
    }
    assert set(index['weight_map'].values()) == shard_names
    assert saved_paths == (
        str(tmp_path / 'config.json'),
        str(tmp_path / 'model-00001-of-00002.safetensors'),
        str(tmp_path / 'model-00002-of-00002.safetensors'),
        str(tmp_path / 'model.safetensors.index.json'),
    )


def test_save_pretrained_shards_lora_adapter_at_size_limit(tmp_path):
    model = Takt.apply_peft(
        TinyStackedPretrainedModel(),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
        ),
    )

    model.save_pretrained(tmp_path, max_shard_size=56)

    shard_names = {
        path.name
        for path in tmp_path.glob('adapter_model-*.safetensors')
    }
    assert shard_names == {
        'adapter_model-00001-of-00002.safetensors',
        'adapter_model-00002-of-00002.safetensors',
    }
    assert not (tmp_path / 'adapter_model.safetensors').exists()

    index_path = tmp_path / 'adapter_model.safetensors.index.json'
    with index_path.open() as index_file:
        index = json.load(index_file)
    assert index['weight_map'] == {
        'layers.0.proj.lora_A':
            'adapter_model-00001-of-00002.safetensors',
        'layers.0.proj.lora_B':
            'adapter_model-00001-of-00002.safetensors',
        'layers.1.proj.lora_A':
            'adapter_model-00002-of-00002.safetensors',
        'layers.1.proj.lora_B':
            'adapter_model-00002-of-00002.safetensors',
    }


def test_save_pretrained_replaces_stale_shards(tmp_path):
    model = TinyStackedPretrainedModel()
    model.save_pretrained(tmp_path, max_shard_size=48)

    model.save_pretrained(tmp_path, max_shard_size='1KB')

    assert not list(tmp_path.glob('model-*.safetensors'))
    assert (tmp_path / 'model.safetensors').exists()
    with (tmp_path / 'model.safetensors.index.json').open() as index_file:
        index = json.load(index_file)
    assert set(index['weight_map'].values()) == {'model.safetensors'}


def test_save_writes_current_transformers_config_after_dtype_override(
    tmp_path,
):
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    source.mkdir()
    save_file(
        {
            'proj.weight': np.arange(
                16,
                dtype=np.float32,
            ).reshape(4, 4),
        },
        source / 'model.safetensors',
    )
    config = ModelConfig(
        architectures=['TinyLoadableModel'],
        hidden_size=4,
        torch_dtype='float32',
    )

    model = TinyLoadableModel.from_pretrained(
        source,
        config,
        local=True,
        dtype='float16',
    )
    model.save_pretrained(output)

    assert model.config.torch_dtype == 'float16'
    with (output / 'config.json').open() as config_file:
        saved_config = json.load(config_file)
    assert saved_config['torch_dtype'] == 'float16'


def test_push_to_hub_uploads_sharded_model(monkeypatch):
    FakeHfApi.calls = {}
    monkeypatch.setattr(pretrained_base, 'HfApi', FakeHfApi)
    model = TinyStackedPretrainedModel()

    url = model.push_to_hub(
        'owner/model',
        private=True,
        token='secret-token',
        revision='experiment',
        commit_message='Upload checkpoint',
        commit_description='Test upload',
        create_pr=True,
        max_shard_size=48,
    )

    assert url == FakeCommitInfo.commit_url
    assert FakeHfApi.calls['init'] == {
        'token': 'secret-token',
        'library_name': 'taktiny',
    }
    assert FakeHfApi.calls['create_repo'] == {
        'repo_id': 'owner/model',
        'private': True,
        'token': 'secret-token',
        'repo_type': 'model',
        'exist_ok': True,
    }
    assert FakeHfApi.calls['create_branch'] == {
        'repo_id': 'resolved/model',
        'branch': 'experiment',
        'token': 'secret-token',
        'exist_ok': True,
    }
    upload = FakeHfApi.calls['upload_folder']
    assert upload['repo_id'] == 'resolved/model'
    assert upload['commit_message'] == 'Upload checkpoint'
    assert upload['commit_description'] == 'Test upload'
    assert upload['revision'] == 'experiment'
    assert upload['create_pr'] is True
    assert upload['delete_patterns'] == [
        'model.safetensors',
        'model-*-of-*.safetensors',
        'model.safetensors.index.json',
        'quantization_config.json',
    ]
    assert upload['files'] == [
        'config.json',
        'model-00001-of-00002.safetensors',
        'model-00002-of-00002.safetensors',
        'model.safetensors.index.json',
    ]
    assert not os.path.exists(FakeHfApi.calls['temporary_directory'])


def test_push_to_hub_preserves_base_weights_for_adapter(monkeypatch):
    FakeHfApi.calls = {}
    monkeypatch.setattr(pretrained_base, 'HfApi', FakeHfApi)
    model = Takt.apply_peft(
        TinyPretrainedModel(),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
            rngs=nn.Rngs(1),
        ),
    )

    model.push_to_hub('owner/adapter')

    upload = FakeHfApi.calls['upload_folder']
    assert upload['delete_patterns'] == [
        'adapter_model.safetensors',
        'adapter_model-*-of-*.safetensors',
        'adapter_model.safetensors.index.json',
    ]
    assert upload['files'] == [
        'adapter_config.json',
        'adapter_model.safetensors',
        'config.json',
    ]
    assert 'create_branch' not in FakeHfApi.calls
