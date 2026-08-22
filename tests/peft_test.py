import jax.numpy as jnp
import json
import numpy as np
import pytest
import qwix
from safetensors.numpy import save_file

from taktiny import Takt, nn
from taktiny.takt import prelude as takt_prelude
from taktiny.cosettes.overture import PretrainedModel
from taktiny.peft import LoraConfig


class TinyModel(PretrainedModel):
    def __init__(self, dtype=jnp.float32):
        self.proj = nn.Linear(
            4,
            3,
            bias=True,
            dtype=dtype,
            rngs=nn.Rngs(0),
        )

    def __call__(self, x):
        return self.proj(x)


class TinyBlock(nn.Module):
    def __init__(self, seed):
        self.proj = nn.Linear(
            4,
            3,
            bias=False,
            rngs=nn.Rngs(seed),
        )


class TinyStackedModel(PretrainedModel):
    def __init__(self):
        self.layers = nn.SeqStack(
            TinyBlock(seed)
            for seed in range(2)
        )


def _adapted_model(dtype=jnp.float32):
    model = Takt.apply_peft(
        TinyModel(dtype),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
            rngs=nn.Rngs(1),
        ),
    )
    model.proj.lora_B.value = jnp.arange(
        6,
        dtype=jnp.float32,
    ).reshape(2, 3) / 10
    return model


def test_merge_peft_preserves_lora_forward_and_restores_linear():
    model = _adapted_model()
    inputs = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    expected = model(inputs)

    result = Takt.merge_peft(model, dtype='float32')

    assert result is model
    assert isinstance(model.proj, nn.Linear)
    assert not hasattr(model, 'peft_config')
    assert not hasattr(model, '_peft_trainable_state')
    assert model.proj.weight.trainable
    assert model.proj.bias.trainable
    assert model.proj.weight.dtype == jnp.float32
    assert jnp.allclose(model(inputs), expected, rtol=1e-6, atol=1e-6)


def test_merge_peft_casts_dense_output_dtype():
    model = _adapted_model()

    Takt.merge_peft(model, dtype='bfloat16')

    assert model.proj.weight.dtype == jnp.bfloat16
    assert not isinstance(model.proj.weight.value, qwix.QArray)


def test_merge_peft_can_requantize_output():
    model = _adapted_model(dtype=jnp.bfloat16)

    Takt.merge_peft(
        model,
        dtype='bfloat16',
        quant='int4',
    )

    assert isinstance(model.proj, nn.Linear)
    assert isinstance(model.proj.weight.value, qwix.QArray)
    assert model.proj.weight.value.qtype == 'int4'


def test_merge_peft_dequantizes_quantized_base_by_default():
    model = TinyModel(dtype=jnp.bfloat16)
    model.proj.weight.value = qwix.quantize(
        model.proj.weight.value,
        'int4',
        channelwise_axes=(1,),
        scale_dtype=jnp.bfloat16,
    )
    model = Takt.apply_peft(
        model,
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
            rngs=nn.Rngs(1),
        ),
    )

    Takt.merge_peft(model)

    assert isinstance(model.proj, nn.Linear)
    assert not isinstance(model.proj.weight.value, qwix.QArray)
    assert model.proj.weight.dtype == jnp.bfloat16


def test_merge_peft_handles_seqstack_layer_axis():
    model = Takt.apply_peft(
        TinyStackedModel(),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
            rngs=nn.Rngs(1),
        ),
    )
    adapter = model.layers.stacked.proj
    adapter.lora_B.value = jnp.arange(
        12,
        dtype=jnp.float32,
    ).reshape(2, 2, 3) / 10
    expected = adapter.base_layer.weight.value + (
        jnp.matmul(adapter.lora_A.value, adapter.lora_B.value)
        * adapter.scaling
    )

    Takt.merge_peft(model)

    projection = model.layers.stacked.proj
    assert isinstance(projection, nn.Linear)
    assert projection.weight.shape == (2, 4, 3)
    assert jnp.allclose(
        projection.weight.value,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_merge_peft_rejects_non_floating_dense_dtype():
    model = _adapted_model()

    with pytest.raises(TypeError, match='must be floating-point'):
        Takt.merge_peft(model, dtype='int8')


def test_merge_peft_requires_adapter_modules():
    with pytest.raises(ValueError, match='No mergeable PEFT'):
        Takt.merge_peft(TinyModel())


def test_load_peft_round_trips_local_adapter(tmp_path):
    source = _adapted_model()
    inputs = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    expected = source(inputs)
    source.save_pretrained(tmp_path)

    model = TinyModel()
    result = Takt.load_peft(model, tmp_path)

    assert result is model
    assert isinstance(model.proj, nn.LoRALinear)
    assert jnp.array_equal(
        model.proj.lora_A.value,
        source.proj.lora_A.value,
    )
    assert jnp.array_equal(
        model.proj.lora_B.value,
        source.proj.lora_B.value,
    )
    assert jnp.allclose(model(inputs), expected, rtol=1e-6, atol=1e-6)
    assert model.proj.base_layer.weight.trainable is False
    assert model.proj.lora_A.trainable is True


def test_load_peft_reconstructs_sharded_seqstack_adapter(tmp_path):
    source = Takt.apply_peft(
        TinyStackedModel(),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
            rngs=nn.Rngs(1),
        ),
    )
    source.layers.stacked.proj.lora_B.value = jnp.arange(
        12,
        dtype=jnp.float32,
    ).reshape(2, 2, 3) / 10
    source.save_pretrained(tmp_path, max_shard_size=56)

    model = TinyStackedModel()
    Takt.load_peft(model, tmp_path)

    assert jnp.array_equal(
        model.layers.stacked.proj.lora_A.value,
        source.layers.stacked.proj.lora_A.value,
    )
    assert jnp.array_equal(
        model.layers.stacked.proj.lora_B.value,
        source.layers.stacked.proj.lora_B.value,
    )


def test_load_peft_populates_existing_lora_wrappers(tmp_path):
    source = _adapted_model()
    source.save_pretrained(tmp_path)
    model = Takt.apply_peft(
        TinyModel(),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
            rngs=nn.Rngs(9),
        ),
    )

    Takt.load_peft(model, tmp_path)

    assert jnp.array_equal(
        model.proj.lora_A.value,
        source.proj.lora_A.value,
    )
    assert jnp.array_equal(
        model.proj.lora_B.value,
        source.proj.lora_B.value,
    )


def test_load_peft_rejects_incomplete_adapter(tmp_path):
    with (tmp_path / 'adapter_config.json').open('w') as config_file:
        json.dump(
            {
                'peft_type': 'LORA',
                'target_modules': ['proj'],
                'rank': 2,
                'alpha': 4,
            },
            config_file,
        )
    save_file(
        {'proj.lora_A': np.ones((4, 2), dtype=np.float32)},
        tmp_path / 'adapter_model.safetensors',
    )

    with pytest.raises(ValueError, match='missing tensors'):
        Takt.load_peft(TinyModel(), tmp_path)


def test_load_peft_rejects_incompatible_existing_wrapper(tmp_path):
    source = _adapted_model()
    source.save_pretrained(tmp_path)
    model = Takt.apply_peft(
        TinyModel(),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=8,
            rngs=nn.Rngs(9),
        ),
    )

    with pytest.raises(ValueError, match='adapter requires'):
        Takt.load_peft(model, tmp_path)


def test_load_peft_downloads_sharded_hub_adapter(
    tmp_path,
    monkeypatch,
):
    source = Takt.apply_peft(
        TinyStackedModel(),
        LoraConfig(
            target_modules='proj',
            rank=2,
            alpha=4,
            rngs=nn.Rngs(1),
        ),
    )
    source.save_pretrained(tmp_path, max_shard_size=56)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs['filename'])

    monkeypatch.setattr(
        takt_prelude,
        'hf_hub_download',
        fake_download,
    )
    model = TinyStackedModel()

    Takt.load_peft(
        model,
        'owner/adapter',
        local=False,
        token='token',
        revision='experiment',
        subfolder='lora',
    )

    assert [call['filename'] for call in calls] == [
        'adapter_config.json',
        'adapter_model.safetensors.index.json',
        'adapter_model-00001-of-00002.safetensors',
        'adapter_model-00002-of-00002.safetensors',
    ]
    assert all(call['repo_id'] == 'owner/adapter' for call in calls)
    assert all(call['token'] == 'token' for call in calls)
    assert all(call['revision'] == 'experiment' for call in calls)
    assert all(call['subfolder'] == 'lora' for call in calls)
    assert jnp.array_equal(
        model.layers.stacked.proj.lora_A.value,
        source.layers.stacked.proj.lora_A.value,
    )
