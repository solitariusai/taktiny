import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taktiny import nn, ModelConfig
from taktiny.maestro.opus.llama import Llama
from taktiny.trainer import Trainer, TrainingConfig
from taktiny.trainer.sft import SFTDatasetConfig, SFTTrainer


class MockTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        ids = [ord(c) % 40 + 1 for c in text]
        if max_length:
            ids = ids[:max_length]
        return {'input_ids': np.asarray([ids], dtype=np.int32)}


def _text_rows():
    return [
        {'text': t}
        for t in [
            'hello world',
            'the quick brown fox',
            'jumps over',
            'the lazy dog',
            'end',
        ]
    ]


@pytest.mark.parametrize('packing', [False, True])
def test_sft_dataloader_builds_packed_and_unpacked(packing):
    cfg = SFTDatasetConfig(
        dataset=_text_rows(),
        tokenizer=MockTokenizer(),
        text_field='text',
        max_length=8,
        packing=packing,
        batch_size=2,
        epochs=2,
        seed=0,
    )

    dataloader = cfg.build()
    batches = list(dataloader)
    first = batches[0]

    assert np.asarray(first['input_ids']).shape == (2, 8)
    assert np.asarray(first['labels']).shape == (2, 8)
    assert 'attention_mask' in first
    if packing:
        assert 'position_ids' in first
    else:
        assert 'position_ids' not in first


def test_sft_dataloader_dataloader_source_passes_through():
    raw = _text_rows()
    cfg = SFTDatasetConfig(dataloader=raw)

    assert cfg.build() is raw


def test_sft_trainer_dataloader_source_trains_like_plain_trainer():
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

    class TinyModel(nn.Module):
        def __init__(self):
            self.weight = nn.Parameter(jnp.asarray(0.0))

    def squared_error(model, batch):
        prediction = model.weight.value * batch['x']
        return jnp.mean((prediction - batch['y']) ** 2)

    model = TinyModel()

    trainer = SFTTrainer(
        model,
        training_config=TrainingConfig(
            max_steps=2,
            learning_rate=0.1,
            log_interval=1,
        ),
        dataset_config=SFTDatasetConfig(dataloader=batches),
        loss_fn=squared_error,
    )

    trainer.train()

    assert float(model.weight.value) != 0.0


def test_sft_trainer_end_to_end_causal_loss():
    config = ModelConfig(
        num_hidden_layers=2,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        dtype='float32',
    )
    model = Llama(config, rngs=nn.Rngs(0), stack_type='stack')

    trainer = SFTTrainer(
        model,
        training_config=TrainingConfig(
            max_steps=1,
            learning_rate=0.1,
            log_interval=1,
        ),
        dataset_config=SFTDatasetConfig(
            dataset=_text_rows(),
            tokenizer=MockTokenizer(),
            text_field='text',
            max_length=16,
            packing=True,
            batch_size=2,
            epochs=2,
            seed=0,
        ),
    )

    trainer.train()

    embedding = model.model.token_embedding.embedding.value
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(embedding))))


class MockChatTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        conversations,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
    ):
        out = []
        for convo in conversations:
            ids = [1, 2, 3]  # user
            for message in convo:
                if message['role'] == 'assistant':
                    ids += [4, 5, 6]
            if add_generation_prompt:
                ids = ids + [9]
            out.append(ids)
        return out


def test_sft_assistant_only_masks_prompt_tokens():
    data = [
        {'messages': [
            {'role': 'user', 'content': 'u'},
            {'role': 'assistant', 'content': 'a'},
        ]},
        {'messages': [
            {'role': 'user', 'content': 'u'},
            {'role': 'assistant', 'content': 'b'},
        ]},
    ]
    cfg = SFTDatasetConfig(
        dataset=data,
        tokenizer=MockChatTokenizer(),
        assistant_only=True,
        max_length=8,
        packing=False,
        batch_size=2,
        seed=0,
    )

    labels = np.asarray(list(cfg.build())[0]['labels'])

    # prompt (1,2,3) masked; assistant (4,5,6) real; padding masked.
    np.testing.assert_array_equal(
        labels[0],
        np.asarray([-100, -100, -100, 4, 5, 6, -100, -100]),
    )


def test_sft_streaming_batches_text_field():
    cfg = SFTDatasetConfig(
        dataset=_text_rows(),
        tokenizer=MockTokenizer(),
        text_field='text',
        max_length=8,
        batch_size=2,
        seed=0,
    )

    batches = list(cfg._streaming_iterable(iter(_text_rows())))

    assert len(batches) == 3
    assert np.asarray(batches[0]['input_ids']).shape == (2, 8)


def test_sft_streaming_rejects_packing():
    cfg = SFTDatasetConfig(
        dataset=_text_rows(),
        tokenizer=MockTokenizer(),
        text_field='text',
        max_length=8,
        packing=True,
        batch_size=2,
        seed=0,
    )

    with pytest.raises(NotImplementedError, match='packing'):
        list(cfg._streaming_iterable(iter(_text_rows())))
