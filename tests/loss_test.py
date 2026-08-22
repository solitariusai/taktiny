import jax
import jax.numpy as jnp
import pytest

from taktiny.trainer.loss import (
    Loss,
    causal_lm_loss,
    cross_entropy_loss,
    dpo_loss,
    focal_loss,
    infonce_loss,
    ipo_loss,
    kl_divergence,
    mae_loss,
    mse_loss,
    smooth_l1_loss,
)


class FixedLogitModel:
    def __init__(self, logits):
        self.logits = logits
        self.call = None

    def __call__(
        self,
        input_ids,
        *,
        attention_mask=None,
        position_ids=None,
        is_causal=False,
        kernel='dot_product',
    ):
        self.call = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'is_causal': is_causal,
            'kernel': kernel,
        }
        return self.logits


def test_loss_preserves_standard_model_batch_contract_by_default():
    batch = {'value': 3}

    loss = Loss(lambda model, received: model + received['value'])

    assert loss(4, batch) == 7


def test_loss_prepares_positional_and_keyword_arguments():
    def prepare(batch):
        return (batch['value'],), {'scale': batch['scale']}

    def calculate(model, value, *, scale):
        return model + value * scale

    loss = Loss(calculate, prepare)

    assert loss(1, {'value': 3, 'scale': 2}) == 7


@pytest.mark.parametrize(
    ('prepare', 'message'),
    [
        (lambda batch: None, r'an \(args, kwargs\) tuple'),
        (lambda batch: ([batch], {}), 'args must be a tuple'),
        (lambda batch: ((batch,), []), 'kwargs must be a mapping'),
        (lambda batch: ((), {1: batch}), 'names must be strings'),
    ],
)
def test_loss_validates_prepared_arguments(prepare, message):
    loss = Loss(lambda model: model, prepare)

    with pytest.raises(TypeError, match=message):
        loss(None, {'value': 1})


def test_cross_entropy_loss_masks_ignored_targets_and_uses_valid_mean():
    logits = jnp.asarray([
        [[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]],
    ])
    labels = jnp.asarray([[0, -100, 1]])

    actual = cross_entropy_loss(logits, labels)
    expected = -jnp.mean(jnp.asarray([
        jax.nn.log_softmax(logits[0, 0])[0],
        jax.nn.log_softmax(logits[0, 2])[1],
    ]))

    assert jnp.allclose(actual, expected)


def test_cross_entropy_loss_empty_mean_is_zero_with_finite_gradient():
    logits = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    labels = jnp.full((1, 2), -100, dtype=jnp.int32)

    value, gradient = jax.value_and_grad(cross_entropy_loss)(logits, labels)

    assert value == 0
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(gradient == 0)


def test_causal_lm_loss_shifts_labels_and_excludes_position_resets():
    logits = jnp.asarray([
        [
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, -4.0, 4.0],
            [0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ])
    model = FixedLogitModel(logits)
    batch = {
        'input_ids': jnp.asarray([[0, 1, 2, 3]]),
        'labels': jnp.asarray([[0, 1, 2, 3]]),
        'position_ids': jnp.asarray([[0, 1, 0, 1]]),
    }

    actual = causal_lm_loss(model, batch)
    expected = cross_entropy_loss(
        logits[:, [0, 2], :],
        jnp.asarray([[1, 3]]),
    )

    assert jnp.allclose(actual, expected)
    assert jnp.array_equal(
        model.call['position_ids'],
        batch['position_ids'],
    )
    assert model.call['is_causal'] is True
    assert model.call['kernel'] == 'dot_product'


def test_causal_lm_loss_converts_padding_mask_for_attention():
    model = FixedLogitModel(jnp.zeros((2, 3, 5)))
    token_mask = jnp.asarray([
        [True, True, False],
        [True, True, True],
    ])
    batch = {
        'input_ids': jnp.ones((2, 3), dtype=jnp.int32),
        'labels': jnp.ones((2, 3), dtype=jnp.int32),
        'attention_mask': token_mask,
    }

    loss = causal_lm_loss(model, batch)

    assert jnp.isfinite(loss)
    assert model.call['attention_mask'].shape == (2, 1, 1, 3)
    assert jnp.array_equal(
        model.call['attention_mask'][:, 0, 0, :],
        token_mask,
    )


@pytest.mark.parametrize(
    'batch',
    [
        {},
        {'input_ids': jnp.ones((1, 2), dtype=jnp.int32)},
    ],
)
def test_causal_lm_loss_requires_inputs_and_labels(batch):
    with pytest.raises(KeyError, match='missing'):
        causal_lm_loss(FixedLogitModel(jnp.zeros((1, 2, 3))), batch)


def _tiny_llama():
    from taktiny import nn, ModelConfig
    from taktiny.maestro.opus.llama import Llama

    config = ModelConfig(
        num_hidden_layers=2,
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        dtype='float32',
    )
    return Llama(config, rngs=nn.Rngs(0), stack_type='stack')


@pytest.mark.parametrize('logits_chunk_size', [1, 3, 7, 32, 100])
def test_chunked_causal_loss_matches_full_loss(logits_chunk_size):
    model = _tiny_llama()
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    batch = {
        'input_ids': jax.random.randint(k1, (2, 64), 0, 512),
        'labels': jax.random.randint(k2, (2, 64), 0, 512),
    }

    full = causal_lm_loss(model, batch)
    chunked = causal_lm_loss(
        model,
        batch,
        logits_chunk_size=logits_chunk_size,
    )

    assert jnp.allclose(chunked, full, atol=1e-5)


def test_chunked_causal_loss_matches_full_with_packing_and_masks():
    model = _tiny_llama()
    key = jax.random.key(1)
    k1, k2 = jax.random.split(key)
    input_ids = jax.random.randint(k1, (2, 40), 0, 512)
    labels = jax.random.randint(k2, (2, 40), 0, 512)
    positions = jnp.concatenate(
        [jnp.arange(15), jnp.arange(25)]
    )[None, :]
    batch = {
        'input_ids': input_ids,
        'labels': labels,
        'position_ids': jnp.broadcast_to(positions, (2, 40)),
        'attention_mask': jnp.ones((2, 40), dtype=jnp.bool_),
    }

    full = causal_lm_loss(model, batch)
    chunked = causal_lm_loss(model, batch, logits_chunk_size=7)

    assert jnp.allclose(chunked, full, atol=1e-5)


def test_chunked_causal_loss_respects_ignored_labels():
    model = _tiny_llama()
    key = jax.random.key(2)
    k1, k2 = jax.random.split(key)
    labels = jax.random.randint(k2, (2, 40), 0, 512)
    labels = labels.at[:, 10:20].set(-100)
    batch = {
        'input_ids': jax.random.randint(k1, (2, 40), 0, 512),
        'labels': labels,
    }

    full = causal_lm_loss(model, batch)
    chunked = causal_lm_loss(model, batch, logits_chunk_size=6)

    assert jnp.allclose(chunked, full, atol=1e-5)


def test_chunked_causal_loss_requires_model_support():
    model = FixedLogitModel(jnp.zeros((1, 2, 3)))
    batch = {
        'input_ids': jnp.ones((1, 2), dtype=jnp.int32),
        'labels': jnp.ones((1, 2), dtype=jnp.int32),
    }

    with pytest.raises(TypeError, match='compute_causal_loss'):
        causal_lm_loss(model, batch, logits_chunk_size=8)


@pytest.mark.parametrize('logits_chunk_size', [0, -3])
def test_chunked_causal_loss_validates_chunk_size(logits_chunk_size):
    model = _tiny_llama()
    batch = {
        'input_ids': jnp.ones((1, 4), dtype=jnp.int32),
        'labels': jnp.ones((1, 4), dtype=jnp.int32),
    }

    with pytest.raises(ValueError, match='logits_chunk_size'):
        causal_lm_loss(model, batch, logits_chunk_size=logits_chunk_size)


def test_causal_lm_loss_accepts_flash_attention_kernel():
    model = _tiny_llama()
    key = jax.random.key(4)
    k1, k2 = jax.random.split(key)
    batch = {
        'input_ids': jax.random.randint(k1, (2, 32), 0, 512),
        'labels': jax.random.randint(k2, (2, 32), 0, 512),
    }

    dot = causal_lm_loss(model, batch)
    flash = causal_lm_loss(model, batch, attention_kernel='flash')
    flash_chunked = causal_lm_loss(
        model,
        batch,
        attention_kernel='flash',
        logits_chunk_size=16,
    )

    assert jnp.allclose(flash, dot, atol=1e-4)
    assert jnp.allclose(flash_chunked, dot, atol=1e-4)


def test_mse_mae_smooth_l1_values():
    prediction = jnp.asarray([1.0, 2.0, 3.0])
    target = jnp.asarray([1.5, 2.0, 4.0])

    assert float(mse_loss(prediction, target)) == pytest.approx(5.0 / 12.0)
    assert float(mae_loss(prediction, target)) == pytest.approx(1.5 / 3.0)
    assert float(smooth_l1_loss(prediction, target, beta=1.0)) == pytest.approx(
        (0.125 + 0.0 + 0.5) / 3.0
    )


def test_regression_losses_validate_shapes():
    with pytest.raises(ValueError, match='equal shapes'):
        mse_loss(jnp.ones((2,)), jnp.ones((3,)))
    with pytest.raises(ValueError, match='beta'):
        smooth_l1_loss(jnp.ones((2,)), jnp.ones((2,)), beta=0)


def test_infonce_positive_first_key():
    query = jnp.asarray([[1.0, 0.0]])
    keys = jnp.asarray([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])

    loss = float(infonce_loss(query, keys, temperature=1.0))
    # loss = -log( exp(1) / (exp(1) + exp(0) + exp(0.5)) )
    expected = -jnp.log(
        jnp.exp(1.0) / (jnp.exp(1.0) + jnp.exp(0.0) + jnp.exp(0.5))
    )
    assert loss == pytest.approx(float(expected))


def test_infonce_respects_positive_mask():
    query = jnp.asarray([[1.0, 0.0]])
    keys = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]]])
    mask = jnp.asarray([[False, True]])

    loss = float(infonce_loss(query, keys, temperature=1.0, positive_mask=mask))
    # positive is the 2nd key (similarity 1); negatives include only the 1st (0).
    expected = -1.0 + jnp.log(jnp.exp(1.0) + jnp.exp(0.0))
    assert loss == pytest.approx(float(expected))


def test_kl_divergence_is_zero_for_identical():
    logits = jnp.asarray([[2.0, 1.0, 0.5]])

    assert float(kl_divergence(logits, logits)) == pytest.approx(0.0)


def test_dpo_and_ipo_values():
    chosen = jnp.asarray([0.8])
    rejected = jnp.asarray([-0.8])
    ref_chosen = jnp.asarray([0.5])
    ref_rejected = jnp.asarray([-0.5])
    # log_ratio = (0.8-0.5) - (-0.8+0.5) = 0.6
    beta = 0.1

    dpo = float(dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=beta))
    assert dpo == pytest.approx(
        float(-jax.nn.log_sigmoid(beta * 0.6))
    )
    ipo = float(ipo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=beta))
    assert ipo == pytest.approx((0.6 - 1.0 / (2.0 * beta)) ** 2)


def test_focal_loss_gamma_zero_matches_ce():
    logits = jnp.asarray([[2.0, 1.0], [0.5, 1.5]])
    labels = jnp.asarray([0, 1])

    focal = float(focal_loss(logits, labels, gamma=0.0))
    ce = float(cross_entropy_loss(logits, labels))
    assert focal == pytest.approx(ce)


def test_focal_loss_masks_ignored():
    logits = jnp.asarray([[2.0, 1.0]])
    labels = jnp.asarray([-100])

    assert float(focal_loss(logits, labels, reduction='mean')) == pytest.approx(
        0.0
    )
