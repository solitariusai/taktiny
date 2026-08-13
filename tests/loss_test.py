import jax
import jax.numpy as jnp
import pytest

from taktiny.cosettes.transformers._ordinario import TransformerContext
from taktiny.trainer.loss import Loss, causal_lm_loss, cross_entropy_loss


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
        ctx=None,
    ):
        self.call = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'ctx': ctx,
        }
        return self.logits, ctx


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
    assert isinstance(model.call['ctx'], TransformerContext)
    assert model.call['ctx'].is_causal is True


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
