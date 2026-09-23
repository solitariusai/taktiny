import jax.numpy as jnp
import numpy as np
import pytest

from taktiny import nn


class SmallModel(nn.Module):
    def __init__(self, seed: int):
        self.encoder = nn.Linear(2, 3, rngs=nn.Rngs(seed))
        self.layers = (nn.Linear(3, 2, rngs=nn.Rngs(seed + 1)),)


def test_state_dict_include_matches_flat_and_nested_paths():
    model = SmallModel(0)
    include = [r'encoder\.kernel', r'0\.bias']

    assert set(model.flat_state_dict(include=include)) == {'encoder.kernel', '0.bias'}
    nested = model.state_dict(include=include)
    assert set(nested) == {'encoder', '0'}
    assert set(nested['encoder']) == {'kernel'}
    assert set(nested['0']) == {'bias'}
    np.testing.assert_array_equal(nested['encoder']['kernel'], model.encoder.kernel.value)
    assert model.state_dict(include=[]) == {}
    assert model.flat_state_dict(include=[]) == {}


def test_load_state_dict_include_updates_only_selected_parameters():
    source, target = SmallModel(0), SmallModel(10)
    before = target.flat_state_dict()
    include = [r'encoder\.kernel', r'0\.bias']

    target.load_state_dict(source.state_dict(), include=include)

    after = target.flat_state_dict()
    for name in after:
        expected = source.flat_state_dict()[name] if name in {'encoder.kernel', '0.bias'} else before[name]
        np.testing.assert_array_equal(after[name], expected)

    target.load_state_dict({'encoder': {'bias': jnp.ones(3)}}, include=['encoder.bias'])
    bias = target.encoder.bias
    assert bias is not None
    np.testing.assert_array_equal(bias.value, jnp.ones(3))
    target.load_state_dict(
        {'encoder': {'kernel': source.encoder.kernel.value}, '0': None},
        include=['encoder.kernel'],
    )


def test_flat_state_dict_include_with_prefix_and_sparse_load():
    source, target = SmallModel(0), SmallModel(10)
    before = target.flat_state_dict()
    include = [r'model\.encoder\.kernel']
    selected = source.flat_state_dict('model.', include=include)

    assert set(selected) == {'model.encoder.kernel'}
    target.load_flat_state_dict(selected, 'model.', include=include)
    bias = target.encoder.bias
    assert bias is not None
    np.testing.assert_array_equal(target.encoder.kernel.value, source.encoder.kernel.value)
    np.testing.assert_array_equal(bias.value, before['encoder.bias'])
    target.load_flat_state_dict({}, include=['encoder.bias'])
    np.testing.assert_array_equal(bias.value, before['encoder.bias'])


@pytest.mark.parametrize('method', [
    'flat_state_dict', 'state_dict', 'load_flat_state_dict', 'load_state_dict',
])
def test_state_include_validation_and_empty_selection(method):
    model = SmallModel(0)
    state = {} if method.startswith('load_') else None

    def call(include):
        operation = getattr(model, method)
        return operation(state, include=include) if state is not None else operation(include=include)

    with pytest.raises(TypeError, match='sequence'):
        call('encoder.kernel')
    with pytest.raises(TypeError, match='regex strings'):
        call([1])
    with pytest.raises(ValueError, match='matched no'):
        call(['kernel'])
    assert call([]) in ({}, None)
