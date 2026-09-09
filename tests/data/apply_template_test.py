import pytest

from taktiny.data.transforms import ApplyTemplate


def test_return_key_replaces_output_without_mutating_source():
    source = {'name': 'Ada', 'greeting': 'old'}
    operation = ApplyTemplate('Hello {name}', return_key='greeting')
    assert operation.return_key == 'greeting'
    assert operation(source) == {'name': 'Ada', 'greeting': 'Hello Ada'}
    assert source == {'name': 'Ada', 'greeting': 'old'}


def test_default_return_key():
    assert ApplyTemplate('{name}')({'name': 'Ada'})['template'] == 'Ada'


@pytest.mark.parametrize('return_key', ['', None, 3])
def test_return_key_requires_nonempty_string(return_key):
    with pytest.raises(TypeError, match='return_key'):
        ApplyTemplate('value', return_key=return_key)
