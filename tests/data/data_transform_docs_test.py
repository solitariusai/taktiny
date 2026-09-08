"""Keep every public data transform's documentation examples executable."""

import doctest
import inspect

import pytest

from taktiny.data import transforms


@pytest.mark.parametrize('name', transforms.__all__)
def test_public_transform_docstring_examples(name):
    transform = getattr(transforms, name)
    doc = inspect.getdoc(transform)
    assert doc is not None
    assert '_summary_' not in doc
    assert '_description_' not in doc
    assert 'Args:' in doc
    tests = doctest.DocTestFinder().find(transform, name, globs=vars(transforms))
    assert any(test.examples for test in tests), f'{name} needs a usage example'
    runner = doctest.DocTestRunner()
    for test in tests:
        runner.run(test)
    assert runner.summarize().failed == 0
