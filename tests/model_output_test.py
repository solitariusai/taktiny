import jax
import jax.numpy as jnp
import pytest

from taktiny.cosettes.overture import ModelOutput


def test_model_output_supports_mapping_and_attribute_access():
    logits = jnp.ones((2, 3))
    output = ModelOutput(logits=logits, kv_cache=None)

    assert output.logits is logits
    assert output['logits'] is logits
    assert output.kv_cache is None
    assert tuple(output) == ('logits', 'kv_cache')


def test_model_output_is_a_jax_pytree_and_blocks_direct_assignment():
    @jax.jit
    def apply(value):
        return ModelOutput(result=value * 2, auxiliary=value + 1)

    output = apply(jnp.asarray(3.0))

    assert jnp.array_equal(output.result, jnp.asarray(6.0))
    assert jnp.array_equal(output.auxiliary, jnp.asarray(4.0))
    with pytest.raises(AttributeError, match='cannot be assigned directly'):
        output.result = jnp.asarray(0.0)


def test_model_output_pop_removes_and_returns_a_field():
    output = ModelOutput(x=jnp.asarray(3.0), kv_cache=None)

    x = output.pop('x')

    assert jnp.array_equal(x, jnp.asarray(3.0))
    assert tuple(output) == ('kv_cache',)
    assert ModelOutput(logits=x, **output).kv_cache is None
    with pytest.raises(KeyError):
        output.pop('x')
    assert output.pop('missing', None) is None
