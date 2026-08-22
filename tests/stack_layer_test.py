import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.cosettes.transformers.ordinario import StackLayer


class _Layer(nn.Module):
    def __init__(self, *, layer_idx: int) -> None:
        self.weight = nn.Parameter(jnp.asarray(layer_idx + 1.0))

    def __call__(self, value):
        return value + self.weight


def _run(stack_type):
    layers = StackLayer.init_stack(
        _Layer,
        num_stacks=3,
        stack_type=stack_type,
    )

    def forward(layer, carry, layer_value):
        carry = layer(carry) + layer_value
        return carry, carry

    return StackLayer.call_stack(
        layers,
        forward,
        jnp.asarray(0.0),
        per_layer=jnp.asarray([10.0, 20.0, 30.0]),
    )


def test_stack_layer_list_and_scan_have_equal_contracts():
    listed_carry, listed_outputs = _run('list')
    scanned_carry, scanned_outputs = _run('stack')

    assert jnp.allclose(listed_carry, scanned_carry)
    assert jnp.allclose(listed_outputs, scanned_outputs)
    assert jnp.allclose(scanned_outputs, jnp.asarray([11.0, 33.0, 66.0]))


@pytest.mark.parametrize('stack_type', ['list', 'stack'])
def test_stack_layer_accepts_no_per_layer_inputs(stack_type):
    layers = StackLayer.init_stack(
        _Layer,
        num_stacks=2,
        stack_type=stack_type,
    )

    def forward(layer, carry, layer_value):
        assert layer_value is None
        return layer(carry), None

    carry, outputs = StackLayer.call_stack(
        layers,
        forward,
        jnp.asarray(0.0),
    )

    assert jnp.allclose(carry, 3.0)
    assert outputs is None


def test_stack_layer_validates_per_layer_leading_axis():
    layers = StackLayer.init_stack(
        _Layer,
        num_stacks=2,
        stack_type='stack',
    )

    with pytest.raises(ValueError, match='leading axes'):
        StackLayer.call_stack(
            layers,
            lambda layer, carry, value: (layer(carry), value),
            jnp.asarray(0.0),
            per_layer=jnp.zeros((3, 1)),
        )
