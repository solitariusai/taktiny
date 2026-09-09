# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Takt model transformation and PEFT adapter framework."""

import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.takt import (
    AdaLoRAAdapter,
    DoRAAdapter,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    Takt,
    VeRAAdapter,
)


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 4, hidden: int = 6, out: int = 2):
        self.fc1 = nn.Linear(in_features, hidden, bias=True, rngs=nn.Rngs(0))
        self.fc2 = nn.Linear(hidden, out, bias=False, rngs=nn.Rngs(1))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.relu(self.fc1(x)))


class NestedModel(nn.Module):
    def __init__(self):
        self.encoder = SimpleMLP(4, 6, 4)
        self.head = nn.Linear(4, 2, rngs=nn.Rngs(2))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(self.encoder(x))


def test_apply_lora_freezes_base_and_preserves_forward():
    model = SimpleMLP()
    x = jnp.ones((2, 4), dtype=jnp.float32)
    expected_output = model(x)

    adapter = LoRAAdapter('fc.*', rank=2, alpha=4.0, rngs=nn.Rngs(10))
    adapted = Takt.apply_adapter(model, adapter)

    assert adapted is model
    assert isinstance(model.fc1, nn.LoRALinear)
    assert isinstance(model.fc2, nn.LoRALinear)

    # Base parameters must be frozen
    assert model.fc1.base.kernel.trainable is False
    assert model.fc1.base.bias.trainable is False
    assert model.fc2.base.kernel.trainable is False

    # Adapter parameters must remain trainable
    assert model.fc1.lora_A.kernel.trainable is True
    assert model.fc1.lora_B.kernel.trainable is True
    assert model.fc2.lora_A.kernel.trainable is True
    assert model.fc2.lora_B.kernel.trainable is True

    # Adapter metadata registered on model
    assert hasattr(model, '_takt_adapters')
    assert adapter in model._takt_adapters

    # Initial forward output must match base model exactly (B initialized to 0)
    actual_output = model(x)
    assert jnp.allclose(actual_output, expected_output, atol=1e-5)


@pytest.mark.parametrize(
    ('adapter_cls', 'expected_module_cls'),
    [
        (LoRAAdapter, nn.LoRALinear),
        (DoRAAdapter, nn.DoRALinear),
        (AdaLoRAAdapter, nn.AdaLoRALinear),
        (LoHaAdapter, nn.LoHaLinear),
        (LoKrAdapter, nn.LoKrLinear),
    ],
)
def test_apply_all_low_rank_adapter_families(adapter_cls, expected_module_cls):
    model = SimpleMLP()
    x = jnp.ones((2, 4), dtype=jnp.float32)
    expected_output = model(x)

    adapter = adapter_cls('fc1', rank=2, alpha=4.0, rngs=nn.Rngs(42))
    Takt.apply_adapter(model, adapter)

    assert isinstance(model.fc1, expected_module_cls)
    assert isinstance(model.fc2, nn.Linear)  # unadapted

    # All pre-existing parameters in the model are frozen upon applying PEFT
    assert model.fc1.base.kernel.trainable is False
    assert model.fc2.kernel.trainable is False

    # Forward pass is identical initially
    assert jnp.allclose(model(x), expected_output, atol=1e-5)


def test_apply_vera_shares_projections_across_targets():
    model = SimpleMLP(in_features=4, hidden=6, out=2)
    x = jnp.ones((2, 4), dtype=jnp.float32)
    expected_output = model(x)

    adapter = VeRAAdapter('fc.*', rank=3, rngs=nn.Rngs(99))
    Takt.apply_adapter(model, adapter)

    assert isinstance(model.fc1, nn.VeRALinear)
    assert isinstance(model.fc2, nn.VeRALinear)

    # VeRA projections must be shared between all targets
    assert model.fc1.vera_A is model.fc2.vera_A
    assert model.fc1.vera_B is model.fc2.vera_B

    # Projections themselves are frozen, but scaling vectors are trainable
    assert model.fc1.vera_A.trainable is False
    assert model.fc1.vera_B.trainable is False
    assert model.fc1.vera_lambda_b.trainable is True
    assert model.fc1.vera_lambda_d.trainable is True

    # Sized to maximum input and output dimensions
    # fc1: 4 -> 6, fc2: 6 -> 2 => max_input = 6, max_output = 6
    assert model.fc1.vera_A.shape == (6, 3)
    assert model.fc1.vera_B.shape == (3, 6)

    # Forward output matches base model
    assert jnp.allclose(model(x), expected_output, atol=1e-5)


def test_selective_regex_target_matching():
    model = SimpleMLP()
    adapter = LoRAAdapter('fc1', rank=2, alpha=2.0, rngs=nn.Rngs(1))
    Takt.apply_adapter(model, adapter)

    assert isinstance(model.fc1, nn.LoRALinear)
    assert isinstance(model.fc2, nn.Linear)


def test_nested_module_targeting():
    model = NestedModel()
    adapter = LoRAAdapter(r'.*\.fc1', rank=2, alpha=2.0, rngs=nn.Rngs(5))
    Takt.apply_adapter(model, adapter)

    assert isinstance(model.encoder.fc1, nn.LoRALinear)
    assert isinstance(model.encoder.fc2, nn.Linear)
    assert isinstance(model.head, nn.Linear)


def test_sequential_container_targeting():
    seq = nn.Sequential([
        nn.Linear(4, 6, rngs=nn.Rngs(0)),
        nn.Linear(6, 2, rngs=nn.Rngs(1)),
    ])
    x = jnp.ones((2, 4), dtype=jnp.float32)
    expected = seq(x)

    Takt.apply_adapter(seq, LoRAAdapter('0', rank=2, alpha=4.0, rngs=nn.Rngs(2)))

    assert isinstance(seq.layers[0], nn.LoRALinear)
    assert isinstance(seq.layers[1], nn.Linear)
    assert jnp.allclose(seq(x), expected, atol=1e-5)


def test_update_adapters_hook():
    model = SimpleMLP()
    adapter = LoRAAdapter('fc1', rank=2, alpha=4.0, rngs=nn.Rngs(0))
    Takt.apply_adapter(model, adapter)

    results = Takt.update_adapters(model, step=10)
    assert results == (None,)


def test_gradient_flow_trains_only_adapter_parameters():
    model = SimpleMLP()
    Takt.apply_adapter(model, LoRAAdapter('fc1', rank=2, alpha=4.0, rngs=nn.Rngs(0)))

    x = jnp.ones((2, 4), dtype=jnp.float32)

    def loss_fn(m):
        return jnp.sum(m(x) ** 2)

    grads = jax.grad(loss_fn)(model)

    # Base parameters had trainable=False, so gradients for them are ignored by trainer.
    # lora_B starts with non-zero gradients (while lora_A is scaled by B=0 on step 0)
    assert jnp.any(grads.fc1.lora_B.kernel.value != 0)
    assert jnp.all(jnp.isfinite(grads.fc1.lora_B.kernel.value))


def test_jitted_forward_pass_matches_eager():
    model = SimpleMLP()
    Takt.apply_adapter(model, LoRAAdapter('fc.*', rank=2, alpha=4.0, rngs=nn.Rngs(0)))

    x = jnp.ones((4, 4), dtype=jnp.float32)
    eager_out = model(x)
    jitted_out = jax.jit(model)(x)

    assert jnp.allclose(eager_out, jitted_out, atol=1e-6)


def test_already_applied_adapter_raises_error():
    model = SimpleMLP()
    adapter = LoRAAdapter('fc1', rank=2, rngs=nn.Rngs(0))
    Takt.apply_adapter(model, adapter)

    with pytest.raises(ValueError, match='already applied'):
        Takt.apply_adapter(model, adapter)


def test_no_matched_targets_raises_error():
    model = SimpleMLP()
    adapter = LoRAAdapter('nonexistent_layer', rank=2, rngs=nn.Rngs(0))

    with pytest.raises(ValueError, match='No modules matched'):
        Takt.apply_adapter(model, adapter)


def test_invalid_target_pattern_raises_error():
    with pytest.raises(ValueError, match='Invalid adapter target pattern'):
        LoRAAdapter('[unclosed_regex', rank=2, rngs=nn.Rngs(0))


def test_empty_targets_raises_error():
    with pytest.raises(ValueError, match='targets must contain at least one pattern'):
        LoRAAdapter([], rank=2, rngs=nn.Rngs(0))


def test_invalid_model_type_raises_error():
    adapter = LoRAAdapter('fc1', rank=2, rngs=nn.Rngs(0))
    with pytest.raises(TypeError, match='Adapters require a Taktiny nn.Module'):
        Takt.apply_adapter(object(), adapter)


def test_invalid_adapter_type_raises_error():
    model = SimpleMLP()
    with pytest.raises(TypeError, match='adapter must be a BaseAdapter instance'):
        Takt.apply_adapter(model, object())
