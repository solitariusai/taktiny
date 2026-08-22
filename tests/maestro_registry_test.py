from taktiny import Maestro
from taktiny.maestro.livret import repertoire


def test_registry_contains_known_architectures():
    assert 'LlamaForCausalLM' in repertoire
    assert 'Qwen2ForCausalLM' in repertoire
    assert 'GemmaForCausalLM' in repertoire


def test_registry_rejects_unknown_architectures():
    assert 'NotARealArchitecture' not in repertoire


def test_is_supported_agrees_with_registry():
    assert Maestro.is_supported('LlamaForCausalLM')
    assert not Maestro.is_supported('NotARealArchitecture')


def test_available_is_sorted_and_unique():
    available = Maestro.available()

    assert available
    assert available == sorted(available)
    assert len(available) == len(set(available))
