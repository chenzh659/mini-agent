"""Unit tests for provider presets."""

from mini_agent.providers import (
    get_provider_preset,
    list_provider_presets,
)


def test_list_provider_presets() -> None:
    presets = list_provider_presets()
    assert len(presets) >= 6
    names = [p.name for p in presets]
    assert "deepseek" in names
    assert "deepseek-r1" in names
    assert "openai" in names
    assert "ollama" in names


def test_get_provider_preset() -> None:
    ds = get_provider_preset("deepseek")
    assert ds is not None
    assert ds.default_model == "deepseek-chat"
    assert "api.deepseek.com" in ds.base_url

    r1 = get_provider_preset("DEEPSEEK-R1")
    assert r1 is not None
    assert r1.default_model == "deepseek-reasoner"

    ollama = get_provider_preset("ollama")
    assert ollama is not None
    assert "11434" in ollama.base_url


def test_get_unknown_provider_preset() -> None:
    unknown = get_provider_preset("non_existent_provider")
    assert unknown is None
