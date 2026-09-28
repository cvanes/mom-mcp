"""Tests for the provider registry, model listing and execution; network access is mocked."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mom_mcp.models import Cost, Model
from mom_mcp.providers import (
    OLLAMA_PROVIDER,
    OPENAI_PROVIDER,
    OPENROUTER_PROVIDER,
    PROVIDERS,
)


def test_registry_exposes_openai_ollama_and_openrouter():
    # Act / Assert
    assert list(PROVIDERS) == ["openai", "ollama", "openrouter"]


def test_execute_delegates_to_execute_chat_with_provider_base_url_and_key(monkeypatch):
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Act
    with patch("mom_mcp.providers.execute_chat") as execute_chat:
        execute_chat.return_value = "ok"
        result = OPENROUTER_PROVIDER.execute("moonshotai/kimi-k3", "hello", ["notes.txt"])

    # Assert
    assert result == "ok"
    execute_chat.assert_called_once_with(
        "https://openrouter.ai/api/v1", "test-key", "moonshotai/kimi-k3", "hello",
        ["notes.txt"], None,
    )


def test_execute_passes_a_system_prompt_through(monkeypatch):
    # Arrange
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Act
    with patch("mom_mcp.providers.execute_chat") as execute_chat:
        OPENAI_PROVIDER.execute("gpt-5.6-luna", "hello", [], "be terse")

    # Assert
    execute_chat.assert_called_once_with(None, "test-key", "gpt-5.6-luna", "hello", [], "be terse")


def test_execute_raises_a_clear_error_when_the_api_key_is_missing(monkeypatch):
    # Arrange
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Act / Assert
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OPENAI_PROVIDER.execute("gpt-5.6-luna", "hello", [])


def test_openai_list_models_excludes_non_chat_models(monkeypatch):
    # Arrange
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Act
    with patch("mom_mcp.providers.OpenAI") as openai_cls:
        openai_cls.return_value.models.list.return_value = [
            SimpleNamespace(id="gpt-5.6-luna"),
            SimpleNamespace(id="text-embedding-4"),
            SimpleNamespace(id="whisper-2"),
        ]
        models = OPENAI_PROVIDER.list_models()

    # Assert
    assert models == [Model(id="gpt-5.6-luna", name="gpt-5.6-luna")]


def test_ollama_list_models_excludes_embedding_models(monkeypatch):
    # Arrange
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

    # Act
    with patch("mom_mcp.providers.OpenAI") as openai_cls:
        openai_cls.return_value.models.list.return_value = [
            SimpleNamespace(id="gemma4:31b"),
            SimpleNamespace(id="embeddinggemma"),
        ]
        models = OLLAMA_PROVIDER.list_models()

    # Assert
    assert models == [Model(id="gemma4:31b", name="gemma4:31b")]


def test_openrouter_list_models_maps_name_modalities_and_cost(monkeypatch):
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Act
    with patch("mom_mcp.providers.OpenAI") as openai_cls:
        openai_cls.return_value.models.list.return_value = [
            SimpleNamespace(
                id="moonshotai/kimi-k3",
                model_extra={
                    "name": "Kimi K3",
                    "architecture": {"input_modalities": ["text", "image"]},
                    "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
                },
            ),
        ]
        models = OPENROUTER_PROVIDER.list_models()

    # Assert
    assert models == [
        Model(
            id="moonshotai/kimi-k3",
            name="Kimi K3",
            modalities=("text", "image"),
            cost=Cost(input=0.4, output=1.6),
        )
    ]


def test_openrouter_cost_is_none_when_pricing_is_missing(monkeypatch):
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Act
    with patch("mom_mcp.providers.OpenAI") as openai_cls:
        openai_cls.return_value.models.list.return_value = [
            SimpleNamespace(id="acme/free-model", model_extra={"name": "Free Model"}),
        ]
        models = OPENROUTER_PROVIDER.list_models()

    # Assert
    assert models == [Model(id="acme/free-model", name="Free Model", cost=None)]
