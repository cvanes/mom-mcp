"""Tests for the Ollama provider. All network calls are mocked."""

from unittest.mock import patch

import pytest

from moa_mcp.providers import ollama_provider


def test_list_models_includes_expected_tags():
    # Arrange
    provider = ollama_provider.OllamaProvider()

    # Act
    model_ids = {model.id for model in provider.list_models()}

    # Assert
    assert model_ids == {
        "deepseek-v4-flash:cloud",
        "deepseek-v4-pro:cloud",
        "minimax-m3:cloud",
        "glm-5.2:cloud",
        "nemotron-3-ultra:cloud",
    }


def test_execute_derives_base_url_from_ollama_host(monkeypatch):
    # Arrange
    monkeypatch.setenv("OLLAMA_HOST", "http://custom-host:1234")
    provider = ollama_provider.OllamaProvider()

    # Act
    with patch.object(ollama_provider, "execute_chat", return_value="ok") as mock_execute:
        result = provider.execute("glm-5.2:cloud", "hello", [])

    # Assert
    mock_execute.assert_called_once_with(
        "http://custom-host:1234/v1", "ollama", "glm-5.2:cloud", "hello", []
    )
    assert result == "ok"


def test_execute_uses_default_host_when_env_var_absent(monkeypatch):
    # Arrange
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    provider = ollama_provider.OllamaProvider()

    # Act
    with patch.object(ollama_provider, "execute_chat", return_value="ok") as mock_execute:
        provider.execute("glm-5.2:cloud", "hello", [])

    # Assert
    mock_execute.assert_called_once_with(
        "http://localhost:11434/v1", "ollama", "glm-5.2:cloud", "hello", []
    )


def test_execute_rejects_unsupported_model_id():
    # Arrange
    provider = ollama_provider.OllamaProvider()

    # Act / Assert
    with pytest.raises(ValueError, match="not supported"):
        provider.execute("not-a-real-model", "hello", [])
