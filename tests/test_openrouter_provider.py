"""Tests for the OpenRouter provider; all network access is mocked."""

from unittest.mock import patch

import pytest

from mom_mcp.providers.openrouter_provider import BASE_URL, PROVIDER


def test_lists_the_four_supported_models():
    # Arrange / Act
    models = PROVIDER.list_models()

    # Assert
    assert [model.id for model in models] == [
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.7-flash",
        "moonshotai/kimi-k3",
        "qwen/qwen3.8-max",
    ]


def test_execute_delegates_with_openrouter_base_url_and_key(monkeypatch):
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Act
    with patch("mom_mcp.providers.openrouter_provider.execute_chat") as execute_chat:
        execute_chat.return_value = "ok"
        result = PROVIDER.execute("moonshotai/kimi-k3", "hello", [])

    # Assert
    assert result == "ok"
    execute_chat.assert_called_once_with(BASE_URL, "test-key", "moonshotai/kimi-k3", "hello", [])


def test_execute_without_api_key_raises_clear_error(monkeypatch):
    # Arrange
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # Act / Assert
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        PROVIDER.execute("moonshotai/kimi-k3", "hello", [])


def test_execute_rejects_unsupported_model(monkeypatch):
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Act / Assert
    with pytest.raises(ValueError, match="is not supported by openrouter"):
        PROVIDER.execute("openai/gpt-4o", "hello", [])
