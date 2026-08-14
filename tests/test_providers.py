"""Tests for the provider registry and single-turn execution; network access is mocked."""

from unittest.mock import patch

import pytest

from mom_mcp.providers import OPENAI_PROVIDER, OPENROUTER_PROVIDER, PROVIDERS


def test_registry_exposes_openai_and_openrouter():
    # Act / Assert
    assert list(PROVIDERS) == ["openai", "openrouter"]


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
        "https://openrouter.ai/api/v1", "test-key", "moonshotai/kimi-k3", "hello", ["notes.txt"]
    )


def test_execute_raises_a_clear_error_when_the_api_key_is_missing(monkeypatch):
    # Arrange
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Act / Assert
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OPENAI_PROVIDER.execute("gpt-5.6-luna", "hello", [])


def test_execute_rejects_an_unsupported_model_id(monkeypatch):
    # Arrange
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Act / Assert
    with pytest.raises(ValueError, match="not supported by openai"):
        OPENAI_PROVIDER.execute("gpt-4o", "hello", [])
