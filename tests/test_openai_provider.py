"""Tests for the OpenAI provider."""

from unittest.mock import patch

import pytest

from moa_mcp.models import Model
from moa_mcp.providers.openai_provider import API_KEY_ENV_VAR, PROVIDER


def test_list_models_returns_the_three_supported_models():
    # Act
    models = PROVIDER.list_models()

    # Assert
    assert models == [
        Model(id="gpt-5.6-sol", name="Sol", modalities=("text", "image")),
        Model(id="gpt-5.6-terra", name="Terra", modalities=("text", "image")),
        Model(id="gpt-5.6-luna", name="Luna", modalities=("text", "image")),
    ]


def test_execute_delegates_to_execute_chat_with_expected_arguments(monkeypatch):
    # Arrange
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")

    # Act
    with patch("moa_mcp.providers.openai_provider.execute_chat") as execute_chat:
        execute_chat.return_value = "ok"
        result = PROVIDER.execute("gpt-5.6-luna", "hello", ["notes.txt"])

    # Assert
    assert result == "ok"
    execute_chat.assert_called_once_with(None, "test-key", "gpt-5.6-luna", "hello", ["notes.txt"])


def test_execute_raises_a_clear_error_when_the_api_key_is_missing(monkeypatch):
    # Arrange
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    # Act / Assert
    with pytest.raises(RuntimeError, match=API_KEY_ENV_VAR):
        PROVIDER.execute("gpt-5.6-luna", "hello", [])


def test_validate_model_rejects_an_unsupported_model_id():
    # Act / Assert
    with pytest.raises(ValueError, match="not supported by openai"):
        PROVIDER.validate_model("gpt-4o")
