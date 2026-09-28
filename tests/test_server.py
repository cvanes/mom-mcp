"""Tests for the MCP tools themselves; provider execution is mocked."""

from unittest.mock import patch

import pytest

from mom_mcp import server


def test_execute_prompt_rejects_an_unknown_provider():
    # Act / Assert
    with pytest.raises(ValueError, match="Unknown provider 'acme'"):
        server.execute_prompt("acme", "some-model", "hello")


def test_execute_prompt_delegates_to_the_provider():
    # Act
    with patch.dict(server.PROVIDERS) as providers:
        providers["openai"] = provider = _FakeProvider()
        result = server.execute_prompt(
            "openai", "gpt-5.6-luna", "hello", system="be terse",
            attachments=["notes.txt"],
        )

    # Assert
    assert result == "fake response"
    assert provider.calls == [
        ("gpt-5.6-luna", "hello", ["notes.txt"], "be terse"),
    ]


def test_get_second_opinion_wraps_the_work_and_applies_the_reviewer_prompt():
    # Act
    with patch.dict(server.PROVIDERS) as providers:
        providers["openai"] = provider = _FakeProvider()
        result = server.get_second_opinion(
            "openai", "gpt-5.6-luna", "my draft essay", focus="argument structure",
            attachments=["draft.md"],
        )

    # Assert
    assert result == "fake response"
    model_id, prompt, attachments, system = provider.calls[0]
    assert model_id == "gpt-5.6-luna"
    assert prompt == (
        "<work>\nmy draft essay\n</work>\n\n"
        "Focus your review on: argument structure"
    )
    assert attachments == ["draft.md"]
    assert system == server._REVIEWER_SYSTEM_PROMPT


def test_get_second_opinion_omits_the_focus_line_when_no_focus_is_given():
    # Act
    with patch.dict(server.PROVIDERS) as providers:
        providers["openai"] = provider = _FakeProvider()
        server.get_second_opinion("openai", "gpt-5.6-luna", "my draft essay")

    # Assert
    assert provider.calls[0][1] == "<work>\nmy draft essay\n</work>"


class _FakeProvider:
    def __init__(self):
        self.calls = []

    def execute(self, model_id, prompt, attachment_paths, system=None):
        self.calls.append((model_id, prompt, attachment_paths, system))
        return "fake response"
