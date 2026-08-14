"""Tests for model search across providers; the fetched catalogues are faked."""

from unittest.mock import patch

import pytest

from mom_mcp.models import Model
from mom_mcp.providers import Provider, search_models

FAKE_CATALOGUES = {
    "openai": [Model(id="gpt-5.6-luna", name="gpt-5.6-luna")],
    "openrouter": [
        Model(id="google/gemini-3.7-flash", name="Gemini 3.7 Flash", modalities=("text", "image")),
        Model(id="moonshotai/kimi-k3", name="Kimi K3", modalities=("text", "image")),
    ],
}


@pytest.fixture(autouse=True)
def fake_catalogues():
    with patch.object(Provider, "list_models", autospec=True) as list_models:
        list_models.side_effect = lambda self: FAKE_CATALOGUES[self.name]
        yield


def test_empty_query_matches_every_model_from_all_providers():
    # Act
    results = search_models("", limit=50)

    # Assert
    assert [result["id"] for result in results] == [
        "gpt-5.6-luna",
        "google/gemini-3.7-flash",
        "moonshotai/kimi-k3",
    ]


def test_matches_model_name_case_insensitively():
    # Act
    results = search_models("GEMINI", limit=50)

    # Assert
    assert [result["id"] for result in results] == ["google/gemini-3.7-flash"]


def test_matches_provider_name():
    # Act
    results = search_models("openrouter", limit=50)

    # Assert
    assert {result["provider"] for result in results} == {"openrouter"}


def test_truncates_results_to_the_limit():
    # Act
    results = search_models("", limit=2)

    # Assert
    assert len(results) == 2


def test_results_include_provider_name_id_and_modalities():
    # Act
    result = search_models("kimi", limit=50)[0]

    # Assert
    assert result == {
        "provider": "openrouter",
        "name": "Kimi K3",
        "id": "moonshotai/kimi-k3",
        "modalities": ["text", "image"],
    }


def test_modalities_are_null_when_the_provider_does_not_report_them():
    # Act
    result = search_models("luna", limit=50)[0]

    # Assert
    assert result["modalities"] is None
