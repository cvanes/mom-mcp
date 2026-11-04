"""Tests for model search across providers; the fetched catalogues are faked."""

from unittest.mock import patch

import pytest

from mom_mcp import providers
from mom_mcp.models import Cost, Model
from mom_mcp.providers import Provider, search_models

FAKE_CATALOGUES = {
    "openai": [Model(id="gpt-5.6-luna", name="gpt-5.6-luna")],
    "ollama": [Model(id="gemma4:31b", name="gemma4:31b")],
    "openrouter": [
        Model(
            id="google/gemini-3.7-flash",
            name="Gemini 3.7 Flash",
            modalities=("text", "image"),
            cost=Cost(input=0.15, output=0.6),
        ),
        Model(id="moonshotai/kimi-k3", name="Kimi K3", modalities=("text", "image")),
    ],
}


@pytest.fixture(autouse=True)
def fake_catalogues():
    providers._catalogue_cache.clear()
    with patch.object(Provider, "list_models", autospec=True) as list_models:
        list_models.side_effect = lambda self: FAKE_CATALOGUES[self.name]
        yield list_models


def all_ids(result):
    return [model["id"] for model in result["models"]]


def test_empty_query_matches_every_model_from_all_providers():
    # Act
    result = search_models("", limit=50)

    # Assert
    assert all_ids(result) == [
        "gpt-5.6-luna",
        "gemma4:31b",
        "google/gemini-3.7-flash",
        "moonshotai/kimi-k3",
    ]
    assert result["truncated"] is False
    assert result["errors"] == []


def test_matches_model_name_case_insensitively():
    # Act
    result = search_models("GEMINI", limit=50)

    # Assert
    assert all_ids(result) == ["google/gemini-3.7-flash"]


def test_matches_provider_name():
    # Act
    result = search_models("openrouter", limit=50)

    # Assert
    assert {model["provider"] for model in result["models"]} == {"openrouter"}


def test_truncates_results_to_the_limit_and_flags_it():
    # Act
    result = search_models("", limit=2)

    # Assert
    assert len(result["models"]) == 2
    assert result["truncated"] is True


def test_limit_is_shared_across_providers_rather_than_filled_by_the_first(fake_catalogues):
    # Arrange
    openai_models = [Model(id=f"gpt-{n}", name=f"gpt-{n}") for n in range(5)]
    fake_catalogues.side_effect = lambda self: (
        openai_models if self.name == "openai" else FAKE_CATALOGUES[self.name]
    )

    # Act
    result = search_models("", limit=3)

    # Assert
    assert [model["provider"] for model in result["models"]] == [
        "openai",
        "ollama",
        "openrouter",
    ]


def test_results_include_provider_name_id_modalities_and_cost():
    # Act
    model = search_models("gemini", limit=50)["models"][0]

    # Assert
    assert model == {
        "provider": "openrouter",
        "name": "Gemini 3.7 Flash",
        "id": "google/gemini-3.7-flash",
        "modalities": ["text", "image"],
        "cost": {"input": 0.15, "output": 0.6},
    }


def test_modalities_and_cost_are_null_when_the_provider_does_not_report_them():
    # Act
    model = search_models("luna", limit=50)["models"][0]

    # Assert
    assert model["modalities"] is None
    assert model["cost"] is None


def test_a_failing_provider_is_skipped_and_reported_not_fatal(fake_catalogues):
    # Arrange
    def list_models(self):
        if self.name == "openai":
            raise RuntimeError("OPENAI_API_KEY is not set")
        return FAKE_CATALOGUES[self.name]

    fake_catalogues.side_effect = list_models

    # Act
    result = search_models("", limit=50)

    # Assert
    assert all_ids(result) == [
        "gemma4:31b",
        "google/gemini-3.7-flash",
        "moonshotai/kimi-k3",
    ]
    assert result["errors"] == [
        {"provider": "openai", "error": "OPENAI_API_KEY is not set"}
    ]


def test_catalogues_are_cached_across_searches(fake_catalogues):
    # Act
    search_models("kimi", limit=50)
    search_models("gemini", limit=50)

    # Assert
    assert fake_catalogues.call_count == len(FAKE_CATALOGUES)
