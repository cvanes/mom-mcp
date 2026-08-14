"""Tests for model search across providers."""

from mom_mcp.providers import PROVIDERS, search_models


def test_empty_query_returns_every_model_from_all_providers():
    # Act
    results = search_models("")

    # Assert
    expected_total = sum(len(provider.models) for provider in PROVIDERS.values())
    assert len(results) == expected_total
    assert {result["provider"] for result in results} == {"openai", "openrouter"}


def test_matches_model_name_case_insensitively():
    # Act
    results = search_models("GEMINI")

    # Assert
    assert [result["id"] for result in results] == [
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.7-flash",
    ]


def test_matches_provider_name():
    # Act
    results = search_models("openrouter")

    # Assert
    assert {result["provider"] for result in results} == {"openrouter"}


def test_results_include_provider_name_id_and_modalities():
    # Act
    result = search_models("kimi")[0]

    # Assert
    assert result == {
        "provider": "openrouter",
        "name": "Kimi K3",
        "id": "moonshotai/kimi-k3",
        "modalities": ["text", "image"],
    }
