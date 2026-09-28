"""Supported model providers, all OpenAI-compatible, and live model search."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, TypedDict

from openai import OpenAI

from mom_mcp.attachments import build_user_message
from mom_mcp.models import Cost, Model

_CATALOGUE_TTL = timedelta(minutes=10)


@dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible provider whose model catalogue is fetched from its API."""

    name: str
    base_url: str | None
    api_key_env_var: str
    parse_model: Callable[[Any], Model | None]

    def list_models(self) -> list[Model]:
        client = OpenAI(base_url=self.base_url, api_key=self._api_key())
        parsed = (self.parse_model(item) for item in client.models.list())
        return [model for model in parsed if model is not None]

    def execute(self, model_id: str, prompt: str, attachment_paths: list[str],
                system: str | None = None) -> str:
        return execute_chat(
            self.base_url, self._api_key(), model_id, prompt, attachment_paths, system
        )

    def _api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env_var} is not set; export it to use the {self.name} provider"
            )
        return api_key


def execute_chat(base_url: str | None, api_key: str, model_id: str,
                 prompt: str, attachment_paths: list[str],
                 system: str | None = None) -> str:
    """Run a single-turn chat completion against any OpenAI-compatible API."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(build_user_message(prompt, attachment_paths))
    response = client.chat.completions.create(model=model_id, messages=messages)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError(f"Model {model_id} returned an empty response")
    return content


# The OpenAI models endpoint does not say which endpoint a model serves, so
# models that cannot handle chat completions are excluded by id keywords.
_OPENAI_NON_CHAT_MARKERS = (
    "embedding", "tts", "whisper", "dall-e", "moderation", "realtime",
    "audio", "transcribe", "image", "davinci", "babbage", "computer-use", "sora",
)


def _parse_openai_model(item: Any) -> Model | None:
    if any(marker in item.id for marker in _OPENAI_NON_CHAT_MARKERS):
        return None
    return Model(id=item.id, name=item.id)


def _parse_openrouter_model(item: Any) -> Model | None:
    extra = item.model_extra or {}
    architecture = extra.get("architecture") or {}
    modalities = tuple(architecture.get("input_modalities") or ())
    return Model(
        id=item.id,
        name=extra.get("name") or item.id,
        modalities=modalities or None,
        cost=_parse_openrouter_cost(extra.get("pricing") or {}),
    )


def _parse_openrouter_cost(pricing: dict) -> Cost | None:
    """Map OpenRouter's per-token USD strings onto per-million-token floats."""
    try:
        return Cost(
            input=round(float(pricing["prompt"]) * 1_000_000, 6),
            output=round(float(pricing["completion"]) * 1_000_000, 6),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_ollama_model(item: Any) -> Model | None:
    if "embed" in item.id:
        return None
    return Model(id=item.id, name=item.id)


OPENAI_PROVIDER = Provider(
    name="openai",
    base_url=None,
    api_key_env_var="OPENAI_API_KEY",
    parse_model=_parse_openai_model,
)

OLLAMA_PROVIDER = Provider(
    name="ollama",
    base_url="https://ollama.com/v1",
    api_key_env_var="OLLAMA_API_KEY",
    parse_model=_parse_ollama_model,
)

OPENROUTER_PROVIDER = Provider(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env_var="OPENROUTER_API_KEY",
    parse_model=_parse_openrouter_model,
)

PROVIDERS: dict[str, Provider] = {
    provider.name: provider
    for provider in (OPENAI_PROVIDER, OLLAMA_PROVIDER, OPENROUTER_PROVIDER)
}


class ModelMatch(TypedDict):
    provider: str
    name: str
    id: str
    modalities: list[str] | None
    cost: dict[str, float] | None


class ProviderError(TypedDict):
    provider: str
    error: str


class SearchResult(TypedDict):
    models: list[ModelMatch]
    truncated: bool
    errors: list[ProviderError]


_catalogue_cache: dict[str, tuple[datetime, list[Model]]] = {}


def _catalogue(provider: Provider) -> list[Model]:
    """Return a provider's model catalogue, cached for a short TTL."""
    cached = _catalogue_cache.get(provider.name)
    if cached and datetime.now() - cached[0] < _CATALOGUE_TTL:
        return cached[1]
    models = provider.list_models()
    _catalogue_cache[provider.name] = (datetime.now(), models)
    return models


def search_models(query: str, limit: int) -> SearchResult:
    """Match models across all providers by name, id or provider name, case-insensitively.

    A provider that fails (missing API key, network error, ...) is skipped and
    reported in errors so that the remaining providers can still be searched.
    """
    needle = query.strip().lower()
    matches: list[ModelMatch] = []
    errors: list[ProviderError] = []
    for provider in PROVIDERS.values():
        try:
            catalogue = _catalogue(provider)
        except Exception as exc:
            errors.append({"provider": provider.name, "error": str(exc)})
            continue
        matches += [
            _to_match(provider, model)
            for model in catalogue
            if needle in model.name.lower()
            or needle in model.id.lower()
            or needle in provider.name.lower()
        ]
    return {
        "models": matches[:limit],
        "truncated": len(matches) > limit,
        "errors": errors,
    }


def _to_match(provider: Provider, model: Model) -> ModelMatch:
    return {
        "provider": provider.name,
        "name": model.name,
        "id": model.id,
        "modalities": list(model.modalities) if model.modalities else None,
        "cost": (
            {"input": model.cost.input, "output": model.cost.output}
            if model.cost
            else None
        ),
    }
