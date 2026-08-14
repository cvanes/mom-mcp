"""Supported model providers, all OpenAI-compatible, and live model search."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from mom_mcp.attachments import build_user_message
from mom_mcp.models import Model


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

    def execute(self, model_id: str, prompt: str, attachment_paths: list[str]) -> str:
        return execute_chat(self.base_url, self._api_key(), model_id, prompt, attachment_paths)

    def _api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env_var} is not set; export it to use the {self.name} provider"
            )
        return api_key


def execute_chat(base_url: str | None, api_key: str, model_id: str,
                 prompt: str, attachment_paths: list[str]) -> str:
    """Run a single-turn chat completion against any OpenAI-compatible API."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    message = build_user_message(prompt, attachment_paths)
    response = client.chat.completions.create(model=model_id, messages=[message])
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
    return Model(id=item.id, name=extra.get("name") or item.id, modalities=modalities or None)


OPENAI_PROVIDER = Provider(
    name="openai",
    base_url=None,
    api_key_env_var="OPENAI_API_KEY",
    parse_model=_parse_openai_model,
)

OPENROUTER_PROVIDER = Provider(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env_var="OPENROUTER_API_KEY",
    parse_model=_parse_openrouter_model,
)

PROVIDERS: dict[str, Provider] = {
    provider.name: provider for provider in (OPENAI_PROVIDER, OPENROUTER_PROVIDER)
}


def search_models(query: str, limit: int) -> list[dict]:
    """Match models across all providers by name, id or provider name, case-insensitively."""
    needle = query.strip().lower()
    matches = [
        {
            "provider": provider.name,
            "name": model.name,
            "id": model.id,
            "modalities": list(model.modalities) if model.modalities else None,
        }
        for provider in PROVIDERS.values()
        for model in provider.list_models()
        if needle in model.name.lower()
        or needle in model.id.lower()
        or needle in provider.name.lower()
    ]
    return matches[:limit]
