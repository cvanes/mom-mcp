"""Supported model providers, all OpenAI-compatible, and model search."""

import os
from dataclasses import dataclass

from openai import OpenAI

from mom_mcp.attachments import build_user_message
from mom_mcp.models import Model


@dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible provider with a curated set of models."""

    name: str
    base_url: str | None
    api_key_env_var: str
    models: tuple[Model, ...]

    def execute(self, model_id: str, prompt: str, attachment_paths: list[str]) -> str:
        self._validate_model(model_id)
        return execute_chat(self.base_url, self._api_key(), model_id, prompt, attachment_paths)

    def _validate_model(self, model_id: str) -> None:
        if model_id not in {model.id for model in self.models}:
            supported = ", ".join(model.id for model in self.models)
            raise ValueError(
                f"Model '{model_id}' is not supported by {self.name}. Supported: {supported}"
            )

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


OPENAI_PROVIDER = Provider(
    name="openai",
    base_url=None,
    api_key_env_var="OPENAI_API_KEY",
    models=(
        Model(id="gpt-5.6-sol", name="Sol", modalities=("text", "image")),
        Model(id="gpt-5.6-terra", name="Terra", modalities=("text", "image")),
        Model(id="gpt-5.6-luna", name="Luna", modalities=("text", "image")),
    ),
)

OPENROUTER_PROVIDER = Provider(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env_var="OPENROUTER_API_KEY",
    models=(
        Model(id="google/gemini-3.1-pro-preview", name="Gemini 3.1 Pro",
              modalities=("text", "image")),
        Model(id="google/gemini-3.7-flash", name="Gemini 3.7 Flash",
              modalities=("text", "image")),
        Model(id="moonshotai/kimi-k3", name="Kimi K3", modalities=("text", "image")),
        Model(id="qwen/qwen3.8-max", name="Qwen 3.8 Max", modalities=("text", "image")),
    ),
)

PROVIDERS: dict[str, Provider] = {
    provider.name: provider for provider in (OPENAI_PROVIDER, OPENROUTER_PROVIDER)
}


def search_models(query: str) -> list[dict]:
    """Match models across all providers by name, id or provider name, case-insensitively."""
    needle = query.strip().lower()
    return [
        {
            "provider": provider.name,
            "name": model.name,
            "id": model.id,
            "modalities": list(model.modalities),
        }
        for provider in PROVIDERS.values()
        for model in provider.models
        if needle in model.name.lower()
        or needle in model.id.lower()
        or needle in provider.name.lower()
    ]
