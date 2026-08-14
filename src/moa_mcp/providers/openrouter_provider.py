"""OpenRouter provider exposing a curated set of frontier models."""

import os

from moa_mcp.models import Model
from moa_mcp.providers.base import Provider, execute_chat

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"

MODELS = [
    Model(id="google/gemini-3.1-pro-preview", name="Gemini 3.1 Pro", modalities=("text", "image")),
    Model(id="google/gemini-3.7-flash", name="Gemini 3.7 Flash", modalities=("text", "image")),
    Model(id="moonshotai/kimi-k3", name="Kimi K3", modalities=("text", "image")),
    Model(id="qwen/qwen3.8-max", name="Qwen 3.8 Max", modalities=("text", "image")),
]


class OpenRouterProvider(Provider):
    """Runs single-turn completions through OpenRouter's OpenAI-compatible API."""

    name = "openrouter"

    def list_models(self) -> list[Model]:
        return list(MODELS)

    def execute(self, model_id: str, prompt: str, attachment_paths: list[str]) -> str:
        self.validate_model(model_id)
        return execute_chat(BASE_URL, _api_key(), model_id, prompt, attachment_paths)


def _api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV_VAR} is not set; cannot call OpenRouter")
    return api_key


PROVIDER = OpenRouterProvider()
