"""OpenAI provider backed by the official API."""

import os

from mom_mcp.models import Model
from mom_mcp.providers.base import Provider, execute_chat

API_KEY_ENV_VAR = "OPENAI_API_KEY"

MODELS = [
    Model(id="gpt-5.6-sol", name="Sol", modalities=("text", "image")),
    Model(id="gpt-5.6-terra", name="Terra", modalities=("text", "image")),
    Model(id="gpt-5.6-luna", name="Luna", modalities=("text", "image")),
]


class OpenAIProvider(Provider):
    """Single-turn execution against the OpenAI chat completions API."""

    name = "openai"

    def list_models(self) -> list[Model]:
        return list(MODELS)

    def execute(self, model_id: str, prompt: str, attachment_paths: list[str]) -> str:
        self.validate_model(model_id)
        return execute_chat(None, _api_key(), model_id, prompt, attachment_paths)


def _api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV_VAR} is not set; export it to use the openai provider")
    return api_key


PROVIDER = OpenAIProvider()
