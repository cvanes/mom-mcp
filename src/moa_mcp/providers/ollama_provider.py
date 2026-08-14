"""Ollama provider: proxies Ollama cloud models via its OpenAI-compatible API."""

import os

from moa_mcp.models import Model
from moa_mcp.providers.base import Provider, execute_chat

_DEFAULT_HOST = "http://localhost:11434"

_MODELS = [
    Model(id="deepseek-v4-flash:cloud", name="DeepSeek V4 Flash"),
    Model(id="deepseek-v4-pro:cloud", name="DeepSeek V4 Pro"),
    Model(id="minimax-m3:cloud", name="MiniMax M3", modalities=("text", "image")),
    Model(id="glm-5.2:cloud", name="GLM 5.2"),
    Model(id="nemotron-3-ultra:cloud", name="Nemotron Ultra"),
]


class OllamaProvider(Provider):
    """Executes prompts against Ollama's OpenAI-compatible cloud model proxy."""

    name = "ollama"

    def list_models(self) -> list[Model]:
        return _MODELS

    def execute(self, model_id: str, prompt: str, attachment_paths: list[str]) -> str:
        self.validate_model(model_id)
        host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST)
        return execute_chat(f"{host}/v1", "ollama", model_id, prompt, attachment_paths)


PROVIDER = OllamaProvider()
