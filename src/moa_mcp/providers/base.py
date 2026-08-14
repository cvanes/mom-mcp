"""Provider abstraction and shared OpenAI-compatible chat execution."""

from abc import ABC, abstractmethod

from openai import OpenAI

from moa_mcp.attachments import build_user_message
from moa_mcp.models import Model


class Provider(ABC):
    """A model provider exposing supported models and single-turn execution."""

    name: str

    @abstractmethod
    def list_models(self) -> list[Model]: ...

    @abstractmethod
    def execute(self, model_id: str, prompt: str, attachment_paths: list[str]) -> str: ...

    def validate_model(self, model_id: str) -> None:
        if model_id not in {model.id for model in self.list_models()}:
            supported = ", ".join(model.id for model in self.list_models())
            raise ValueError(
                f"Model '{model_id}' is not supported by {self.name}. Supported: {supported}"
            )


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
