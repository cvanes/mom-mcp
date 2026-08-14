"""mom-mcp: a local stdio MCP server for running prompts across model providers."""

from pathlib import Path

from dotenv import load_dotenv
from mcp.server import MCPServer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from mom_mcp.providers import PROVIDERS  # noqa: E402  (providers read env at import)

mcp = MCPServer("mom")


@mcp.tool()
def list_models() -> dict:
    """List supported model providers and the models available under each.

    Returns a mapping of provider name to a list of models, each with a
    human-friendly name, the provider-specific model id to pass to
    execute_prompt, and the input modalities the model accepts (text-only
    models reject image attachments).
    """
    return {
        provider.name: [
            {"name": model.name, "id": model.id, "modalities": list(model.modalities)}
            for model in provider.list_models()
        ]
        for provider in PROVIDERS.values()
    }


@mcp.tool()
def execute_prompt(provider: str, model_id: str, prompt: str,
                   attachments: list[str] | None = None) -> str:
    """Execute a single-turn prompt against a model and return its response.

    provider and model_id must match an entry from list_models. attachments
    is an optional list of local file paths (text or images) included as
    context with the prompt.
    """
    if provider not in PROVIDERS:
        supported = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown provider '{provider}'. Supported: {supported}")
    selected = PROVIDERS[provider]
    selected.validate_model(model_id)
    return selected.execute(model_id, prompt, attachments or [])


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
