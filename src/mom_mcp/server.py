"""mom-mcp: a local stdio MCP server for running prompts across model providers."""

from pathlib import Path

from dotenv import load_dotenv
from mcp.server import MCPServer

from mom_mcp import providers
from mom_mcp.providers import PROVIDERS

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

mcp = MCPServer("mom")

SUPPORTED_PROVIDERS = ", ".join(PROVIDERS)


@mcp.tool(description=f"""Search models across the supported providers: {SUPPORTED_PROVIDERS}.

The query is matched case-insensitively against each model's name, model id and
provider name; an empty query returns every model. Each match includes the
provider, a human-friendly name, the model id to pass to execute_prompt and the
input modalities the model accepts (text-only models reject image attachments).""")
def search_models(query: str = "") -> list[dict]:
    return providers.search_models(query)


@mcp.tool(description=f"""Execute a single-turn prompt against a model and return its response.

provider must be one of: {SUPPORTED_PROVIDERS}. model_id must be a model id
returned by search_models for that provider. attachments is an optional list of
local file paths (text or images) included as context with the prompt.""")
def execute_prompt(provider: str, model_id: str, prompt: str,
                   attachments: list[str] | None = None) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Supported: {SUPPORTED_PROVIDERS}")
    return PROVIDERS[provider].execute(model_id, prompt, attachments or [])


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
