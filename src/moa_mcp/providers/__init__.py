"""Registry of supported model providers."""

from moa_mcp.providers.base import Provider
from moa_mcp.providers.ollama_provider import PROVIDER as OLLAMA
from moa_mcp.providers.openai_provider import PROVIDER as OPENAI
from moa_mcp.providers.openrouter_provider import PROVIDER as OPENROUTER

PROVIDERS: dict[str, Provider] = {
    provider.name: provider for provider in (OPENAI, OLLAMA, OPENROUTER)
}
