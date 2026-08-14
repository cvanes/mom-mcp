"""Registry of supported model providers."""

from mom_mcp.providers.base import Provider
from mom_mcp.providers.ollama_provider import PROVIDER as OLLAMA
from mom_mcp.providers.openai_provider import PROVIDER as OPENAI
from mom_mcp.providers.openrouter_provider import PROVIDER as OPENROUTER

PROVIDERS: dict[str, Provider] = {
    provider.name: provider for provider in (OPENAI, OLLAMA, OPENROUTER)
}
