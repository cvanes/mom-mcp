# mom-mcp

A local stdio MCP server (mixture of models) for running single-turn prompts across multiple model providers - OpenAI and OpenRouter. Similar in spirit to Claude's advisor tool, but not limited to one family of models: agents can fan the same prompt out to several models and combine the output.

## Tools

- `search_models` - searches models across all providers, with catalogues fetched live from each provider's API. The query is matched case-insensitively against model name, model id and provider name (an empty query matches everything); at most `limit` results are returned. Each match includes the `provider`, a human-friendly `name`, the `id` to pass to `execute_prompt` and the input `modalities` the model accepts (`null` where the provider does not report them).
- `execute_prompt` - executes a single-turn prompt against a given `provider` and `model_id`. Optional `attachments` is a list of local file paths (text or images) included as context; paths work because the server always runs on the same machine as the caller.

## Setup

```sh
cp .env.example .env   # then fill in the keys
uv sync
```

## Registering with Claude Code

```sh
claude mcp add --scope user mom -- uv run --directory /path/to/mom-mcp mom-mcp
```

## Development

```sh
uv run pytest
```

Providers are declared in `src/mom_mcp/providers.py` as instances of a single OpenAI-compatible `Provider` dataclass, each with a parser that maps its `/v1/models` response onto the shared `Model` shape. To add a provider, declare a new `Provider` and register it in `PROVIDERS`.
