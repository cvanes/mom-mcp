# moa-mcp

A local stdio MCP server (mixture of agents) for running single-turn prompts across multiple model providers - OpenAI, Ollama (cloud models proxied locally) and OpenRouter. Similar in spirit to Claude's advisor tool, but not limited to one family of models: agents can fan the same prompt out to several models and combine the output.

## Tools

- `list_models` - returns supported providers and their models, each with a human-friendly `name` and the provider-specific `id` to pass to `execute_prompt`.
- `execute_prompt` - executes a single-turn prompt against a given `provider` and `model_id`. Optional `attachments` is a list of local file paths (text or images) included as context; paths work because the server always runs on the same machine as the caller.

## Setup

```sh
cp .env.example .env   # then fill in the keys
uv sync
```

Ollama must be running locally (`ollama serve`) and signed in to ollama.com for the cloud models.

## Registering with Claude Code

```sh
claude mcp add --scope user moa -- uv run --directory /path/to/moa-mcp moa-mcp
```

## Development

```sh
uv run pytest
```

Providers live in `src/moa_mcp/providers/`, one module each, all delegating to a shared OpenAI-compatible chat helper in `base.py`. To add a model, extend the hardcoded list in the relevant provider module.
