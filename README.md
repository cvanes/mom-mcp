# mom-mcp

A local stdio MCP server (mixture of models) for running single-turn prompts across multiple model providers - OpenAI, Ollama (cloud) and OpenRouter. Similar in spirit to Claude's advisor tool, but not limited to one family of models: agents can fan the same prompt out to several models and combine the output.

## Tools

- `get_second_opinion` - sends a piece of work (writing, code, a design, a technical solution) to another model with reviewer instructions: lead with a verdict, then specific, actionable findings ordered by importance. An optional `focus` narrows the review; optional `attachments` add context.
- `execute_prompt` - executes a single-turn prompt against a given `provider` and `model_id`, with an optional `system` prompt. Optional `attachments` is a list of local file paths included as context: text files are inlined, images are embedded for vision models, and documents (PDF, Word, PowerPoint, Excel, ...) are converted to markdown with markitdown. Paths work because the server always runs on the same machine as the caller.
- `search_models` - searches models across all providers, with catalogues fetched live from each provider's API and cached for a few minutes. The query is matched case-insensitively against model name, model id and provider name (an empty query matches everything). Returns an object with:
  - `models` - at most `limit` matches, interleaved across providers so one large catalogue cannot crowd out the others, each with the `provider`, a human-friendly `name`, the `id` to pass to the other tools, the input `modalities` (`null` where the provider does not report them) and the `cost` in USD per million input/output tokens (`null` where pricing is not reported or is subscription-based, e.g. Ollama cloud);
  - `truncated` - `true` when more models matched than `limit` returned;
  - `errors` - one entry per provider whose catalogue could not be fetched (e.g. a missing API key); those providers are skipped rather than failing the whole search.

## Setup

```sh
cp .env.example .env   # then fill in the keys you use; every provider is optional
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
