"""mom-mcp: a local stdio MCP server for running prompts across model providers."""

from dotenv import load_dotenv
from mcp.server import MCPServer

from mom_mcp import providers
from mom_mcp.providers import PROVIDERS, Provider, SearchResult

load_dotenv()

mcp = MCPServer("mom")

SUPPORTED_PROVIDERS = ", ".join(PROVIDERS)

_REVIEWER_SYSTEM_PROMPT = """\
You are an expert reviewer asked for a candid second opinion on a piece of \
work, which may be writing, code, a design or a technical solution.

- Lead with your overall verdict, then list findings ordered by importance.
- Be specific: quote or point to the exact part each finding refers to, and \
suggest a concrete fix or alternative.
- Challenge weak assumptions and reasoning, but do not invent problems to \
seem thorough; if the work is sound, say so plainly.
- When a focus is given, keep the review centred on it."""


def _provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Supported: {SUPPORTED_PROVIDERS}")
    return PROVIDERS[name]


@mcp.tool(description=f"""Search models across the supported providers: {SUPPORTED_PROVIDERS}.

Model catalogues are fetched live from each provider's API and cached for a
few minutes. The query is matched case-insensitively against each model's
name, model id and provider name; an empty query matches every model. At most
limit results are returned, so refine the query rather than raising the limit.

Returns an object with:
- models: matching models, each with the provider, a human-friendly name, the
  model id to pass to execute_prompt/get_second_opinion, the input modalities
  the model accepts (null when the provider does not report them; text-only
  models reject image attachments) and the cost in USD per million input and
  output tokens (null when the provider does not report pricing or bills by
  subscription);
- truncated: true when more models matched than the limit returned;
- errors: one entry per provider whose catalogue could not be fetched (for
  example a missing API key); those providers are skipped, not fatal.""")
def search_models(query: str = "", limit: int = 50) -> SearchResult:
    return providers.search_models(query, limit)


@mcp.tool(description=f"""Execute a single-turn prompt against a model and return its response.

provider must be one of: {SUPPORTED_PROVIDERS}. model_id must be a model id
returned by search_models for that provider. system is an optional system
prompt steering the model's behaviour. attachments is an optional list of
local file paths included as context with the prompt: text files are inlined,
images are embedded for vision models, and documents (PDF, Word, PowerPoint,
Excel, ...) are converted to markdown.""")
def execute_prompt(provider: str, model_id: str, prompt: str,
                   system: str | None = None,
                   attachments: list[str] | None = None) -> str:
    return _provider(provider).execute(model_id, prompt, attachments or [], system)


@mcp.tool(description=f"""Get a candid second opinion on a piece of work from another model.

Pass the work to review (writing, code, a design, a technical solution, ...)
in work; it is sent to the model along with instructions to review it
rigorously, lead with a verdict and give specific, actionable findings. Use
focus to narrow the review (for example "argument structure" or "thread
safety"); leave it empty for a general review. attachments is an optional
list of local file paths included as extra context. provider must be one of:
{SUPPORTED_PROVIDERS}; model_id must be a model id returned by search_models
for that provider.""")
def get_second_opinion(provider: str, model_id: str, work: str,
                       focus: str = "",
                       attachments: list[str] | None = None) -> str:
    prompt = f"<work>\n{work}\n</work>"
    if focus.strip():
        prompt += f"\n\nFocus your review on: {focus.strip()}"
    return _provider(provider).execute(
        model_id, prompt, attachments or [], _REVIEWER_SYSTEM_PROMPT
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
