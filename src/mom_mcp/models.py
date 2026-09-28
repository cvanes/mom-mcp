"""Shared data structures describing providers and their supported models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cost:
    """Per-token pricing in USD per million tokens, where a provider reports it."""

    input: float
    output: float


@dataclass(frozen=True)
class Model:
    """A model available from a provider.

    id is the provider-specific model identifier used in API calls;
    name is a human-friendly label agents can use to map across providers;
    modalities lists the input types the model accepts, or None when the
    provider does not report them;
    cost is the per-token price, or None when the provider does not report
    pricing (e.g. OpenAI) or bills by subscription (e.g. Ollama cloud).
    """

    id: str
    name: str
    modalities: tuple[str, ...] | None = None
    cost: Cost | None = None
