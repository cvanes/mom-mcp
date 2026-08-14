"""Shared data structures describing providers and their supported models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """A model supported by a provider.

    id is the provider-specific model identifier used in API calls;
    name is a human-friendly label agents can use to map across providers;
    modalities lists the input types the model accepts via this server.
    """

    id: str
    name: str
    modalities: tuple[str, ...] = ("text",)
