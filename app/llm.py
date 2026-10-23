from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, TypeVar

import openai
from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)

ToolChoice = Literal["auto", "none"]
T = TypeVar("T")

_MALFORMED_TOOL_CALL = "tool_use_failed"


class LLMError(Exception):
    """A provider rejected the request in a way that failing over won't fix."""


class ModelGlitchError(Exception):
    """The model repeatedly produced output the provider refused to return (e.g. malformed tool calls)."""


# "This provider can't serve us right now" -> try the next one.
# Anything else (400 bad request, 401 bad key) is a bug or misconfiguration and should surface.
_FAILOVER_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,  # includes timeouts
    openai.InternalServerError,
    ModelGlitchError,
)


class LLMUnavailableError(LLMError):
    """Every configured provider is rate-limited or down."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    # Provider-specific fields that must be echoed back verbatim, e.g. Gemini's
    # extra_content.google.thought_signature (the next request is rejected without it).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    """Tokens reported by the provider for one call; cached_prompt is the part served from its prompt cache."""

    prompt: int = 0
    completion: int = 0
    cached_prompt: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum two usages, e.g. across the turns of one question."""
        return TokenUsage(self.prompt + other.prompt, self.completion + other.completion,
                          self.cached_prompt + other.cached_prompt)


@dataclass(frozen=True)
class AssistantTurn:
    """Provider-agnostic view of a reply, so the agent doesn't depend on SDK types."""

    content: str | None
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


Chat = Callable[..., Awaitable[AssistantTurn]]


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    client: Any  # AsyncOpenAI, or a fake in tests


class LLMClient:
    """Chat-completion client that tries each configured provider in order."""

    def __init__(self, providers: list[Provider], temperature: float, *, max_rate_limit_wait: float = 0.0):
        """Create a client over providers, tried in list order.

        Args:
            providers: Tried in order; the first is the primary.
            temperature: Sampling temperature for every call.
            max_rate_limit_wait: Longest Retry-After (seconds) worth waiting for on the same provider.

        Raises:
            ValueError: If providers is empty.
        """
        if not providers:
            raise ValueError("At least one LLM provider is required")
        self._providers = providers
        self._temperature = temperature
        self._max_rate_limit_wait = max_rate_limit_wait

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMClient:
        """Build one AsyncOpenAI client per configured provider."""
        providers = [
            Provider(
                name=config.name,
                model=config.model,
                client=AsyncOpenAI(
                    base_url=config.base_url,
                    api_key=config.api_key.get_secret_value(),
                    timeout=settings.llm_timeout_seconds,
                    max_retries=settings.llm_max_retries,
                ),
            )
            for config in settings.providers
        ]
        return cls(providers, settings.llm_temperature, max_rate_limit_wait=settings.llm_max_rate_limit_wait_seconds)

    async def run_conversation(self, conversation: Callable[[Chat], Awaitable[T]]) -> T:
        """Run a multi-turn conversation on one provider, restarting it on the next if that provider fails.

        Failover is per conversation, not per turn: a tool-call history produced by one provider is
        not valid input for another (Gemini rejects tool calls that lack its own thought signatures),
        so switching providers mid-conversation would turn a rate limit into a hard error.

        Args:
            conversation: Receives a `chat(messages, tools=None, tool_choice="auto")` function bound
                to a single provider; it must be safe to run again from scratch.

        Returns:
            Whatever the conversation returns on the first provider that completes it.

        Raises:
            LLMUnavailableError: Every provider was rate-limited, timed out or returned 5xx.
            LLMError: A provider rejected the request (e.g. bad key or malformed request).
        """
        for provider in self._providers:
            try:
                return await conversation(partial(self._chat, provider))
            except _FAILOVER_ERRORS as error:
                logger.warning("LLM provider %s failed (%s), restarting on next", provider.name, type(error).__name__)
        raise LLMUnavailableError("All LLM providers are unavailable")

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
    ) -> AssistantTurn:
        """Single-turn convenience over run_conversation (same failover and errors)."""
        return await self.run_conversation(lambda chat: chat(messages, tools=tools, tool_choice=tool_choice))

    async def _chat(
        self,
        provider: Provider,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
    ) -> AssistantTurn:
        """One completion on one provider; failover errors propagate, other API errors become LLMError."""
        request: dict[str, Any] = {"messages": messages, "temperature": self._temperature}
        if tools:
            request |= {"tools": tools, "tool_choice": tool_choice}
        try:
            response = await self._create_with_short_wait(provider, request)
        except _FAILOVER_ERRORS:
            raise  # subclasses of APIError, but run_conversation handles them by switching provider
        except openai.APIError as error:
            logger.error("LLM provider %s rejected request: %s", provider.name, error)
            raise LLMError(f"LLM provider {provider.name} rejected the request") from error

        logger.info("LLM reply from %s (%s)", provider.name, provider.model)
        return _to_turn(response.choices[0].message, provider.model, _usage(response))

    async def _create_with_short_wait(self, provider: Provider, request: dict[str, Any]) -> Any:
        """Call the provider, retrying once on the two failures a second try usually fixes.

        - A short rate limit: per-minute token limits reset within seconds, so waiting keeps the
          conversation on this provider. A long Retry-After (a daily quota) is left to failover.
        - A malformed tool call the provider refused to return (Groq's `tool_use_failed`): a resample
          usually succeeds; if it fails again the model is struggling, so fail over.
        """
        create = partial(provider.client.chat.completions.create, model=provider.model, **request)
        has_waited = has_resampled = False
        while True:
            try:
                return await create()
            except openai.RateLimitError as error:
                wait = _retry_after_seconds(error)
                if has_waited or wait is None or wait > self._max_rate_limit_wait:
                    raise
                logger.info("LLM provider %s rate-limited, waiting %.1fs", provider.name, wait)
                has_waited = True
                await asyncio.sleep(wait)
            except openai.BadRequestError as error:
                if error.code != _MALFORMED_TOOL_CALL:
                    raise
                if has_resampled:
                    raise ModelGlitchError(f"{provider.name} keeps generating malformed tool calls") from error
                logger.warning("LLM provider %s generated a malformed tool call, resampling", provider.name)
                has_resampled = True

    async def close(self) -> None:
        """Close every provider's HTTP connection pool."""
        for provider in self._providers:
            await provider.client.close()


def _retry_after_seconds(error: openai.RateLimitError) -> float | None:
    """The provider's Retry-After header in seconds, or None if absent or not a number."""
    try:
        return float(error.response.headers.get("retry-after", ""))
    except ValueError:
        return None


def _usage(response: Any) -> TokenUsage:
    """Token counts from a completion; providers that omit them (or the cache detail) count as zero."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    details = getattr(usage, "prompt_tokens_details", None)
    return TokenUsage(
        prompt=getattr(usage, "prompt_tokens", 0) or 0,
        completion=getattr(usage, "completion_tokens", 0) or 0,
        cached_prompt=getattr(details, "cached_tokens", 0) or 0,
    )


def _to_turn(message: Any, model: str, usage: TokenUsage | None = None) -> AssistantTurn:
    """Convert an SDK chat message into an AssistantTurn."""
    return AssistantTurn(
        content=message.content,
        model=model,
        usage=usage or TokenUsage(),
        tool_calls=[
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments or "{}",
                extra=dict(getattr(call, "model_extra", None) or {}),
            )
            for call in message.tool_calls or []
        ],
    )
