from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletionMessage

from app.llm import LLMClient, LLMError, LLMUnavailableError, Provider

_REQUEST = httpx.Request("POST", "https://llm.test/v1/chat/completions")
_MESSAGES = [{"role": "user", "content": "hello"}]


def api_error(error_type: type[openai.APIStatusError], status_code: int) -> openai.APIStatusError:
    """Build an SDK status error as the client would raise it."""
    return error_type("boom", response=httpx.Response(status_code, request=_REQUEST), body=None)


def reply(content: str) -> SimpleNamespace:
    """A minimal chat-completion response with a text answer."""
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    """Stand-in for AsyncOpenAI whose completions either return or raise a fixed outcome."""

    def __init__(self, outcome: Any):
        """Make every create() call return outcome, or raise it if it's an exception."""
        self.calls = 0

        async def create(**request: Any) -> Any:
            """Count the call and return or raise the configured outcome."""
            self.calls += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def client_with(*outcomes: Any) -> tuple[LLMClient, list[FakeOpenAI]]:
    """An LLMClient with one fake provider per outcome, in order."""
    fakes = [FakeOpenAI(outcome) for outcome in outcomes]
    providers = [Provider(name=f"p{i}", model=f"model-{i}", client=fake) for i, fake in enumerate(fakes)]
    return LLMClient(providers, temperature=0), fakes


async def test_uses_primary_when_healthy():
    """A healthy primary answers and the fallback is never called."""
    llm, fakes = client_with(reply("hi"), reply("unused"))
    turn = await llm.complete(_MESSAGES)
    assert (turn.content, turn.model) == ("hi", "model-0")
    assert fakes[1].calls == 0


@pytest.mark.parametrize("error", [
    api_error(openai.RateLimitError, 429),
    api_error(openai.InternalServerError, 503),
    openai.APITimeoutError(request=_REQUEST),
])
async def test_fails_over_on_transient_errors(error):
    """Rate limits, 5xx and timeouts move on to the fallback provider."""
    llm, _ = client_with(error, reply("from fallback"))
    turn = await llm.complete(_MESSAGES)
    assert (turn.content, turn.model) == ("from fallback", "model-1")


async def test_does_not_fail_over_on_client_errors():
    """A client error such as a bad key surfaces instead of being masked by the fallback."""
    llm, fakes = client_with(api_error(openai.AuthenticationError, 401), reply("unused"))
    with pytest.raises(LLMError):
        await llm.complete(_MESSAGES)
    assert fakes[1].calls == 0


async def test_raises_unavailable_when_all_providers_fail():
    """When every provider is rate-limited, LLMUnavailableError is raised."""
    llm, _ = client_with(api_error(openai.RateLimitError, 429), api_error(openai.RateLimitError, 429))
    with pytest.raises(LLMUnavailableError):
        await llm.complete(_MESSAGES)


async def test_provider_specific_tool_call_fields_are_kept():
    """Extra tool-call fields such as Gemini's thought_signature survive the conversion to ToolCall."""
    message = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_mythology", "arguments": "{\"query\": \"Loki\"}"},
            "extra_content": {"google": {"thought_signature": "sig"}},
        }],
    })
    llm, _ = client_with(SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    turn = await llm.complete(_MESSAGES)
    assert turn.tool_calls[0].extra == {"extra_content": {"google": {"thought_signature": "sig"}}}


async def test_failover_restarts_the_whole_conversation():
    """A provider failing mid-conversation hands the fallback a fresh start, not the old history."""
    flaky = FakeOpenAI(api_error(openai.RateLimitError, 429))
    healthy = FakeOpenAI(reply("done"))
    llm = LLMClient([Provider("p0", "model-0", flaky), Provider("p1", "model-1", healthy)], temperature=0)
    attempts: list[int] = []

    async def conversation(chat):
        """Two-turn conversation that records how many times it was started."""
        attempts.append(len(attempts))
        await chat(_MESSAGES)
        return await chat(_MESSAGES)

    turn = await llm.run_conversation(conversation)
    assert turn.model == "model-1"
    assert attempts == [0, 1]
    assert (flaky.calls, healthy.calls) == (1, 2)


def rate_limited(retry_after: str) -> openai.RateLimitError:
    """A 429 carrying a Retry-After header."""
    response = httpx.Response(429, request=_REQUEST, headers={"retry-after": retry_after})
    return openai.RateLimitError("slow down", response=response, body=None)


class SequenceOpenAI:
    """Fake client returning (or raising) a fixed sequence of outcomes, one per call."""

    def __init__(self, outcomes: list[Any]):
        """Queue outcomes in call order."""
        self.calls = 0
        queue = list(outcomes)

        async def create(**request: Any) -> Any:
            """Pop the next outcome."""
            self.calls += 1
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def test_short_rate_limit_is_waited_out_on_the_same_provider():
    """A Retry-After within the budget is honoured instead of failing over."""
    primary = SequenceOpenAI([rate_limited("0.01"), reply("after waiting")])
    fallback = FakeOpenAI(reply("unused"))
    llm = LLMClient([Provider("p0", "model-0", primary), Provider("p1", "model-1", fallback)],
                    temperature=0, max_rate_limit_wait=1)
    turn = await llm.complete(_MESSAGES)
    assert (turn.content, primary.calls, fallback.calls) == ("after waiting", 2, 0)


async def test_long_rate_limit_fails_over_immediately():
    """A Retry-After beyond the budget (e.g. a daily quota) goes straight to the fallback."""
    primary = SequenceOpenAI([rate_limited("146")])
    llm = LLMClient([Provider("p0", "model-0", primary), Provider("p1", "model-1", FakeOpenAI(reply("fallback")))],
                    temperature=0, max_rate_limit_wait=10)
    turn = await llm.complete(_MESSAGES)
    assert (turn.content, primary.calls) == ("fallback", 1)


def malformed_tool_call() -> openai.BadRequestError:
    """Groq's 400 for a tool call that didn't match the schema."""
    response = httpx.Response(400, request=_REQUEST)
    return openai.BadRequestError("bad tool call", response=response, body={"code": "tool_use_failed"})


async def test_malformed_tool_call_is_resampled():
    """One malformed generation is retried on the same provider."""
    primary = SequenceOpenAI([malformed_tool_call(), reply("fixed")])
    llm = LLMClient([Provider("p0", "model-0", primary)], temperature=0)
    turn = await llm.complete(_MESSAGES)
    assert (turn.content, primary.calls) == ("fixed", 2)


async def test_repeated_malformed_tool_calls_fail_over():
    """A model that keeps producing malformed tool calls is treated like an outage."""
    primary = SequenceOpenAI([malformed_tool_call(), malformed_tool_call()])
    llm = LLMClient([Provider("p0", "model-0", primary), Provider("p1", "model-1", FakeOpenAI(reply("fallback")))],
                    temperature=0)
    turn = await llm.complete(_MESSAGES)
    assert turn.content == "fallback"


async def test_retries_cover_a_rate_limit_followed_by_a_malformed_tool_call():
    """Each kind of transient failure gets its own retry, even when they happen back to back."""
    primary = SequenceOpenAI([rate_limited("0.01"), malformed_tool_call(), reply("third time lucky")])
    llm = LLMClient([Provider("p0", "model-0", primary)], temperature=0, max_rate_limit_wait=1)
    turn = await llm.complete(_MESSAGES)
    assert (turn.content, primary.calls) == ("third time lucky", 3)


async def test_token_usage_is_reported_including_cache_hits():
    """Prompt, completion and cached tokens come through; a provider without usage reports zeros."""
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=40,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=640))
    response = reply("hi")
    response.usage = usage
    llm, _ = client_with(response)
    turn = await llm.complete(_MESSAGES)
    assert (turn.usage.prompt, turn.usage.completion, turn.usage.cached_prompt) == (900, 40, 640)

    llm_without_usage, _ = client_with(reply("hi"))
    assert (await llm_without_usage.complete(_MESSAGES)).usage.prompt == 0
