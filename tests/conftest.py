from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import pytest

from app.llm import AssistantTurn, Chat, ToolCall, ToolChoice
from app.retrieval.base import Chunk
from app.retrieval.bm25 import BM25Retriever
from app.sources.superhero import Hero, SuperheroAPIError

CHUNKS = [
    Chunk("thor:0", "mythology", "Thor", "Introduction", "https://en.wikipedia.org/wiki/Thor",
          "Thor is a hammer-wielding god associated with thunder, lightning and storms. He wields Mjölnir."),
    Chunk("odin:0", "mythology", "Odin", "Introduction", "https://en.wikipedia.org/wiki/Odin",
          "Odin is a widely revered god associated with wisdom, healing, death and royalty."),
    Chunk("zeus:0", "mythology", "Zeus", "Introduction", "https://en.wikipedia.org/wiki/Zeus",
          "Zeus is the sky and thunder god in ancient Greek religion and king of the gods on Mount Olympus."),
]

T = TypeVar("T")

COLLECTIONS = {"mythology": "Norse and Greek mythology."}

THOR_HERO = Hero(id="659", name="Thor", profile={"name": "Thor", "publisher": "Marvel Comics"})


@pytest.fixture
def retriever() -> BM25Retriever:
    """A BM25 retriever over three tiny mythology chunks."""
    return BM25Retriever(CHUNKS)


class ScriptedLLM:
    """Returns pre-written turns in order and records what it was sent."""

    def __init__(self, turns: list[AssistantTurn]):
        """Queue the turns to return, one per complete() call."""
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def run_conversation(self, conversation: Callable[[Chat], Awaitable[T]]) -> T:
        """Run the conversation once against the scripted turns (no failover)."""
        return await conversation(self.complete)

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                       tool_choice: ToolChoice = "auto") -> AssistantTurn:
        """Record a snapshot of the request and return the next scripted turn."""
        self.calls.append({"messages": [dict(message) for message in messages], "tool_choice": tool_choice})
        return self._turns.pop(0)


class FakeSuperheroes:
    """In-memory stand-in for SuperheroClient."""

    def __init__(self, heroes: dict[str, list[Hero]] | None = None, *, is_down: bool = False):
        """Serve heroes keyed by lowercase name, or fail every call when is_down."""
        self._heroes = heroes or {}
        self._is_down = is_down

    async def search(self, name: str) -> list[Hero]:
        """Return the configured heroes for name, or raise if the fake API is down."""
        if self._is_down:
            raise SuperheroAPIError("Superhero API is unavailable")
        return self._heroes.get(name.lower(), [])


def tool_turn(*calls: tuple[str, dict[str, Any]], model: str = "fake-model") -> AssistantTurn:
    """An assistant turn that calls the given (tool name, arguments) pairs."""
    return AssistantTurn(
        content=None,
        model=model,
        tool_calls=[ToolCall(id=f"call_{i}", name=name, arguments=json.dumps(args))
                    for i, (name, args) in enumerate(calls)],
    )


def answer_turn(text: str, model: str = "fake-model") -> AssistantTurn:
    """An assistant turn that answers with text and calls no tools."""
    return AssistantTurn(content=text, model=model)
