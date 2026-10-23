from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError, LLMUnavailableError
from app.main import app, get_agent
from app.schemas import AskResponse, Source, SourceType


class StubAgent:
    """Agent stand-in that returns a fixed response or raises a fixed error."""

    def __init__(self, result: AskResponse | None = None, error: Exception | None = None):
        """Configure the response to return, or the error to raise."""
        self._result = result
        self._error = error
        self.questions: list[str] = []

    async def ask(self, question: str) -> AskResponse:
        """Record the question, then return or raise as configured."""
        self.questions.append(question)
        if self._error:
            raise self._error
        return self._result


@pytest.fixture
def client_for() -> Iterator[Callable[[StubAgent], TestClient]]:
    """Factory for a TestClient whose /ask endpoint uses the given stub agent."""
    # TestClient without a `with` block skips the lifespan, so no real keys or network are needed.
    def make(agent: StubAgent) -> TestClient:
        """Point the app at agent and return a client for it."""
        app.dependency_overrides[get_agent] = lambda: agent
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_ask_returns_answer_with_sources(client_for):
    """A valid question returns the agent's answer and sources, with whitespace stripped."""
    agent = StubAgent(AskResponse(
        answer="Thor is a god [S1].",
        sources=[Source(id="S1", type=SourceType.MYTHOLOGY_DATASET, title="Thor — Introduction")],
        tools_used=["search_mythology"],
    ))
    response = client_for(agent).post("/ask", json={"question": "  Who is Thor?  "})

    assert response.status_code == 200
    assert response.json()["sources"][0]["id"] == "S1"
    assert agent.questions == ["Who is Thor?"]


@pytest.mark.parametrize("payload", [
    {},
    {"question": ""},
    {"question": "   "},
    {"question": "x" * 1001},
    {"question": "Who is Thor?", "extra": 1},
])
def test_invalid_requests_are_rejected(client_for, payload):
    """Missing, blank, too-long or extra-field payloads get a 422."""
    assert client_for(StubAgent()).post("/ask", json=payload).status_code == 422


@pytest.mark.parametrize(("error", "status_code"), [(LLMUnavailableError(), 503), (LLMError(), 502)])
def test_llm_failures_map_to_http_errors(client_for, error, status_code):
    """An LLM outage maps to 503 and a provider rejection to 502."""
    response = client_for(StubAgent(error=error)).post("/ask", json={"question": "Who is Thor?"})
    assert response.status_code == status_code

