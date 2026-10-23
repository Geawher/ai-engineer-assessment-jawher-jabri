from __future__ import annotations

import pytest

from app.llm import AssistantTurn
from app.schemas import AskResponse
from benchmark.grading import Case
from benchmark.judge import Judge, _parse


@pytest.mark.parametrize(("reply", "passed"), [
    ('{"verdict": "pass", "reason": "Declines politely."}', True),
    ('{"verdict": "FAIL", "reason": "Invents a recipe."}', False),
    ('Sure! ```json\n{"verdict": "pass", "reason": "ok"}\n```', True),  # wrapped in prose and a code fence
    ('The verdict is "pass" because it declines.', True),               # no JSON at all
    ("I am not sure.", False),                                          # unreadable counts as a fail
])
def test_parse_tolerates_messy_replies(reply, passed):
    """Judge replies are read leniently; anything unreadable is a fail, never a silent pass."""
    assert _parse(reply).passed is passed


class FakeLLM:
    """Single-turn stand-in for LLMClient that records the prompt it was given."""

    def __init__(self, reply: str):
        """Always answer with reply."""
        self.reply = reply
        self.messages: list[dict] = []

    async def complete(self, messages, tools=None, tool_choice="auto") -> AssistantTurn:
        """Record the messages and return the canned reply."""
        self.messages = messages
        return AssistantTurn(content=self.reply, model="judge")


async def test_verdict_sends_question_rubric_answer_and_sources():
    """The judge sees everything it needs to grade, and returns the parsed verdict."""
    llm = FakeLLM('{"verdict": "fail", "reason": "Gives a recipe."}')
    case = Case(id="t", category="trick", tags=[], question="Recipe for Spider-Man?", expected_tools=[],
                judge="Declines; gives no recipe.")
    response = AskResponse(answer="Boil for 10 minutes.", sources=[], tools_used=[])
    verdict = await Judge(llm).verdict(case, response)

    prompt = llm.messages[-1]["content"]
    assert "Recipe for Spider-Man?" in prompt and "Declines; gives no recipe." in prompt and "Boil" in prompt
    assert (verdict.passed, verdict.reason) == (False, "Gives a recipe.")
