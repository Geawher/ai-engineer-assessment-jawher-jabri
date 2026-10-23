"""An LLM judge for the behaviours keywords can't grade: declining, not inventing, correcting a false premise.

Configured with JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL (any OpenAI-compatible endpoint, ideally a
different model or quota from the one under test). Without them, judged cases are graded by keywords only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import AsyncOpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm import LLMClient, Provider
from app.schemas import AskResponse
from benchmark.grading import Case

JUDGE_PROMPT = """\
You grade one answer from a chatbot. The chatbot must answer only from its two data sources \
(Wikipedia articles on Norse and Greek mythology, and a superhero database), cite them, say when \
the sources don't contain the answer, and decline requests outside mythology and superheroes.

Judge ONLY against the rubric. Do not use your own knowledge to reward or punish facts the rubric \
doesn't mention. Reply with JSON only: {"verdict": "pass" or "fail", "reason": "<one sentence>"}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JUDGE_", env_file=".env", extra="ignore")

    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


class Judge:
    """Grades an answer against a case's rubric with an LLM."""

    def __init__(self, llm: LLMClient):
        """Use llm (a single-provider client) for every verdict."""
        self._llm = llm

    @classmethod
    def from_env(cls) -> Judge | None:
        """A judge from JUDGE_* settings, or None when they aren't all set."""
        settings = JudgeSettings()
        if not (settings.base_url and settings.api_key and settings.model):
            return None
        client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key.get_secret_value(),
                             timeout=60, max_retries=2)
        return cls(LLMClient([Provider("judge", settings.model, client)], temperature=0, max_rate_limit_wait=30))

    async def verdict(self, case: Case, response: AskResponse) -> Verdict:
        """Pass/fail for the answer, judged against case.judge.

        Raises:
            LLMError: If the judge model can't be reached.
        """
        sources = ", ".join(source.title for source in response.sources) or "none"
        prompt = (f"Question: {case.question}\n\nRubric: {case.judge}\n\n"
                  f"Answer:\n{response.answer}\n\nSources cited: {sources}")
        turn = await self._llm.complete([{"role": "system", "content": JUDGE_PROMPT},
                                         {"role": "user", "content": prompt}])
        return _parse(turn.content or "")

    async def close(self) -> None:
        """Release the judge's HTTP connections."""
        await self._llm.close()


def _parse(reply: str) -> Verdict:
    """Read the judge's JSON; a reply that isn't valid JSON is judged by which verdict word it contains."""
    match = _JSON_OBJECT_RE.search(reply)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", "")).lower() or ("pass" if '"pass"' in reply.lower() else "fail")
    return Verdict(passed=verdict == "pass", reason=str(data.get("reason", reply.strip()[:200])))

