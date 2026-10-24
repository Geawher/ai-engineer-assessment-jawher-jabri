"""Run the benchmark against the real agent and write a results file plus a summary.

    uv run python -m benchmark.run                      # full run (live LLM + Superhero API)
    uv run python -m benchmark.run --ids myth-01 hero-03
    uv run python -m benchmark.run --retrieval-only     # BM25 recall on the raw questions, no LLM
    uv run python -m benchmark.run --summarize benchmark/results/<stamp>-results.jsonl   # rebuild a report
    uv run python -m benchmark.run --resume benchmark/results/<stamp>-results.jsonl      # continue a stopped run
    uv run python -m benchmark.run --regrade benchmark/results/<stamp>-results.jsonl [--ids q1 q2]
        # re-grade stored answers with the current grader; the judge re-runs only for --ids

Results are appended as each question finishes. If every LLM provider stays unavailable (typically a daily
quota), the run stops without recording that question; --resume picks up from there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import httpx

from app.agent import Agent, ToolName
from app.config import Settings, get_settings
from app.llm import AssistantTurn, Chat, LLMClient, LLMError, LLMUnavailableError, ToolChoice
from app.retrieval.base import Chunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.titles import TitleIndex
from app.retrieval.loader import load_collections
from app.schemas import AskResponse
from app.sources.superhero import Hero, SuperheroAPIError, SuperheroClient
from benchmark.grading import Case, Check, extra_tools, fold, grade, load_cases
from benchmark.judge import Judge
from benchmark.report import summarize, summarize_retrieval

RESULTS_DIR = Path(__file__).parent / "results"
RETRY_DELAYS_SECONDS = (20, 45, 90)  # free tiers recover within a minute or two
RANK_DEPTH = 50  # how deep retrieval_baseline looks for the first expected article
T = TypeVar("T")


@dataclass
class Trace:
    """What happened inside the agent while answering one question (last provider attempt only)."""

    search_queries: list[dict[str, Any]] = field(default_factory=list)
    hero_lookups: list[dict[str, Any]] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    # Per LLM call, abandoned attempts included: they cost tokens too.
    usage: list[dict[str, int]] = field(default_factory=list)
    provider_attempts: int = 0

    def start_attempt(self) -> None:
        """Forget tool calls from an abandoned provider attempt so they aren't counted twice."""
        self.search_queries.clear()
        self.hero_lookups.clear()
        self.provider_attempts += 1


class RecordingRetriever:
    """Retriever proxy that logs each query and the articles it returned."""

    def __init__(self, inner: BM25Retriever, trace: Trace):
        """Wrap inner and append to trace."""
        self._inner = inner
        self.trace = trace

    def search(self, query: str, k: int, collection: str | None = None) -> list[Chunk]:
        """Delegate and record the query with the returned article titles."""
        chunks = self._inner.search(query, k, collection)
        self.trace.search_queries.append({"query": query, "titles": [chunk.title for chunk in chunks]})
        return chunks


class RecordingSuperheroes:
    """Superhero client proxy that logs each lookup and its result."""

    def __init__(self, inner: SuperheroClient, trace: Trace):
        """Wrap inner and append to trace."""
        self._inner = inner
        self.trace = trace

    async def search(self, name: str) -> list[Hero]:
        """Delegate and record the name with the returned heroes (or the error)."""
        try:
            heroes = await self._inner.search(name)
        except SuperheroAPIError as error:
            self.trace.hero_lookups.append({"name": name, "error": str(error)})
            raise
        self.trace.hero_lookups.append({"name": name, "heroes": [f"{hero.name}#{hero.id}" for hero in heroes]})
        return heroes


class RecordingLLM:
    """LLM client proxy that logs provider attempts and which model produced each turn."""

    def __init__(self, inner: LLMClient, trace: Trace):
        """Wrap inner and append to trace."""
        self._inner = inner
        self.trace = trace

    async def run_conversation(self, conversation: Callable[[Chat], Awaitable[T]]) -> T:
        """Delegate, marking each provider attempt and recording every turn's model."""

        async def recorded(chat: Chat) -> T:
            """Run one attempt with a chat function that records the answering model."""
            self.trace.start_attempt()

            async def recording_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                                     tool_choice: ToolChoice = "auto") -> AssistantTurn:
                """Delegate one turn and record its model."""
                turn = await chat(messages, tools=tools, tool_choice=tool_choice)
                self.trace.models.append(turn.model)
                self.trace.usage.append(asdict(turn.usage))
                return turn

            return await conversation(recording_chat)

        return await self._inner.run_conversation(recorded)


async def run_benchmark(cases: list[Case], settings: Settings, delay_seconds: float, results_path: Path) -> bool:
    """Ask every case through a fully wired agent, grade it, and append the result to results_path.

    Returns:
        True if every case ran; False if the run stopped because no LLM provider was available.
    """
    chunks, collections = load_collections(settings.documents_dir)
    retriever = BM25Retriever(chunks)
    titles = TitleIndex(chunks)
    texts = passage_texts(chunks)
    llm = LLMClient.from_settings(settings)
    judge = Judge.from_env()
    print(f"LLM judge: {'on' if judge else 'off (set JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL)'}")
    completed = True
    async with httpx.AsyncClient(timeout=settings.superhero_timeout_seconds, follow_redirects=True) as http:
        superheroes = SuperheroClient(
            http,
            base_url=settings.superhero_api_base_url,
            token=settings.superhero_api_token.get_secret_value(),
            cache_ttl_seconds=settings.superhero_cache_ttl_seconds,
        )

        def make_agent(trace: Trace) -> Agent:
            """A fully wired agent whose dependencies record into trace."""
            return Agent(
                RecordingLLM(llm, trace), RecordingRetriever(retriever, trace), RecordingSuperheroes(superheroes, trace),
                collections=collections, titles=titles,
                top_k=settings.retrieval_top_k, max_tool_rounds=settings.max_tool_rounds,
            )

        for index, case in enumerate(cases, start=1):
            result = await _run_case(case, make_agent, judge, texts)
            if is_out_of_quota(result):
                # Recording this as a failure would blame the agent for a quota; stop and resume later instead.
                print(f"Stopped at {case.id}: no LLM provider available ({result['error']}).\n"
                      f"Continue with: python -m benchmark.run --resume {results_path}")
                completed = False
                break
            with results_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(cases)}] {case.id:8} {'PASS' if result['passed'] else 'FAIL'} "
                  f"{result['latency_seconds']:5.1f}s {result.get('failed_checks') or result.get('error') or ''}")
            await asyncio.sleep(delay_seconds)
    await llm.close()
    if judge:
        await judge.close()
    return completed


def passage_texts(chunks: list[Chunk]) -> dict[str, str]:
    """Text per passage source title ("Thor — Introduction"), as the agent titles document sources."""
    texts: dict[str, list[str]] = {}
    for chunk in chunks:
        texts.setdefault(f"{chunk.title} — {chunk.section}", []).append(chunk.text)
    return {title: " ".join(parts) for title, parts in texts.items()}


async def regrade(results: list[dict[str, Any]], cases: dict[str, Case], texts: dict[str, str],
                  rejudge_ids: set[str]) -> None:
    """Re-grade stored answers in place with the current grader, without asking the bot again.

    Judge verdicts are kept as stored, except for rejudge_ids (e.g. after a rubric fix), which are judged again.
    """
    judge = Judge.from_env() if rejudge_ids else None
    for row in results:
        if row.get("error"):
            continue
        case = cases[row["id"]]
        response = AskResponse(answer=row["answer"], sources=row["sources"], tools_used=row["tools_used"],
                               warnings=row["warnings"])
        checks = grade(case, response, texts)
        if case.judge and judge and row["id"] in rejudge_ids:
            verdict = await judge.verdict(case, response)
            checks[Check.JUDGE], row["judge_reason"] = verdict.passed, verdict.reason
        elif Check.JUDGE.value in row.get("checks", {}):
            checks[Check.JUDGE] = row["checks"][Check.JUDGE.value]
        row.update(passed=all(checks.values()), checks={check.value: ok for check, ok in checks.items()},
                   failed_checks=[check.value for check, ok in checks.items() if not ok])
    if judge:
        await judge.close()


def is_out_of_quota(result: dict[str, Any]) -> bool:
    """Whether a case failed only because every LLM provider stayed unavailable through all retries."""
    return (result.get("error") or "").startswith(LLMUnavailableError.__name__)


def load_results(path: Path) -> list[dict[str, Any]]:
    """Rows of a results file; a missing file has none."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


async def _run_case(case: Case, make_agent: Callable[[Trace], Agent], judge: Judge | None,
                    texts: dict[str, str]) -> dict[str, Any]:
    """Ask one question (retrying when every provider is rate-limited) and grade it, with the judge if any."""
    started = time.monotonic()
    response: AskResponse | None = None
    error = None
    benchmark_retries = 0
    for retry_delay in (*RETRY_DELAYS_SECONDS, None):
        trace = Trace()  # a retry starts clean, so its tool calls aren't mixed with the failed attempt's
        try:
            response = await make_agent(trace).ask(case.question)
            break
        except LLMUnavailableError as exc:
            error = f"{type(exc).__name__}: {exc}"
            if retry_delay is None:
                break
            print(f"    all providers unavailable, retrying in {retry_delay}s")
            benchmark_retries += 1
            await asyncio.sleep(retry_delay)
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            break

    result: dict[str, Any] = {
        "id": case.id,
        "category": case.category,
        "tags": case.tags,
        "question": case.question,
        "latency_seconds": round(time.monotonic() - started, 2),
        "benchmark_retries": benchmark_retries,
        **asdict(trace),
    }
    if response is None:
        return result | {"passed": False, "error": error, "checks": {}}

    checks = grade(case, response, texts)
    judge_reason = None
    if case.judge and judge:
        try:
            verdict = await judge.verdict(case, response)
            checks[Check.JUDGE], judge_reason = verdict.passed, verdict.reason
        except LLMError as exc:  # a judge outage shouldn't fail the agent
            judge_reason = f"judge unavailable: {exc}"
    return result | {
        "judge_reason": judge_reason,
        "passed": all(checks.values()),
        "checks": {check.value: ok for check, ok in checks.items()},
        "failed_checks": [check.value for check, ok in checks.items() if not ok],
        "retrieval_hit": _retrieval_hit(case, trace),
        "extra_tools": extra_tools(case, response),
        "answer": response.answer,
        "tools_used": response.tools_used,
        "sources": [source.model_dump(mode="json") for source in response.sources],
        "warnings": response.warnings,
    }


def _expected_articles(case: Case, titles: set[str]) -> set[str]:
    """The corpus articles among a case's expected sources (hero names are ignored)."""
    return {fold(name) for group in case.expected_sources for name in group if fold(name) in titles}


def _retrieval_hit(case: Case, trace: Trace) -> bool | None:
    """Whether any search the model ran returned an expected article; None if not applicable."""
    expected = {fold(name) for group in case.expected_sources for name in group}
    if not expected or not trace.search_queries or ToolName.SEARCH_DOCUMENTS not in case.expected_tools:
        return None
    return any(fold(title) in expected for query in trace.search_queries for title in query["titles"])


def retrieval_baseline(cases: list[Case], settings: Settings) -> list[dict[str, Any]]:
    """Rank of the first expected-article chunk when the raw question is the BM25 query."""
    chunks, _ = load_collections(settings.documents_dir)
    retriever = BM25Retriever(chunks)
    titles = {fold(chunk.title) for chunk in chunks}
    rows = []
    for case in cases:
        expected = _expected_articles(case, titles)
        if not expected or ToolName.SEARCH_DOCUMENTS not in case.expected_tools:
            continue
        ranked = [fold(chunk.title) for chunk in retriever.search(case.question, RANK_DEPTH)]
        rank = next((position for position, title in enumerate(ranked, start=1) if title in expected), None)
        rows.append({"id": case.id, "tags": case.tags, "question": case.question, "rank": rank,
                     "top3": ranked[:3]})
    return rows


def main() -> None:
    """Parse arguments, run the requested mode and write results under benchmark/results/."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", help="Only run these case ids")
    parser.add_argument("--retrieval-only", action="store_true", help="BM25 recall on raw questions, no LLM calls")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait between questions")
    parser.add_argument("--summarize", type=Path, metavar="RESULTS_JSONL",
                        help="Rebuild the report for an existing results file instead of running")
    parser.add_argument("--resume", type=Path, metavar="RESULTS_JSONL",
                        help="Continue a stopped run: skip questions already in this file and append to it")
    parser.add_argument("--regrade", type=Path, metavar="RESULTS_JSONL",
                        help="Re-grade stored answers with the current grader; the judge re-runs only for --ids")
    args = parser.parse_args()

    settings = get_settings()
    cases = [case for case in load_cases() if args.regrade or not args.ids or case.id in args.ids]
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    retrieval_rows = retrieval_baseline(cases, settings)

    if args.retrieval_only:
        report = summarize_retrieval(retrieval_rows)
        (RESULTS_DIR / f"{stamp}-bm25-baseline.md").write_text(report, encoding="utf-8")
        print(report)
        return

    results_path = args.summarize or args.regrade or args.resume or RESULTS_DIR / f"{stamp}-results.jsonl"
    if args.regrade:
        results = load_results(results_path)
        texts = passage_texts(load_collections(settings.documents_dir)[0])
        asyncio.run(regrade(results, {case.id: case for case in cases}, texts, set(args.ids or ())))
        with results_path.open("w", encoding="utf-8") as output:
            output.writelines(json.dumps(result, ensure_ascii=False) + "\n" for result in results)
    elif not args.summarize:
        done = {result["id"] for result in load_results(results_path)}
        remaining = [case for case in cases if case.id not in done]
        print(f"{len(done)} questions already in {results_path.name}, {len(remaining)} to run")
        asyncio.run(run_benchmark(remaining, settings, args.delay, results_path))
    results = load_results(results_path)
    summary_path = results_path.with_name(results_path.name.replace("-results.jsonl", "-summary.md"))

    titles = {fold(chunk.title) for chunk in load_collections(settings.documents_dir)[0]}
    summary = summarize(results, {case.id: case for case in load_cases()}, titles, retrieval_rows)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
