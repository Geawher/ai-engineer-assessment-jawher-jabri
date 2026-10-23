"""Run the benchmark against the real agent and write a results file plus a summary.

    uv run python -m benchmark.run                      # full run (live LLM + Superhero API)
    uv run python -m benchmark.run --ids myth-01 hero-03
    uv run python -m benchmark.run --retrieval-only     # BM25 recall on the raw questions, no LLM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import httpx

from app.agent import Agent
from app.config import Settings, get_settings
from app.llm import AssistantTurn, Chat, LLMClient, LLMError, LLMUnavailableError, ToolChoice
from app.retrieval.base import Chunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.loader import load_corpus
from app.schemas import AskResponse
from app.sources.superhero import Hero, SuperheroAPIError, SuperheroClient
from benchmark.grading import Case, Check, extra_tools, fold, grade, load_cases
from benchmark.judge import Judge

RESULTS_DIR = Path(__file__).parent / "results"
RETRY_DELAYS_SECONDS = (20, 45, 90)  # free tiers recover within a minute or two
RECALL_DEPTHS = (1, 4, 10, 50)
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

    def search(self, query: str, k: int) -> list[Chunk]:
        """Delegate and record the query with the returned article titles."""
        chunks = self._inner.search(query, k)
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


async def run_benchmark(cases: list[Case], settings: Settings, delay_seconds: float) -> list[dict[str, Any]]:
    """Ask every case through a fully wired agent and grade the answers."""
    retriever = BM25Retriever(load_corpus(settings.data_dir))
    llm = LLMClient.from_settings(settings)
    judge = Judge.from_env()
    print(f"LLM judge: {'on' if judge else 'off (set JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL)'}")
    results = []
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
                top_k=settings.retrieval_top_k, max_tool_rounds=settings.max_tool_rounds,
            )

        for index, case in enumerate(cases, start=1):
            result = await _run_case(case, make_agent, judge)
            results.append(result)
            print(f"[{index}/{len(cases)}] {case.id:8} {'PASS' if result['passed'] else 'FAIL'} "
                  f"{result['latency_seconds']:5.1f}s {result.get('failed_checks') or result.get('error') or ''}")
            await asyncio.sleep(delay_seconds)
    await llm.close()
    if judge:
        await judge.close()
    return results


async def _run_case(case: Case, make_agent: Callable[[Trace], Agent], judge: Judge | None) -> dict[str, Any]:
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

    checks = grade(case, response)
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
    if not expected or not trace.search_queries or "search_mythology" not in case.expected_tools:
        return None
    return any(fold(title) in expected for query in trace.search_queries for title in query["titles"])


def retrieval_baseline(cases: list[Case], settings: Settings) -> list[dict[str, Any]]:
    """Rank of the first expected-article chunk when the raw question is the BM25 query."""
    chunks = load_corpus(settings.data_dir)
    retriever = BM25Retriever(chunks)
    titles = {fold(chunk.title) for chunk in chunks}
    deepest = max(RECALL_DEPTHS)
    rows = []
    for case in cases:
        expected = _expected_articles(case, titles)
        if not expected or "search_mythology" not in case.expected_tools:
            continue
        ranked = [fold(chunk.title) for chunk in retriever.search(case.question, deepest)]
        rank = next((position for position, title in enumerate(ranked, start=1) if title in expected), None)
        rows.append({"id": case.id, "tags": case.tags, "question": case.question, "rank": rank,
                     "top3": ranked[:3]})
    return rows


def summarize(results: list[dict[str, Any]]) -> str:
    """Markdown summary: pass rates by category, tag and check, latency, models, and failures."""
    lines = [f"# Benchmark results ({len(results)} questions)", ""]
    lines += _rate_table("Category", _group_by(results, lambda result: [result["category"]]))
    lines += _rate_table("Tag", _group_by(results, lambda result: result["tags"]))

    check_totals: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        for check, ok in result.get("checks", {}).items():
            check_totals[check].append(ok)
    hits = [result["retrieval_hit"] for result in results if result.get("retrieval_hit") is not None]
    check_totals["retrieval_hit (model's own queries)"] = hits
    lines += ["## Checks", "", "| Check | Pass | Rate |", "|---|---|---|"]
    lines += [f"| {check} | {sum(oks)}/{len(oks)} | {_percent(oks)} |" for check, oks in check_totals.items()]

    latencies = sorted(result["latency_seconds"] for result in results)
    models = Counter(model for result in results for model in result["models"])
    errors = [result for result in results if result.get("error")]
    lines += ["", "## Latency and providers", "",
              f"- p50 {statistics.median(latencies):.1f}s, p95 {latencies[int(len(latencies) * 0.95) - 1]:.1f}s, "
              f"max {latencies[-1]:.1f}s",
              f"- LLM turns by model: {dict(models)}",
              f"- Questions needing a provider restart: {sum(result['provider_attempts'] > 1 for result in results)}; "
              f"benchmark-level retries (all providers down): {sum(result['benchmark_retries'] for result in results)}",
              f"- LLM calls per question: {sum(len(result['models']) for result in results) / len(results):.2f}",
              f"- Questions with an extra (unneeded) tool call: {sum(bool(result.get('extra_tools')) for result in results)}",
              f"- Hard errors: {len(errors)}", ""]
    lines += _token_lines(results)
    lines += _judge_lines(results)

    lines += ["## Failures", ""]
    for result in results:
        if result["passed"]:
            continue
        lines += [f"### {result['id']} — {result['question']}",
                  f"- failed: {result.get('failed_checks') or result.get('error')}",
                  f"- tools: {result.get('tools_used')}; searches: {[q['query'] for q in result['search_queries']]}; "
                  f"heroes: {result['hero_lookups']}",
                  f"- sources: {[source['title'] for source in result.get('sources', [])]}",
                  f"- answer: {(result.get('answer') or '').replace(chr(10), ' ')[:400]}",
                  *([f"- judge: {result['judge_reason']}"] if result.get("judge_reason") else []), ""]
    return "\n".join(lines)


def _token_lines(results: list[dict[str, Any]]) -> list[str]:
    """Average tokens per question and the share of prompt tokens served from the provider's cache."""
    totals = [{key: sum(call[key] for call in result["usage"]) for key in ("prompt", "completion", "cached_prompt")}
              for result in results if result["usage"]]
    if not totals:
        return []
    prompt = sum(total["prompt"] for total in totals)
    cached = sum(total["cached_prompt"] for total in totals)
    return ["## Tokens", "",
            f"- Per question (avg over {len(totals)}): {prompt / len(totals):.0f} prompt, "
            f"{sum(total['completion'] for total in totals) / len(totals):.0f} completion",
            f"- Prompt tokens served from the provider cache: {100 * cached / prompt if prompt else 0:.0f}%", ""]


def _judge_lines(results: list[dict[str, Any]]) -> list[str]:
    """Judge pass rate, and how often it agrees with the keyword checks on the same answer."""
    judged = [result for result in results if Check.JUDGE.value in result.get("checks", {})]
    if not judged:
        return []
    keyword_checks = (Check.UNKNOWN.value, Check.REFUSAL.value, Check.ANSWER.value)
    pairs = [(result["checks"][Check.JUDGE.value], result["checks"][check])
             for result in judged for check in keyword_checks if check in result["checks"]]
    agree = sum(judge_ok == keyword_ok for judge_ok, keyword_ok in pairs)
    lines = ["## Judge", "", f"- Passed {sum(result['checks'][Check.JUDGE.value] for result in judged)}/{len(judged)}"]
    if pairs:
        lines.append(f"- Agrees with the keyword check on {agree}/{len(pairs)} answers")
    return lines + [""]


def _group_by(results: list[dict[str, Any]], keys_of) -> dict[str, list[bool]]:
    """Collect pass/fail per key, where keys_of maps a result to the keys it belongs to."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        for key in keys_of(result):
            groups[key].append(result["passed"])
    return dict(sorted(groups.items()))


def _rate_table(label: str, groups: dict[str, list[bool]]) -> list[str]:
    """A markdown table of pass counts and rates."""
    rows = [f"| {key} | {sum(oks)}/{len(oks)} | {_percent(oks)} |" for key, oks in groups.items()]
    return [f"## By {label.lower()}", "", f"| {label} | Pass | Rate |", "|---|---|---|", *rows, ""]


def _percent(oks: list[bool]) -> str:
    """Pass rate as a percentage string."""
    return f"{100 * sum(oks) / len(oks):.0f}%" if oks else "n/a"


def _summarize_baseline(rows: list[dict[str, Any]]) -> str:
    """Recall@k table for the raw-question BM25 baseline, plus the misses."""
    lines = [f"# BM25 baseline on raw questions ({len(rows)} questions)", "", "| k | Recall |", "|---|---|"]
    for depth in RECALL_DEPTHS:
        hits = [row["rank"] is not None and row["rank"] <= depth for row in rows]
        lines.append(f"| {depth} | {sum(hits)}/{len(hits)} ({_percent(hits)}) |")
    lines += ["", "## Not in top 4", ""]
    lines += [f"- {row['id']} ({', '.join(row['tags'])}) rank={row['rank']}: {row['question']} -> top3 {row['top3']}"
              for row in rows if row["rank"] is None or row["rank"] > 4]
    return "\n".join(lines)


def main() -> None:
    """Parse arguments, run the requested mode and write results under benchmark/results/."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", help="Only run these case ids")
    parser.add_argument("--retrieval-only", action="store_true", help="BM25 recall on raw questions, no LLM calls")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait between questions")
    args = parser.parse_args()

    settings = get_settings()
    cases = [case for case in load_cases() if not args.ids or case.id in args.ids]
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.retrieval_only:
        report = _summarize_baseline(retrieval_baseline(cases, settings))
        (RESULTS_DIR / f"{stamp}-bm25-baseline.md").write_text(report, encoding="utf-8")
        print(report)
        return

    results = asyncio.run(run_benchmark(cases, settings, args.delay))
    with (RESULTS_DIR / f"{stamp}-results.jsonl").open("w", encoding="utf-8") as output:
        output.writelines(json.dumps(result, ensure_ascii=False) + "\n" for result in results)
    summary = summarize(results)
    (RESULTS_DIR / f"{stamp}-summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
