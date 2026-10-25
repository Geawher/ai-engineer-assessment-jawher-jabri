"""Turn benchmark results into a markdown report: behaviour rates, routing, citations, retrieval, cost, failures."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any, NamedTuple

from app.agent import ToolName
from app.config import Settings
from app.schemas import SourceType
from benchmark.grading import Case, Check, fold

RECALL_DEPTHS = (1, 3, 4, 10)
TOP_K = Settings.model_fields["retrieval_top_k"].default  # passages per search the service returns
Result = dict[str, Any]


class Tag(StrEnum):
    """Question tags (in questions.jsonl) that decide a case's expected behaviour."""

    PROMPT_INJECTION = "prompt-injection"
    OFF_TOPIC_ENTITY = "off-topic-entity"
    FALSE_PREMISE = "false-premise"
    LEADING = "leading"
    OUTSIDE_KNOWLEDGE = "outside-knowledge"
    GROUNDING = "grounding"
    UNANSWERABLE = "unanswerable"
    NONEXISTENT = "nonexistent"
    MISSING_HERO = "missing-hero"
    NUMBER_BAIT = "number-bait"
    AGGREGATE = "aggregate"


class SourceSet(StrEnum):
    """Which sources a question needed or used, for the routing table."""

    DOCUMENTS = "documents"
    SUPERHERO = "superhero"
    BOTH = "both"
    NONE = "none"


class Behaviour(NamedTuple):
    label: str
    meaning: str
    holds_back: bool  # passing means NOT producing something (declining, "I don't know", resisting)
    applies: Callable[[Case], bool]


def _tagged(case: Case, *tags: Tag) -> bool:
    """Whether the case has any of these tags."""
    return bool(set(tags) & set(case.tags))


# Expected behaviour, decided from each case's expectations and tags. Order matters: the first match wins,
# so a prompt injection about Thor counts as "resist injection", not "answer from documents".
BEHAVIOURS = [
    Behaviour("Resist prompt injection", "answers the real question (if any) and reveals nothing", True,
              lambda case: _tagged(case, Tag.PROMPT_INJECTION)),
    Behaviour("Decline off-topic requests", "declines without making anything up", True,
              lambda case: case.expect_refusal or _tagged(case, Tag.OFF_TOPIC_ENTITY)),
    Behaviour("Correct false or leading premises", "contradicts the premise from the sources", True,
              lambda case: _tagged(case, Tag.FALSE_PREMISE, Tag.LEADING)),
    Behaviour("Refuse outside knowledge", "says the sources don't cover it instead of using what the model knows", True,
              lambda case: _tagged(case, Tag.OUTSIDE_KNOWLEDGE, Tag.GROUNDING)),
    Behaviour("Say it doesn't know", "says the sources don't contain the answer instead of inventing one", True,
              lambda case: case.expect_unknown or _tagged(case, Tag.UNANSWERABLE, Tag.NONEXISTENT, Tag.MISSING_HERO,
                                                          Tag.NUMBER_BAIT, Tag.AGGREGATE)),
    Behaviour("Answer from both sources", "correct answer citing a document passage and the superhero data", False,
              lambda case: set(case.expected_tools) == {ToolName.SEARCH_DOCUMENTS, ToolName.GET_SUPERHERO}),
    Behaviour("Answer from superhero data", "correct answer citing the superhero data", False,
              lambda case: case.expected_tools == [ToolName.GET_SUPERHERO]),
    Behaviour("Answer from documents", "correct answer citing the right document passage", False,
              lambda case: case.expected_tools == [ToolName.SEARCH_DOCUMENTS]),
]
# Checks whose failure means the bot said something it shouldn't have (as opposed to, say, a routing miss).
_CONTENT_CHECKS = {Check.ANSWER, Check.UNKNOWN, Check.REFUSAL, Check.FORBIDDEN_TEXT, Check.JUDGE}


def behaviour(case: Case) -> str:
    """The behaviour a case tests (see BEHAVIOURS)."""
    return next((item.label for item in BEHAVIOURS if item.applies(case)), "Other")


def summarize(results: list[Result], cases: dict[str, Case], article_titles: set[str],
              retrieval_rows: list[Result]) -> str:
    """The full markdown report.

    Args:
        results: Rows from a *-results.jsonl file.
        cases: Benchmark cases by id (for expectations the results don't store).
        article_titles: Folded titles of every corpus article, to tell articles from hero names.
        retrieval_rows: Output of retrieval_baseline(): BM25 ranks for the raw questions.
    """
    lines = [f"# Benchmark results ({len(results)} questions)", ""]
    lines += _headline(results)
    lines += _behaviour_lines(results, cases)
    lines += _routing_lines(results, cases)
    lines += _citation_lines(results)
    lines += _retrieval_lines(results, cases, article_titles, retrieval_rows)
    lines += _rate_table("Category", _group_by(results, lambda result: [result["category"]]))
    lines += _rate_table("Tag", _group_by(results, lambda result: result["tags"]))
    lines += _check_lines(results)
    lines += _cost_lines(results)
    lines += _judge_lines(results)
    lines += _failure_lines(results)
    return "\n".join(lines)


def summarize_retrieval(rows: list[Result]) -> str:
    """A standalone report for --retrieval-only runs."""
    return "\n".join([f"# BM25 retrieval on raw questions ({len(rows)} questions)", "",
                      *_raw_retrieval_table(rows), "", f"## Not in the top {TOP_K}", "",
                      *[f"- {row['id']} rank={row['rank']}: {row['question']} -> top 3 {row['top3']}"
                        for row in rows if row["rank"] is None or row["rank"] > TOP_K]])


def _headline(results: list[Result]) -> list[str]:
    """One line with the overall pass rate and hard errors."""
    passed = [result["passed"] for result in results]
    errors = sum(bool(result.get("error")) for result in results)
    return [f"**{sum(passed)}/{len(passed)} passed ({_percent(passed)})**, {errors} hard errors.", ""]


def _behaviour_lines(results: list[Result], cases: dict[str, Case]) -> list[str]:
    """Pass rate per expected behaviour, plus how often the bot said something it should have held back."""
    groups: dict[str, list[bool]] = defaultdict(list)
    holding_back: list[Result] = []
    holds_back = {item.label for item in BEHAVIOURS if item.holds_back}
    for result in results:
        label = behaviour(cases[result["id"]])
        groups[label].append(result["passed"])
        if label in holds_back:
            holding_back.append(result)
    lines = ["## By expected behaviour", "", "| Behaviour | Behaved correctly | Rate | Passing means |", "|---|---|---|---|"]
    lines += [f"| {item.label} | {sum(oks)}/{len(oks)} | {_percent(oks)} | {item.meaning} |"
              for item in BEHAVIOURS if (oks := groups.get(item.label))]
    if holding_back:
        said_too_much = sum(bool(_CONTENT_CHECKS & set(result.get("failed_checks", []))) for result in holding_back)
        lines += ["", f"When it should have declined, said it doesn't know, or resisted a premise or injection, "
                      f"it said something it shouldn't have {said_too_much}/{len(holding_back)} times "
                      f"({100 * said_too_much / len(holding_back):.0f}%)."]
    return lines + [""]


def _source_set(tools: Iterable[str]) -> SourceSet:
    """Which sources a set of tools covers."""
    tools = set(tools)
    if {ToolName.SEARCH_DOCUMENTS, ToolName.GET_SUPERHERO} <= tools:
        return SourceSet.BOTH
    if ToolName.SEARCH_DOCUMENTS in tools:
        return SourceSet.DOCUMENTS
    if ToolName.GET_SUPERHERO in tools:
        return SourceSet.SUPERHERO
    return SourceSet.NONE


def _routing_lines(results: list[Result], cases: dict[str, Case]) -> list[str]:
    """Which sources each question needed (rows) against which it used (columns)."""
    columns = list(SourceSet)
    matrix: dict[str, Counter] = defaultdict(Counter)
    for result in results:
        if result.get("error"):
            continue
        matrix[_source_set(cases[result["id"]].expected_tools)][_source_set(result.get("tools_used", []))] += 1
    lines = ["## Routing (needed → used)", "", "| Needed \\ Used | " + " | ".join(columns) + " |",
             "|---" * (len(columns) + 1) + "|"]
    lines += [f"| {row} | " + " | ".join(str(matrix[row][column]) for column in columns) + " |"
              for row in columns if matrix[row]]
    extra = sum(bool(result.get("extra_tools")) for result in results)
    return lines + ["", f"Questions with an unneeded extra lookup (costs tokens, not counted as a failure): {extra}", ""]


def _citation_lines(results: list[Result]) -> list[str]:
    """How answers were attributed."""
    answered = [result for result in results if not result.get("error")]
    real = [any(source["type"] != SourceType.LLM for source in result["sources"]) for result in answered]
    sources_checked = [result["checks"][Check.SOURCES.value] for result in answered if Check.SOURCES.value in result["checks"]]
    counts = [sum(source["type"] != SourceType.LLM for source in result["sources"]) for result in answered]
    return ["## Citations", "",
            f"- Answers backed by at least one real source: {sum(real)}/{len(real)} ({_percent(real)}); "
            f"the rest are attributed to the model itself (refusals, \"I don't know\", off-topic)",
            f"- Answers that cited the expected source(s): {sum(sources_checked)}/{len(sources_checked)} "
            f"({_percent(sources_checked)})",
            f"- Sources per answer: {statistics.mean(counts):.1f} on average" if counts else "", ""]


def _retrieval_lines(results: list[Result], cases: dict[str, Case], article_titles: set[str],
                     raw_rows: list[Result]) -> list[str]:
    """BM25 quality: on the raw questions (no LLM), and with the queries the model actually wrote."""
    lines = ["## BM25 retrieval", "", "On the raw question text (no LLM involved):", "", *_raw_retrieval_table(raw_rows), ""]
    best_ranks, empty_searches, searches = [], 0, []
    for result in results:
        case = cases[result["id"]]
        expected = {fold(name) for group in case.expected_sources for name in group} & article_titles
        queries = result.get("search_queries", [])
        searches.append(len(queries))
        empty_searches += sum(not query["titles"] for query in queries)
        if not expected or not queries or ToolName.SEARCH_DOCUMENTS not in case.expected_tools:
            continue
        ranks = [rank for query in queries for rank, title in enumerate(query["titles"], start=1) if fold(title) in expected]
        best_ranks.append(min(ranks) if ranks else None)
    if not best_ranks:
        return lines
    found = [rank is not None for rank in best_ranks]
    rank_counts = Counter(rank for rank in best_ranks if rank is not None)
    lines += ["With the model's own search queries:", "",
              f"- Expected article retrieved: {sum(found)}/{len(found)} ({_percent(found)}); "
              f"best rank: " + ", ".join(f"#{rank}: {count}" for rank, count in sorted(rank_counts.items())),
              f"- Mean reciprocal rank: {_mrr(best_ranks):.2f}",
              f"- Searches per question that searched: "
              f"{statistics.mean([count for count in searches if count]):.1f}; searches returning nothing: {empty_searches}",
              ""]
    return lines


def _raw_retrieval_table(rows: list[Result]) -> list[str]:
    """Recall@k and MRR for BM25 on the raw question text."""
    if not rows:
        return ["(no rows)"]
    ranks = [row["rank"] for row in rows]
    table = ["| Metric | Value |", "|---|---|"]
    for depth in RECALL_DEPTHS:
        hits = [rank is not None and rank <= depth for rank in ranks]
        table.append(f"| Recall@{depth} | {sum(hits)}/{len(hits)} ({_percent(hits)}) |")
    return table + [f"| MRR | {_mrr(ranks):.2f} |"]


def _mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank; a miss counts as 0."""
    return statistics.mean(1 / rank if rank else 0 for rank in ranks) if ranks else 0.0


def _check_lines(results: list[Result]) -> list[str]:
    """Pass rate per individual check."""
    totals: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        for check, ok in result.get("checks", {}).items():
            totals[check].append(ok)
    return ["## Checks", "", "| Check | Pass | Rate |", "|---|---|---|",
            *[f"| {check} | {sum(oks)}/{len(oks)} | {_percent(oks)} |" for check, oks in totals.items()], ""]


def _cost_lines(results: list[Result]) -> list[str]:
    """Latency, provider usage, retries and tokens."""
    if not results:
        return []
    latencies = sorted(result["latency_seconds"] for result in results)
    models = Counter(model for result in results for model in result["models"])
    lines = ["## Cost and latency", "",
             f"- Latency: p50 {statistics.median(latencies):.1f}s, p95 {latencies[math.ceil(len(latencies) * 0.95) - 1]:.1f}s, "
             f"max {latencies[-1]:.1f}s",
             f"- LLM calls per question: {sum(len(result['models']) for result in results) / len(results):.2f}; "
             f"by model: {dict(models)}",
             f"- Questions that needed a provider restart: {sum(result['provider_attempts'] > 1 for result in results)}; "
             f"benchmark retries (all providers down): {sum(result['benchmark_retries'] for result in results)}"]
    lines += _work_time_lines(results)
    totals = [{key: sum(call[key] for call in result.get("usage", [])) for key in ("prompt", "completion", "cached_prompt")}
              for result in results if result.get("usage")]
    if totals:
        prompt = sum(total["prompt"] for total in totals)
        cached = sum(total["cached_prompt"] for total in totals)
        lines += [f"- Tokens per question: {prompt / len(totals):.0f} prompt, "
                  f"{sum(total['completion'] for total in totals) / len(totals):.0f} completion; "
                  f"{100 * cached / prompt if prompt else 0:.0f}% of prompt tokens served from the provider cache"]
    return lines + [""]


def _work_time_lines(results: list[Result]) -> list[str]:
    """Latency with rate-limit and retry waits taken out, split into LLM and everything else.

    Waits depend on the free tier's quota, not on the pipeline, so comparing pipeline changes needs them removed.
    Results files written before timing was recorded have nothing to report.
    """
    timed = [result for result in results if "rate_limit_waits" in result]
    if not timed:
        return []
    waits = [sum(result["rate_limit_waits"]) + result.get("benchmark_wait_seconds", 0) for result in timed]
    work = sorted(result["latency_seconds"] - wait for result, wait in zip(timed, waits, strict=True))
    llm = sum(sum(result["llm_seconds"]) - sum(result["rate_limit_waits"]) for result in timed)
    return [f"- Latency without rate-limit waits: p50 {statistics.median(work):.1f}s, "
            f"p95 {work[math.ceil(len(work) * 0.95) - 1]:.1f}s, max {work[-1]:.1f}s; "
            f"LLM calls are {100 * llm / sum(work):.0f}% of it",
            f"- Rate-limit waits: {sum(wait > 0 for wait in waits)} questions, {sum(waits):.0f}s in total"]


def _judge_lines(results: list[Result]) -> list[str]:
    """Judge pass rate, and how often it agrees with the keyword checks on the same answer."""
    judged = [result for result in results if Check.JUDGE.value in result.get("checks", {})]
    if not judged:
        return []
    keyword_checks = (Check.UNKNOWN.value, Check.REFUSAL.value, Check.ANSWER.value)
    pairs = [(result["checks"][Check.JUDGE.value], result["checks"][check])
             for result in judged for check in keyword_checks if check in result["checks"]]
    lines = ["## LLM judge", "", f"- Passed {sum(result['checks'][Check.JUDGE.value] for result in judged)}/{len(judged)}"]
    if pairs:
        agree = sum(judge_ok == keyword_ok for judge_ok, keyword_ok in pairs)
        lines.append(f"- Agrees with the keyword check on {agree}/{len(pairs)} answers")
    return lines + [""]


def _failure_lines(results: list[Result]) -> list[str]:
    """Every failure with the tool calls, sources, answer and judge reason behind it."""
    lines = ["## Failures", ""]
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
    return lines


def _group_by(results: list[Result], keys_of: Callable[[Result], list[str]]) -> dict[str, list[bool]]:
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
