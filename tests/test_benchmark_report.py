from __future__ import annotations

import pytest

from benchmark.grading import Case
from benchmark.report import _mrr, _source_set, behaviour


def make_case(**fields) -> Case:
    """A case with defaults; tests set only what drives the behaviour label."""
    return Case(**({"id": "t", "category": "test", "tags": [], "question": "q", "expected_tools": []} | fields))


@pytest.mark.parametrize(("fields", "label"), [
    ({"tags": ["prompt-injection"], "expected_tools": ["search_documents"]}, "Resist prompt injection"),
    ({"expect_refusal": True}, "Decline off-topic requests"),
    ({"tags": ["off-topic-entity"]}, "Decline off-topic requests"),
    ({"tags": ["leading"], "expected_tools": ["get_superhero"]}, "Correct false or leading premises"),
    ({"tags": ["outside-knowledge"]}, "Refuse outside knowledge"),
    ({"expect_unknown": True, "expected_tools": ["get_superhero"]}, "Say it doesn't know"),
    ({"expected_tools": ["search_documents", "get_superhero"]}, "Answer from both sources"),
    ({"expected_tools": ["get_superhero"]}, "Answer from superhero data"),
    ({"expected_tools": ["search_documents"]}, "Answer from documents"),
])
def test_behaviour_labels_first_match_wins(fields, label):
    """Holding-back behaviours take priority over the answer-from-a-source labels."""
    assert behaviour(make_case(**fields)) == label


def test_mrr_counts_misses_as_zero():
    """Ranks 1, 2 and a miss give (1 + 0.5 + 0) / 3."""
    assert _mrr([1, 2, None]) == pytest.approx(0.5)


@pytest.mark.parametrize(("tools", "name"), [
    (["search_documents"], "documents"), (["get_superhero"], "superhero"),
    (["get_superhero", "search_documents"], "both"), ([], "none"),
])
def test_source_set_names(tools, name):
    """Tool sets map to the routing table's row and column names."""
    assert _source_set(tools) == name


def test_quota_stop_is_detected_only_for_provider_outages():
    """Only "all providers unavailable" stops a run; agent failures and other errors are recorded."""
    from benchmark.run import is_out_of_quota

    assert is_out_of_quota({"error": "LLMUnavailableError: All LLM providers are unavailable"})
    assert not is_out_of_quota({"error": "LLMError: LLM provider x rejected the request"})
    assert not is_out_of_quota({"passed": False, "failed_checks": ["answer"]})


def test_load_results_reads_appended_rows(tmp_path):
    """A results file written line by line reads back in order; a missing file is empty."""
    from benchmark.run import load_results

    path = tmp_path / "x-results.jsonl"
    assert load_results(path) == []
    path.write_text('{"id": "a"}\n{"id": "b"}\n')
    assert [row["id"] for row in load_results(path)] == ["a", "b"]


def test_summary_never_overwrites_the_results_file(tmp_path):
    """Results files named any way get a separate summary file."""
    from benchmark.run import summary_path_for

    assert summary_path_for(tmp_path / "x-results.jsonl").name == "x-summary.md"
    assert summary_path_for(tmp_path / "foo.jsonl").name == "foo-summary.md"
    assert summary_path_for(tmp_path / "foo.jsonl") != tmp_path / "foo.jsonl"


def test_write_results_is_atomic_and_round_trips(tmp_path):
    """Rewriting goes through a temporary file and leaves no leftovers."""
    from benchmark.run import load_results, write_results

    path = tmp_path / "x-results.jsonl"
    write_results(path, [{"id": "a"}, {"id": "b"}])
    assert [row["id"] for row in load_results(path)] == ["a", "b"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x-results.jsonl"]


def test_quota_flag_takes_precedence_over_error_text():
    """New rows carry an explicit flag; old rows fall back to the error prefix."""
    from benchmark.run import is_out_of_quota

    assert not is_out_of_quota({"error": "LLMUnavailableError: x", "out_of_quota": False})
    assert is_out_of_quota({"error": "something", "out_of_quota": True})


def test_cost_lines_handle_no_results():
    """An empty results list produces no cost section instead of crashing."""
    from benchmark.report import _cost_lines

    assert _cost_lines([]) == []


async def test_regrade_drops_judge_verdicts_for_removed_rubrics():
    """A stored judge failure no longer counts once the case has no rubric."""
    from benchmark.grading import Case
    from benchmark.run import regrade

    case = Case(id="q", category="test", tags=[], question="q", expected_tools=[])
    row = {"id": "q", "answer": "a", "sources": [], "tools_used": [], "warnings": [],
           "checks": {"route": True, "judge": False}}
    await regrade([row], {"q": case}, {}, set())
    assert row["passed"] and "judge" not in row["checks"]


def test_work_time_takes_rate_limit_and_retry_waits_out_of_latency():
    """A question that waited 8s on a rate limit and 20s on a benchmark retry worked for 2s, 1.5s of it in the LLM."""
    from benchmark.report import _work_time_lines

    result = {"latency_seconds": 30.0, "rate_limit_waits": [0.0, 8.0], "llm_seconds": [0.5, 9.0],
              "benchmark_wait_seconds": 20.0}
    lines = _work_time_lines([result])
    assert lines[0].startswith("- Latency without rate-limit waits: p50 2.0s") and "75% of it" in lines[0]
    assert lines[1] == "- Rate-limit waits: 1 questions, 28s in total"
    assert _work_time_lines([{"latency_seconds": 3.0}]) == []  # results recorded before timing existed
