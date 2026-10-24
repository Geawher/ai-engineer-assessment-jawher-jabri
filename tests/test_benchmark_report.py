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
