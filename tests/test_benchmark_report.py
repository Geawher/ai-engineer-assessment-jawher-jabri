from __future__ import annotations

import pytest

from benchmark.grading import Case
from benchmark.report import _mrr, _source_set, behaviour


def make_case(**fields) -> Case:
    """A case with defaults; tests set only what drives the behaviour label."""
    return Case(**({"id": "t", "category": "test", "tags": [], "question": "q", "expected_tools": []} | fields))


@pytest.mark.parametrize(("fields", "label"), [
    ({"tags": ["prompt-injection"], "expected_tools": ["search_mythology"]}, "Resist prompt injection"),
    ({"expect_refusal": True}, "Decline off-topic requests"),
    ({"tags": ["off-topic-entity"]}, "Decline off-topic requests"),
    ({"tags": ["leading"], "expected_tools": ["get_superhero"]}, "Correct false or leading premises"),
    ({"tags": ["outside-knowledge"]}, "Refuse outside knowledge"),
    ({"expect_unknown": True, "expected_tools": ["get_superhero"]}, "Say it doesn't know"),
    ({"expected_tools": ["search_mythology", "get_superhero"]}, "Answer from both sources"),
    ({"expected_tools": ["get_superhero"]}, "Answer from superhero data"),
    ({"expected_tools": ["search_mythology"]}, "Answer from mythology"),
])
def test_behaviour_labels_first_match_wins(fields, label):
    """Holding-back behaviours take priority over the answer-from-a-source labels."""
    assert behaviour(make_case(**fields)) == label


def test_mrr_counts_misses_as_zero():
    """Ranks 1, 2 and a miss give (1 + 0.5 + 0) / 3."""
    assert _mrr([1, 2, None]) == pytest.approx(0.5)


@pytest.mark.parametrize(("tools", "name"), [
    (["search_mythology"], "mythology"), (["get_superhero"], "superhero"),
    (["get_superhero", "search_mythology"], "both"), ([], "none"),
])
def test_source_set_names(tools, name):
    """Tool sets map to the routing table's row and column names."""
    assert _source_set(tools) == name
