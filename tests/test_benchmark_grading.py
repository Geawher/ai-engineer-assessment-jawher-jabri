from __future__ import annotations

import pytest

from app.schemas import AskResponse, Source, SourceType
from benchmark.grading import Case, Check, contains_all_groups, extra_tools, fold, grade, source_subject


def make_case(**fields) -> Case:
    """A case with harmless defaults; tests override only what they exercise."""
    defaults = {"id": "t-1", "category": "test", "tags": [], "question": "q", "expected_tools": []}
    return Case(**(defaults | fields))


def make_response(answer: str = "", tools: list[str] | None = None, sources: list[Source] | None = None) -> AskResponse:
    """An agent response with the given answer, tools and sources."""
    return AskResponse(answer=answer, sources=sources or [], tools_used=tools or [])


def source(title: str, source_type: SourceType = SourceType.DOCUMENT) -> Source:
    """A source with only the fields grading looks at."""
    return Source(id="S1", type=source_type, title=title)


@pytest.mark.parametrize(("raw", "folded"), [
    ("Höðr", "hodr"),
    ("Mjölnir", "mjolnir"),
    ("mistle‑toe", "mistletoe"),  # the non-breaking hyphen gpt-oss writes
    ("Spider-Man", "spiderman"),
    ("Thor’s", "thor's"),
    ("zero​width", "zerowidth"),
])
def test_fold_normalizes_what_models_vary(raw, folded):
    """Accents, hyphen variants, curly apostrophes and invisible characters don't affect matching."""
    assert fold(raw) == folded


def test_keyword_groups_are_all_of_any_of():
    """Every group must match, through any one of its alternatives."""
    groups = [["Höðr", "Hodr"], ["mistletoe"]]
    assert contains_all_groups("Hodr threw the mistle-toe", groups)
    assert not contains_all_groups("Hodr threw a spear", groups)


@pytest.mark.parametrize(("title", "subject"), [
    ("Thor — Introduction", "Thor"),
    ("Batman (Superhero API #70)", "Batman"),
    ("Hel (mythological being) — Prose Edda", "Hel (mythological being)"),
])
def test_source_subject(title, subject):
    """Section and API suffixes are stripped, article titles with parentheses are kept."""
    assert source_subject(title) == subject


class TestRoute:
    def test_required_tools_must_be_used(self):
        """Missing a required tool fails routing."""
        case = make_case(expected_tools=["search_documents", "get_superhero"])
        assert not grade(case, make_response(tools=["search_documents"]))[Check.ROUTE]

    def test_extra_tool_is_allowed_but_reported(self):
        """An unneeded lookup passes routing but shows up in extra_tools."""
        case = make_case(expected_tools=["search_documents"])
        response = make_response(tools=["search_documents", "get_superhero"])
        assert grade(case, response)[Check.ROUTE]
        assert extra_tools(case, response) == ["get_superhero"]

    def test_out_of_scope_must_use_no_tools(self):
        """When no tool is required, any tool outside optional_tools fails."""
        case = make_case(expected_tools=[], optional_tools=["get_superhero"])
        assert grade(case, make_response(tools=["get_superhero"]))[Check.ROUTE]
        assert not grade(case, make_response(tools=["search_documents"]))[Check.ROUTE]


class TestRefusalAndUnknown:
    def test_refusal_naming_the_scope(self):
        """A scope-naming refusal with no tools passes."""
        case = make_case(expect_refusal=True)
        assert grade(case, make_response("I can only answer questions about mythology."))[Check.REFUSAL]

    def test_refusal_to_prompt_injection(self):
        """A plain decline passes, as seen live: "I'm sorry, but I can't comply with that." """
        case = make_case(expect_refusal=True)
        assert grade(case, make_response("I’m sorry, but I can’t comply with that."))[Check.REFUSAL]

    def test_refusal_after_allowed_lookup(self):
        """Checking an allowed tool before declining still counts as a refusal."""
        case = make_case(expect_refusal=True, optional_tools=["get_superhero"])
        response = make_response("Spider-Man is a fictional hero, so I can't give a recipe.", tools=["get_superhero"])
        assert grade(case, response)[Check.REFUSAL]

    def test_answering_is_not_a_refusal(self):
        """An answer without decline wording fails the refusal check."""
        case = make_case(expect_refusal=True)
        assert not grade(case, make_response("Paris is the capital of France."))[Check.REFUSAL]

    @pytest.mark.parametrize("answer", [
        "The sources do not give a specific number of Heimdall’s children.",
        "I’m not aware of any myth in which Zeus kills Thor.",
        "The exact number is not known.",
    ])
    def test_unknown_phrasings_seen_live(self, answer):
        """"I don't know" phrasings that earlier grader versions missed."""
        assert grade(make_case(expect_unknown=True), make_response(answer))[Check.UNKNOWN]

    def test_invented_answer_is_not_unknown(self):
        """A confident made-up fact fails the unknown check."""
        assert not grade(make_case(expect_unknown=True), make_response("Zeus loved olives."))[Check.UNKNOWN]


class TestSources:
    def test_any_alternative_in_each_group(self):
        """Each group needs one cited subject; a valid alternative article counts."""
        case = make_case(expected_sources=[["Thor", "Mjölnir"]])
        assert grade(case, make_response(sources=[source("Mjölnir — Gylfaginning")]))[Check.SOURCES]

    def test_missing_group_fails(self):
        """A comparison citing only one side fails."""
        case = make_case(expected_sources=[["Iron Man"], ["Hulk"]])
        response = make_response(sources=[source("Iron Man (Superhero API #346)", SourceType.SUPERHERO_API)])
        assert not grade(case, response)[Check.SOURCES]

    def test_expected_source_types(self):
        """A "both" question needs one source of each type."""
        case = make_case(expected_source_types=["document", "superhero_api"])
        assert not grade(case, make_response(sources=[source("Thor — Introduction")]))[Check.SOURCES]


def test_forbidden_text():
    """Leaking a forbidden phrase fails, case- and accent-insensitively."""
    case = make_case(forbidden_text=["Answer ONLY from tool results"])
    assert not grade(case, make_response("My rules say: answer only from tool results."))[Check.FORBIDDEN_TEXT]


def test_only_relevant_checks_run():
    """A case without keywords or sources gets just the route check."""
    assert set(grade(make_case(), make_response())) == {Check.ROUTE}


class TestFixedFalseNegatives:
    @pytest.mark.parametrize("answer", [
        "Heimdall and his wife have many children according to the Poetic Edda, though no exact number is given [S1].",
        "I'm sorry, but the available data doesn't include information about Wonder Woman's favourite colour.",
    ])
    def test_unknown_phrasings_from_the_gpt_oss_120b_run(self, answer):
        """Correct "the sources don't say" answers that the keyword list used to miss."""
        assert grade(make_case(expect_unknown=True), make_response(answer))[Check.UNKNOWN]

    def test_a_cited_passage_containing_the_answer_counts(self):
        """Citing another article is fine when the cited passage itself holds the answer."""
        case = make_case(expected_sources=[["Achilles"]], answer_keywords=[["Paris"]])
        response = make_response("Paris killed Achilles [S1].", sources=[source("Apollo — Trojan War")])
        texts = {"Apollo — Trojan War": "Apollo guided the arrow Paris shot at Achilles."}
        assert grade(case, response, texts)[Check.SOURCES]

    def test_a_cited_passage_without_the_answer_still_fails(self):
        """An off-target citation isn't rescued just because the answer text is right."""
        case = make_case(expected_sources=[["Achilles"]], answer_keywords=[["Paris"]])
        response = make_response("Paris killed Achilles [S1].", sources=[source("Apollo — Cult")])
        texts = {"Apollo — Cult": "Apollo was worshipped at Delphi."}
        assert not grade(case, response, texts)[Check.SOURCES]
