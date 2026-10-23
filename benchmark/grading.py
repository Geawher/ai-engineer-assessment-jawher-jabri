"""Benchmark cases and the pure functions that grade an agent response against them."""

from __future__ import annotations

import json
import unicodedata
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import AskResponse

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
_SUPERHERO_TITLE_SUFFIX = " (Superhero API"
_MATCHING_FIXES = str.maketrans({"-": None, "\u2010": None, "\u2011": None, "\u200b": None})
_SECTION_SEPARATOR = " — "

# Phrases that signal "the sources don't say" rather than an invented answer.
_UNKNOWN_MARKERS = (
    "don't know", "do not know", "not found", "no information", "couldn't find", "could not find",
    "not available", "unable", "doesn't contain", "does not contain", "do not contain", "don't contain",
    "no superhero", "not mention", "don't have", "do not have", "no record", "cannot", "can't",
    "not provide", "doesn't provide", "does not provide", "no data", "not specif", "no mention",
    "isn't mentioned", "is not mentioned", "no details", "not included", "only answer", "doesn't say",
    "does not say", "no evidence", "not listed", "there is no", "not known", "unknown", "not aware",
    "no account", "do not give", "does not give", "doesn't give", "don't give", "not recorded",
)
# A refusal either names the bot's scope or declines outright (e.g. to a prompt injection).
_REFUSAL_MARKERS = ("mytholog", "superhero", "comic", "can't", "cannot", "won't", "unable", "sorry")


class Check(StrEnum):
    ROUTE = "route"
    ANSWER = "answer"
    SOURCES = "sources"
    UNKNOWN = "unknown"
    REFUSAL = "refusal"
    FORBIDDEN_TEXT = "forbidden_text"
    JUDGE = "judge"


class Case(BaseModel):
    """One benchmark question with its expectations. Keyword and source lists are any-of groups."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: str
    tags: list[str]
    question: str
    expected_tools: list[str]
    optional_tools: list[str] = Field(default_factory=list)
    expected_sources: list[list[str]] = Field(default_factory=list)
    expected_source_types: list[str] = Field(default_factory=list)
    answer_keywords: list[list[str]] = Field(default_factory=list)
    expect_refusal: bool = False
    expect_unknown: bool = False
    forbidden_text: list[str] = Field(default_factory=list)
    # Rubric for the LLM judge, for behaviour keywords can't grade (declining, not inventing, correcting a premise).
    judge: str | None = None


def load_cases(path: Path = QUESTIONS_PATH) -> list[Case]:
    """Parse the JSONL benchmark file."""
    with path.open(encoding="utf-8") as lines:
        return [Case.model_validate(json.loads(line)) for line in lines if line.strip()]


def fold(text: str) -> str:
    """Normalize for matching: lowercase, no accents ("Höðr" -> "hodr"), straight apostrophes, no hyphens.

    Hyphens are dropped because models hyphenate freely ("mistle-toe", "Spider-Man" vs "Spiderman").
    """
    decomposed = unicodedata.normalize("NFKD", text.lower().replace("’", "'").replace("ð", "d"))
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_accents.translate(_MATCHING_FIXES)


def source_subject(title: str) -> str:
    """The article or hero a source title refers to, e.g. "Thor — Introduction" -> "Thor"."""
    return title.split(_SECTION_SEPARATOR)[0].split(_SUPERHERO_TITLE_SUFFIX)[0]


def contains_all_groups(text: str, groups: list[list[str]]) -> bool:
    """Whether every group has at least one alternative appearing in text (accent-insensitive)."""
    folded = fold(text)
    return all(any(fold(alternative) in folded for alternative in group) for group in groups)


def grade(case: Case, response: AskResponse) -> dict[Check, bool]:
    """Run every check that applies to the case.

    Returns:
        Check -> passed, containing only the checks relevant to this case.
    """
    used = set(response.tools_used)
    required = set(case.expected_tools)
    # Extra lookups cost tokens but aren't wrong (most names exist in both sources); they're counted
    # separately by extra_tools(). Only questions that need no tools at all must use none.
    allowed_when_none_required = set(case.optional_tools)
    route_ok = required <= used if required else used <= allowed_when_none_required
    results = {Check.ROUTE: route_ok}

    if case.answer_keywords:
        results[Check.ANSWER] = contains_all_groups(response.answer, case.answer_keywords)
    if case.expected_sources or case.expected_source_types:
        results[Check.SOURCES] = _sources_match(case, response)
    if case.expect_unknown:
        results[Check.UNKNOWN] = any(marker in fold(response.answer) for marker in _UNKNOWN_MARKERS)
    if case.expect_refusal:
        # Same tool rule as routing: a lookup the case allows (e.g. checking who Spider-Man is before
        # declining to give a recipe) doesn't spoil a refusal.
        results[Check.REFUSAL] = route_ok and any(marker in fold(response.answer) for marker in _REFUSAL_MARKERS)
    if case.forbidden_text:
        results[Check.FORBIDDEN_TEXT] = not any(fold(text) in fold(response.answer) for text in case.forbidden_text)
    return results


def extra_tools(case: Case, response: AskResponse) -> list[str]:
    """Tools the agent used beyond the expected and optional ones (an efficiency signal, not a failure)."""
    return sorted(set(response.tools_used) - set(case.expected_tools) - set(case.optional_tools))


def _sources_match(case: Case, response: AskResponse) -> bool:
    """Every expected source group is cited, and every expected source type is present."""
    subjects = [fold(source_subject(source.title)) for source in response.sources]
    groups_ok = all(any(fold(name) in subjects for name in group) for group in case.expected_sources)
    types_ok = set(case.expected_source_types) <= {source.type.value for source in response.sources}
    return groups_ok and types_ok
