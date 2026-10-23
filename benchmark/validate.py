"""Check that every benchmark expectation is actually answerable from the data.

Mythology keywords must appear in one of the expected articles; superhero keywords must
appear in the live API profile of an expected hero. A failure here is a bad question, not
a bad agent.

    uv run python -m benchmark.validate
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

from app.config import get_settings
from app.retrieval.loader import load_corpus
from app.sources.superhero import SuperheroClient
from benchmark.grading import Case, contains_all_groups, fold, load_cases

CATEGORIES_FROM_CORPUS = {"mythology"}
CATEGORIES_FROM_API = {"superhero"}


async def main() -> int:
    """Validate all cases; return a process exit code (1 if any case is unanswerable)."""
    settings = get_settings()
    articles = _article_texts(settings.data_dir)
    cases = load_cases()
    problems: list[str] = []

    async with httpx.AsyncClient(timeout=settings.superhero_timeout_seconds, follow_redirects=True) as http:
        superheroes = SuperheroClient(
            http,
            base_url=settings.superhero_api_base_url,
            token=settings.superhero_api_token.get_secret_value(),
            cache_ttl_seconds=settings.superhero_cache_ttl_seconds,
        )
        for case in cases:
            problems += _check_source_names(case, articles)
            if case.category in CATEGORIES_FROM_CORPUS and case.answer_keywords:
                problems += _check_corpus_answer(case, articles)
            if case.category in CATEGORIES_FROM_API and case.answer_keywords:
                problems += await _check_api_answer(case, superheroes)

    print("\n".join(problems) or f"All {len(cases)} cases are answerable from the data.")
    return 1 if problems else 0


def _article_texts(data_dir) -> dict[str, str]:
    """Full text per article title."""
    texts: dict[str, list[str]] = {}
    for chunk in load_corpus(data_dir):
        texts.setdefault(chunk.title, []).append(chunk.text)
    return {title: " ".join(parts) for title, parts in texts.items()}


def _check_source_names(case: Case, articles: dict[str, str]) -> list[str]:
    """Mythology-only cases must name sources that exist in the corpus (catches title typos)."""
    if case.category not in CATEGORIES_FROM_CORPUS:
        return []
    known = {fold(title) for title in articles}
    return [
        f"{case.id}: unknown article '{name}'"
        for group in case.expected_sources for name in group if fold(name) not in known
    ]


def _check_corpus_answer(case: Case, articles: dict[str, str]) -> list[str]:
    """Every keyword group must be found in at least one expected article."""
    candidates = [articles[title] for group in case.expected_sources for title in group if title in articles]
    missing = [group for group in case.answer_keywords if not any(contains_all_groups(text, [group]) for text in candidates)]
    return [f"{case.id}: keywords {group} not in expected articles" for group in missing]


async def _check_api_answer(case: Case, superheroes: SuperheroClient) -> list[str]:
    """Every keyword group must be found in the profile of some hero the expected names resolve to."""
    profiles = []
    for group in case.expected_sources:
        for name in group:
            profiles += [json.dumps(hero.profile, ensure_ascii=False) for hero in await superheroes.search(name)]
    missing = [group for group in case.answer_keywords if not any(contains_all_groups(text, [group]) for text in profiles)]
    return [f"{case.id}: keywords {group} not in API profiles" for group in missing]


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
