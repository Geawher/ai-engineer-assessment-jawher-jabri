from __future__ import annotations

import re
from collections import defaultdict

from app.retrieval.base import Chunk
from app.retrieval.bm25 import tokenize

_QUALIFIER_RE = re.compile(r"\s*\(.*?\)")  # "Hel (mythological being)" -> "Hel"


class TitleIndex:
    """Article names per collection, so the agent can tell when a name exists in more than one source.

    Each article is known by its title without a parenthetical qualifier, and, for epithet titles like
    "Ajax the Great", also by the single name before " the " ("Ajax"). Longer phrases before " the "
    ("Return of the King") are not split, so they can't produce fragment aliases.
    """

    def __init__(self, chunks: list[Chunk]):
        """Index every distinct article title in chunks."""
        self._collections: dict[str, set[str]] = defaultdict(set)
        self._display: dict[str, str] = {}  # key -> the alias as written, used for lookups in other sources
        for chunk in chunks:
            base = _QUALIFIER_RE.sub("", chunk.title).strip()
            head = base.split(" the ")[0]
            for alias in {base, head} if " " not in head else {base}:
                key = _key(alias)
                self._collections[key].add(chunk.collection)
                self._display.setdefault(key, alias)

    def collections_for(self, name: str) -> list[str]:
        """Collections with an article named name (case- and accent-insensitive), sorted."""
        return sorted(self._collections.get(_key(name), ()))

    def names_in(self, text: str) -> list[str]:
        """Article names (or aliases, e.g. "Ajax") mentioned as whole words in text, in order of first appearance."""
        padded = f" {_key(text)} "
        found = [(padded.find(f" {key} "), self._display[key]) for key in self._display if f" {key} " in padded]
        return list(dict.fromkeys(name for _, name in sorted(found)))


def _key(text: str) -> str:
    """Normalized form for matching: folded tokens joined by single spaces (stopwords included)."""
    return " ".join(re.findall(r"\w+", " ".join(tokenize(text, keep_stopwords=True))))
