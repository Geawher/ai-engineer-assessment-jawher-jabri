from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    id: str
    collection: str
    title: str
    section: str
    url: str
    text: str


class Retriever(Protocol):
    """Anything that can rank chunks for a query. BM25 today; embeddings or hybrid later."""

    def search(self, query: str, k: int, collection: str | None = None) -> list[Chunk]:
        """Return up to k chunks relevant to the query, best first, optionally from one collection only."""
        ...
