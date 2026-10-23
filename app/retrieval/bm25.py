from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi

from app.retrieval.base import Chunk

_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset("""
a an and are as at be but by did do does for from had has have he her his how i in into is it
its me my of on or she so than that the their them then there these they this to was we were
what when where which who whom why will with you your about tell
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, accent-fold and split text into word tokens, dropping stopwords."""
    # Fold accents so "Mjolnir" matches "Mjölnir" and "Ragnarok" matches "Ragnarök".
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return [token for token in _TOKEN_RE.findall(folded) if token not in _STOPWORDS]


class BM25Retriever:
    """In-memory BM25 keyword retriever over a fixed set of chunks."""

    def __init__(self, chunks: list[Chunk]):
        """Build the index.

        Raises:
            ValueError: If chunks is empty (BM25 can't be built over nothing).
        """
        if not chunks:
            raise ValueError("Cannot build a BM25 index from an empty corpus")
        self._chunks = chunks
        # Title and section are indexed with the body so "Thor" ranks chunks *about* Thor.
        self._index = BM25Okapi([tokenize(f"{chunk.title} {chunk.section} {chunk.text}") for chunk in chunks])

    def search(self, query: str, k: int) -> list[Chunk]:
        """Return up to k chunks with a positive BM25 score, best first."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [self._chunks[i] for i in ranked[:k] if scores[i] > 0]
