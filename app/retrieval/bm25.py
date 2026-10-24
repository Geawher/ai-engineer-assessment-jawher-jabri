from __future__ import annotations

import re
import unicodedata
from itertools import islice

from rank_bm25 import BM25Okapi

from app.retrieval.base import Chunk

_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset("""
a an and are as at be but by did do does for from had has have he her his how i in into is it
its me my of on or she so than that the their them then there these they this to was we were
what when where which who whom why will with you your about tell
""".split())
# Letters that are not accented forms of a Latin letter, so Unicode decomposition (NFKD) leaves them alone.
# Without this, "Höðr" is indexed as "hoðr" and never matches a search for "Hodr", nor "Þórr" one for "Thorr".
_LETTER_FOLDS = str.maketrans({"ð": "d", "þ": "th", "æ": "ae", "ø": "o", "œ": "oe", "ß": "ss", "đ": "d", "ł": "l"})


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lowercase, accent-fold and split text into word tokens, dropping stopwords unless asked to keep them."""
    # Fold accents so "Mjolnir" matches "Mjölnir" and "Ragnarok" matches "Ragnarök".
    folded = unicodedata.normalize("NFKD", text.lower().translate(_LETTER_FOLDS))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return [token for token in _TOKEN_RE.findall(folded) if keep_stopwords or token not in _STOPWORDS]


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

    def search(self, query: str, k: int, collection: str | None = None) -> list[Chunk]:
        """Return up to k chunks with a positive BM25 score, best first, optionally from one collection.

        One index covers every collection so scores stay comparable when searching them all.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        matches = (self._chunks[i] for i in ranked
                   if scores[i] > 0 and collection in (None, self._chunks[i].collection))
        return list(islice(matches, k))
