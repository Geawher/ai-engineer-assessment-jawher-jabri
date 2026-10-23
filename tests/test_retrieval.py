from __future__ import annotations

from app.retrieval.bm25 import tokenize
from app.retrieval.loader import MAX_CHUNK_CHARS, parse_article

ARTICLE = """# Thor
Source: https://en.wikipedia.org/wiki/Thor

## Introduction
Thor is a prominent god in Germanic paganism.

## Name
The name Thor derives from Old Norse Þórr.

## Attributes
""" + ("Thor carries the hammer Mjölnir. " * 80) + """

## Name
A second section that reuses an earlier heading.
"""


def test_tokenize_folds_accents_and_drops_stopwords():
    """Accented and unaccented spellings produce the same tokens; stopwords are removed."""
    assert tokenize("The hammer Mjölnir at Ragnarök") == ["hammer", "mjolnir", "ragnarok"]


def test_parse_article_splits_by_section_with_anchors():
    """Chunks link to their section, respect the size cap and get unique ids despite repeated headings."""
    chunks = parse_article("thor", ARTICLE, "mythology")

    assert chunks[0].section == "Introduction"
    assert chunks[0].url == "https://en.wikipedia.org/wiki/Thor"
    assert chunks[1].url == "https://en.wikipedia.org/wiki/Thor#Name"
    attributes = [chunk for chunk in chunks if chunk.section == "Attributes"]
    assert len(attributes) > 1 and all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in attributes)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_bm25_ranks_relevant_chunk_first(retriever):
    """The chunk about the queried subject ranks first."""
    assert retriever.search("Mjolnir hammer", k=2)[0].title == "Thor"
    assert retriever.search("king of the gods Olympus", k=2)[0].title == "Zeus"


def test_bm25_returns_nothing_for_unrelated_query(retriever):
    """Queries with no overlapping terms (or only stopwords) return no chunks."""
    assert retriever.search("quantum chromodynamics", k=3) == []
    assert retriever.search("the and of", k=3) == []
