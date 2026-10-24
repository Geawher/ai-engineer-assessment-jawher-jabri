from __future__ import annotations

import json

import pytest

from app.retrieval.base import Chunk
from app.retrieval.loader import load_collections
from app.retrieval.titles import TitleIndex


def chunk(title: str, collection: str = "mythology") -> Chunk:
    """A chunk with only the fields the title index reads."""
    return Chunk(id=title, collection=collection, title=title, section="Introduction", url="", text="")


@pytest.fixture
def index() -> TitleIndex:
    """Titles from two collections, including a qualified and a "the"-style title."""
    return TitleIndex([chunk("Loki"), chunk("Hel (mythological being)"), chunk("Ajax the Great"), chunk("Mjölnir"),
                       chunk("Loki", collection="eddas")])


@pytest.mark.parametrize(("name", "collections"), [
    ("Loki", ["eddas", "mythology"]),
    ("loki", ["eddas", "mythology"]),
    ("Hel", ["mythology"]),            # parenthetical qualifier dropped
    ("Ajax", ["mythology"]),           # "Ajax the Great" also known as "Ajax"
    ("Mjolnir", ["mythology"]),        # accent-insensitive
    ("Batman", []),
])
def test_collections_for(index, name, collections):
    """Names resolve to every collection with a matching article."""
    assert index.collections_for(name) == collections


def test_names_in_finds_whole_word_titles_in_order(index):
    """Article names mentioned in free text are found as whole words, in order of appearance."""
    assert index.names_in("Mjölnir and Loki's children with Angrboda") == ["Mjölnir", "Loki"]
    assert index.names_in("Hello there") == []  # "Hel" must not match inside "Hello"


def test_load_collections_reads_folders_with_a_description(tmp_path):
    """Each folder with collection.json is a collection; folders without one are ignored."""
    (tmp_path / "myths").mkdir()
    (tmp_path / "myths" / "collection.json").write_text(json.dumps({"description": "Myths."}))
    (tmp_path / "myths" / "thor.md").write_text("# Thor\nSource: https://x/Thor\n\n## Introduction\nThor is a god.\n")
    (tmp_path / "drafts").mkdir()
    chunks, descriptions = load_collections(tmp_path)
    assert descriptions == {"myths": "Myths."}
    assert [(c.collection, c.title) for c in chunks] == [("myths", "Thor")]


def test_load_collections_requires_one(tmp_path):
    """No collection at all is a startup error, not an empty index."""
    with pytest.raises(ValueError, match="No document collection"):
        load_collections(tmp_path)
