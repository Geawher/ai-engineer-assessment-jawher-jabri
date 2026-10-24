from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from app.retrieval.base import Chunk

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 1200
INTRO_SECTION = "Introduction"
COLLECTION_FILE = "collection.json"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def load_collections(documents_dir: Path) -> tuple[list[Chunk], dict[str, str]]:
    """Load every document collection under documents_dir.

    A collection is a sub-folder holding markdown articles and a collection.json with a "description"
    (shown to the model). Adding a dataset means adding a folder: no code or prompt changes.

    Returns:
        All chunks, and each collection's description by name.

    Raises:
        ValueError: If documents_dir holds no collection.
    """
    chunks: list[Chunk] = []
    descriptions: dict[str, str] = {}
    for folder in sorted(path for path in documents_dir.iterdir() if path.is_dir()):
        if not (folder / COLLECTION_FILE).is_file():
            if any(folder.glob("*.md")):
                logger.warning("Skipping %s: it has articles but no %s", folder, COLLECTION_FILE)
            continue
        descriptions[folder.name] = _description(folder / COLLECTION_FILE)
        collection_chunks = load_corpus(folder, collection=folder.name)
        if not collection_chunks:
            raise ValueError(f"Collection {folder} has a {COLLECTION_FILE} but no markdown articles")
        chunks += collection_chunks
    if not descriptions:
        raise ValueError(f"No document collection (a folder with {COLLECTION_FILE}) under {documents_dir}")
    return chunks, descriptions


def _description(path: Path) -> str:
    """The "description" string from a collection.json.

    Raises:
        ValueError: Naming the file, if it isn't valid JSON or has no string "description".
    """
    try:
        description = json.loads(path.read_text(encoding="utf-8"))["description"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f'{path} must be JSON like {{"description": "..."}}: {error}') from error
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f'{path}: "description" must be a non-empty string')
    return description


def load_corpus(folder: Path, collection: str) -> list[Chunk]:
    """Load and chunk every markdown article in one collection folder, in a stable (sorted) order."""
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks.extend(parse_article(path.stem, path.read_text(encoding="utf-8"), collection))
    return chunks


def parse_article(doc_id: str, markdown: str, collection: str) -> list[Chunk]:
    """Split one article into section-scoped chunks.

    Args:
        doc_id: Stable identifier for the article, used as the chunk id prefix.
        collection: Name of the collection the article belongs to.
        markdown: Article in the format written by scripts/fetch_mythology.py:
            "# Title", "Source: <url>", then "## Section" headings with paragraphs.

    Returns:
        Chunks in document order, each at most MAX_CHUNK_CHARS long, with a URL
        anchored to its section.
    """
    lines = markdown.splitlines()
    title = lines[0].removeprefix("# ").strip()
    url = lines[1].removeprefix("Source:").strip()

    # Section names repeat within an article (e.g. "Prose Edda" under several parents),
    # so ids use a running index rather than the section name.
    pieces = [(section, text) for section, paragraphs in _sections(lines[2:]) for text in _pack(paragraphs)]
    return [
        Chunk(id=f"{collection}/{doc_id}:{i}", collection=collection, title=title, section=section, url=url + _anchor(section), text=text)
        for i, (section, text) in enumerate(pieces)
    ]


def _anchor(section: str) -> str:
    """URL fragment linking to a section on its Wikipedia page; the intro has none."""
    return "" if section == INTRO_SECTION else "#" + section.replace(" ", "_")


def _sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group body lines into (heading, paragraphs) pairs; text before the first heading is the intro."""
    sections: list[tuple[str, list[str]]] = [(INTRO_SECTION, [])]
    for line in lines:
        if line.startswith("#"):
            sections.append((line.lstrip("#").strip(), []))
        elif line.strip():
            sections[-1][1].append(line.strip())
    return [(heading, paragraphs) for heading, paragraphs in sections if paragraphs]


def _pack(paragraphs: list[str]) -> list[str]:
    """Greedily pack paragraphs into chunks of at most MAX_CHUNK_CHARS, splitting long ones by sentence."""
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= MAX_CHUNK_CHARS:
            pieces.append(paragraph)
        else:
            pieces.extend(_SENTENCE_RE.split(paragraph))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks
