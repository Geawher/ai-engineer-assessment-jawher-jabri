from __future__ import annotations

import re
from pathlib import Path

from app.retrieval.base import Chunk

MAX_CHUNK_CHARS = 1200
INTRO_SECTION = "Introduction"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def load_corpus(data_dir: Path) -> list[Chunk]:
    """Load and chunk every markdown article in data_dir, in a stable (sorted) order."""
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        chunks.extend(parse_article(path.stem, path.read_text(encoding="utf-8")))
    return chunks


def parse_article(doc_id: str, markdown: str) -> list[Chunk]:
    """Split one article into section-scoped chunks.

    Args:
        doc_id: Stable identifier for the article, used as the chunk id prefix.
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
        Chunk(id=f"{doc_id}:{i}", title=title, section=section, url=url + _anchor(section), text=text)
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
