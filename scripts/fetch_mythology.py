"""Download the mythology dataset from Wikipedia into data/mythology/*.md.

The output is committed to the repo so the app runs without hitting Wikipedia.
Re-run only to refresh or extend the article list:

    uv run python scripts/fetch_mythology.py
"""

from __future__ import annotations

import re
import time
import unicodedata
from http import HTTPStatus
from pathlib import Path

import httpx

API_URL = "https://en.wikipedia.org/w/api.php"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mythology"
USER_AGENT = "superhero-mythology-bot/0.1 (educational project; https://github.com/Geawher)"
MAX_ATTEMPTS = 5
TOP_LEVEL_HEADING = 2  # "== Heading ==" in Wikipedia plaintext
REQUEST_DELAY_SECONDS = 1

NORSE = [
    "Norse mythology", "Odin", "Thor", "Loki", "Frigg", "Freyja", "Freyr", "Baldr",
    "Heimdall", "Týr", "Hel (mythological being)", "Sif", "Fenrir", "Jörmungandr",
    "Valkyrie", "Valhalla", "Asgard", "Yggdrasil", "Ragnarök", "Mjölnir", "Bifröst",
    "Jötunn", "Surtr", "Ymir",
]
GREEK = [
    "Greek mythology", "Twelve Olympians", "Zeus", "Hera", "Poseidon", "Hades",
    "Athena", "Ares", "Apollo", "Artemis", "Hermes", "Aphrodite", "Hephaestus",
    "Heracles", "Hercules", "Achilles", "Perseus", "Medusa", "Prometheus", "Titans",
    "Cronus", "Amazons", "Hippolyta", "Ajax the Great", "Pandora",
]

# Reference-style sections add noise to retrieval without adding facts.
SKIP_SECTIONS = {
    "references", "see also", "notes", "external links", "further reading",
    "bibliography", "sources", "citations", "footnotes", "primary sources",
    "secondary sources", "notes and references", "works cited",
}

HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")


def fetch_extract(client: httpx.Client, title: str) -> tuple[str, str]:
    """Fetch an article's plaintext, retrying on rate limits.

    Returns:
        The canonical title (after redirects) and the plaintext extract.

    Raises:
        httpx.HTTPStatusError: On a non-rate-limit HTTP error, or rate limits beyond MAX_ATTEMPTS.
        ValueError: If the page doesn't exist.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "redirects": 1,
        "titles": title,
    }
    for attempt in range(MAX_ATTEMPTS):
        response = client.get(API_URL, params=params)
        if response.status_code != HTTPStatus.TOO_MANY_REQUESTS or attempt == MAX_ATTEMPTS - 1:
            break
        wait = float(response.headers.get("Retry-After", 2 ** (attempt + 2)))
        print(f"  rate limited, retrying in {wait:.0f}s")
        time.sleep(wait)
    response.raise_for_status()
    page = next(iter(response.json()["query"]["pages"].values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {title}")
    return page["title"], page["extract"]


def to_markdown(title: str, extract: str) -> str:
    """Convert Wikipedia's '== Heading ==' plaintext into markdown, dropping reference sections."""
    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    lines = [f"# {title}", f"Source: {url}", "", "## Introduction"]
    skipping = False
    for line in extract.splitlines():
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            heading = match.group(2)
            if level == TOP_LEVEL_HEADING:
                skipping = heading.lower() in SKIP_SECTIONS
            if not skipping:
                lines += ["", "#" * level + " " + heading]
        elif not skipping and line.strip():
            lines.append(line.strip())
    return "\n".join(lines) + "\n"


def slugify(title: str) -> str:
    """ASCII-only, lowercase filename stem for a title (e.g. "Ragnarök" -> "ragnarok")."""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_title.lower()).strip("_")


def main() -> None:
    """Fetch every article in NORSE + GREEK and write it to OUT_DIR."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        for title in NORSE + GREEK:
            resolved, extract = fetch_extract(client, title)
            path = OUT_DIR / f"{slugify(resolved)}.md"
            path.write_text(to_markdown(resolved, extract), encoding="utf-8")
            print(f"{title} -> {path.name} ({len(extract):,} chars)")
            time.sleep(REQUEST_DELAY_SECONDS)  # stay under Wikipedia's rate limit


if __name__ == "__main__":
    main()
