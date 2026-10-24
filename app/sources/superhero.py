from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

MAX_RESULTS = 3
MAX_ATTEMPTS = 2  # one retry: the API has occasional connect timeouts
RETRY_BACKOFF_SECONDS = 0.5
_ERROR_RESPONSE = "error"
_NOT_FOUND_MARKER = "not found"
# The API uses these placeholders instead of omitting missing fields.
_MISSING_VALUES = (None, "", "-", "null", ["-"])


class SuperheroAPIError(Exception):
    """The Superhero API could not be reached or rejected the request (e.g. bad token)."""


@dataclass(frozen=True)
class Hero:
    id: str
    name: str
    profile: dict[str, Any]


class TTLCache:
    """Minimal in-memory cache whose entries expire after a fixed time-to-live."""

    def __init__(self, ttl_seconds: float):
        """Create an empty cache whose entries live for ttl_seconds."""
        self._ttl = ttl_seconds
        self._items: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        """Return the cached value, or None if it's missing or expired."""
        item = self._items.get(key)
        if item is None or item[0] < time.monotonic():
            self._items.pop(key, None)
            return None
        return item[1]

    def set(self, key: str, value: Any) -> None:
        """Store a value, resetting its expiry."""
        self._items[key] = (time.monotonic() + self._ttl, value)


class SuperheroClient:
    """Async client for the Superhero API's name search, with a TTL cache."""

    def __init__(self, http: httpx.AsyncClient, *, base_url: str, token: str, cache_ttl_seconds: float):
        """Wrap a shared httpx client; the caller owns its lifecycle."""
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._cache = TTLCache(cache_ttl_seconds)

    async def search(self, name: str) -> list[Hero]:
        """Search heroes by name (case-insensitive substring match).

        Returns:
            Up to MAX_RESULTS heroes: only exact name matches if any exist, else whole-word matches;
            [] when nothing matches.

        Raises:
            SuperheroAPIError: On network errors, bad responses, or API-level errors such as an invalid token.
        """
        key = name.strip().lower()
        if (cached := self._cache.get(key)) is not None:
            return cached

        payload = await self._fetch(key)
        if payload.get("response") == _ERROR_RESPONSE:
            # The API signals "no match" as a 200 with an error payload.
            error = payload.get("error", "unknown")
            if _NOT_FOUND_MARKER not in error.lower():
                raise SuperheroAPIError(f"Superhero API error: {error}")
            self._cache.set(key, [])
            return []

        heroes = [Hero(id=str(result.get("id")), name=result.get("name", "?"), profile=_profile(result))
                  for result in _rank(payload.get("results") or [], key)[:MAX_RESULTS]]
        self._cache.set(key, heroes)
        return heroes

    async def _fetch(self, name: str) -> dict[str, Any]:
        """GET the search endpoint and decode its JSON body, retrying once on transient failures.

        Raises:
            SuperheroAPIError: If every attempt fails or the response isn't JSON.
        """
        url = f"{self._base_url}/{self._token}/search/{quote(name, safe='')}"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._http.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as error:
                # Log the status only: the URL (and any redirect Location) contains the API token.
                logger.warning("Superhero API returned HTTP %d (attempt %d)", error.response.status_code, attempt)
                if not error.response.is_server_error or attempt == MAX_ATTEMPTS:
                    raise SuperheroAPIError("Superhero API is unavailable") from error
            except httpx.TransportError as error:
                logger.warning("Superhero API request failed: %s (attempt %d)", type(error).__name__, attempt)
                if attempt == MAX_ATTEMPTS:
                    raise SuperheroAPIError("Superhero API is unavailable") from error
            except ValueError as error:
                raise SuperheroAPIError("Superhero API returned invalid JSON") from error
            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
        raise AssertionError("unreachable: the last attempt always returns or raises")


def _rank(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Pick the results worth sending to the model for a lowercase query.

    Exact name matches win outright ("thor" -> Thor, not Thor Girl), which saves tokens; several exact
    matches all stay (two characters are called "Batman"). Otherwise whole-word matches are kept, and
    substring-only hits ("thor" in "Lex Luthor") are dropped unless nothing else matched.
    """
    exact = [result for result in results if result.get("name", "").lower() == key]
    if exact:
        return exact
    word = re.compile(rf"\b{re.escape(key)}\b")
    return [result for result in results if word.search(result.get("name", "").lower())] or results


def _profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields useful for answering questions; drop the rest to save LLM tokens."""
    biography = raw.get("biography", {})
    appearance = raw.get("appearance", {})
    work = raw.get("work", {})
    connections = raw.get("connections", {})
    profile = {
        "name": raw.get("name"),
        "full_name": biography.get("full-name"),
        "aliases": biography.get("aliases"),
        "publisher": biography.get("publisher"),
        "alignment": biography.get("alignment"),
        "first_appearance": biography.get("first-appearance"),
        "place_of_birth": biography.get("place-of-birth"),
        "powerstats": raw.get("powerstats"),
        "gender": appearance.get("gender"),
        "race": appearance.get("race"),
        "height": appearance.get("height"),
        "weight": appearance.get("weight"),
        "occupation": work.get("occupation"),
        "base": work.get("base"),
        "group_affiliation": connections.get("group-affiliation"),
        "relatives": connections.get("relatives"),
    }
    return {field: value for field, value in profile.items() if value not in _MISSING_VALUES}
