from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from app.sources.superhero import SuperheroAPIError, SuperheroClient

BASE_URL = "https://superheroapi.test/api"
SEARCH_URL = f"{BASE_URL}/token123/search/thor"

THOR_PAYLOAD = {
    "response": "success",
    "results": [
        {"id": "660", "name": "Thor Girl", "biography": {"publisher": "Marvel Comics"}},
        {"id": "659", "name": "Thor", "powerstats": {"strength": "100"},
         "biography": {"publisher": "Marvel Comics", "place-of-birth": "-"}},
    ],
}


@pytest.fixture
async def client() -> AsyncIterator[SuperheroClient]:
    """A SuperheroClient pointed at a fake base URL (requests are intercepted by respx)."""
    async with httpx.AsyncClient() as http:
        yield SuperheroClient(http, base_url=BASE_URL, token="token123", cache_ttl_seconds=60)


@respx.mock
async def test_search_keeps_only_exact_matches_and_trims_fields(client):
    """An exact name match wins outright, and placeholder values like "-" are dropped."""
    respx.get(SEARCH_URL).respond(json=THOR_PAYLOAD)
    heroes = await client.search("Thor")

    assert [hero.name for hero in heroes] == ["Thor"]
    assert heroes[0].profile == {"name": "Thor", "publisher": "Marvel Comics", "powerstats": {"strength": "100"}}


@respx.mock
async def test_not_found_returns_empty_list(client):
    """The API's "not found" error payload means no results, not a failure."""
    respx.get(SEARCH_URL).respond(json={"response": "error", "error": "character with given name not found"})
    assert await client.search("thor") == []


@respx.mock
async def test_api_error_payload_raises(client):
    """Other API-level errors (e.g. a bad token) raise SuperheroAPIError."""
    respx.get(SEARCH_URL).respond(json={"response": "error", "error": "access denied"})
    with pytest.raises(SuperheroAPIError):
        await client.search("thor")


@respx.mock
async def test_timeout_raises(client):
    """Transport failures raise SuperheroAPIError."""
    respx.get(SEARCH_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(SuperheroAPIError):
        await client.search("thor")


@respx.mock
async def test_results_are_cached(client):
    """Repeat lookups, regardless of case and whitespace, hit the API once."""
    route = respx.get(SEARCH_URL).respond(json=THOR_PAYLOAD)
    await client.search("Thor")
    await client.search("  thor ")
    assert route.call_count == 1


@respx.mock
async def test_transient_failure_is_retried_once(client):
    """A connect timeout followed by a success returns the result."""
    route = respx.get(SEARCH_URL).mock(side_effect=[httpx.ConnectTimeout("slow"), httpx.Response(200, json=THOR_PAYLOAD)])
    heroes = await client.search("thor")
    assert heroes[0].name == "Thor"
    assert route.call_count == 2


@respx.mock
async def test_client_errors_are_not_retried(client):
    """A 4xx fails immediately; retrying won't help."""
    route = respx.get(SEARCH_URL).respond(status_code=403)
    with pytest.raises(SuperheroAPIError):
        await client.search("thor")
    assert route.call_count == 1


@respx.mock
async def test_substring_only_matches_are_dropped(client):
    """Without an exact match, whole-word matches stay and substring-only hits ("Lex Luthor") go."""
    payload = {"response": "success", "results": [{"id": "405", "name": "Lex Luthor"}, {"id": "660", "name": "Thor Girl"}]}
    respx.get(SEARCH_URL).respond(json=payload)
    assert [hero.name for hero in await client.search("thor")] == ["Thor Girl"]


@respx.mock
async def test_several_exact_matches_all_stay(client):
    """Two characters named "Batman" are both returned so the model can tell them apart."""
    payload = {"response": "success", "results": [
        {"id": "69", "name": "Batman"}, {"id": "70", "name": "Batman"}, {"id": "71", "name": "Batman II"},
    ]}
    respx.get(f"{BASE_URL}/token123/search/batman").respond(json=payload)
    assert [hero.id for hero in await client.search("Batman")] == ["69", "70"]
