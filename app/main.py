from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status

from app.agent import Agent
from app.config import get_settings
from app.llm import LLMClient, LLMError, LLMUnavailableError
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.loader import load_collections
from app.retrieval.titles import TitleIndex
from app.schemas import AskRequest, AskResponse
from app.sources.superhero import SuperheroClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the index and clients once at startup and close them on shutdown."""
    settings = get_settings()
    chunks, collections = load_collections(settings.documents_dir)  # raises if there is none: fail at startup
    logger.info("Indexed %d chunks from collections %s", len(chunks), ", ".join(collections))
    retriever = BM25Retriever(chunks)

    llm = LLMClient.from_settings(settings)
    async with httpx.AsyncClient(timeout=settings.superhero_timeout_seconds, follow_redirects=True) as http:
        superheroes = SuperheroClient(
            http,
            base_url=settings.superhero_api_base_url,
            token=settings.superhero_api_token.get_secret_value(),
            cache_ttl_seconds=settings.superhero_cache_ttl_seconds,
        )
        app.state.agent = Agent(llm, retriever, superheroes,
                                collections=collections, titles=TitleIndex(chunks),
                                top_k=settings.retrieval_top_k,
                                max_tool_rounds=settings.max_tool_rounds,
                                prefetch=settings.prefetch_sources)
        yield
    await llm.close()


app = FastAPI(title="Mythology & Superhero Bot", lifespan=lifespan)


def get_agent(request: Request) -> Agent:
    """FastAPI dependency returning the agent built at startup (overridden in tests)."""
    return request.app.state.agent


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, agent: Agent = Depends(get_agent)) -> AskResponse:
    """Answer a question about mythology and/or superheroes, with sources."""
    try:
        return await agent.ask(body.question)
    except LLMUnavailableError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="The language model is temporarily unavailable. Try again shortly.")
    except LLMError:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="The language model provider returned an error.")
