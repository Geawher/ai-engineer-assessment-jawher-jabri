from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class SourceType(StrEnum):
    DOCUMENT = "document"
    SUPERHERO_API = "superhero_api"
    LLM = "llm"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question


class Source(BaseModel):
    id: str
    type: SourceType
    title: str
    url: str | None = None
    collection: str | None = None  # set for documents: which collection the passage came from


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    tools_used: list[str]
    warnings: list[str] = Field(default_factory=list)
