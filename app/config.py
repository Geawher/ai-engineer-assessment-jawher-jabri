from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


RequiredSecret = Annotated[SecretStr, Field(min_length=1)]


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key: SecretStr
    model: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Any OpenAI-compatible endpoint works (Groq, Gemini, Cerebras, NVIDIA NIM, ...).
    llm_primary_base_url: str
    llm_primary_api_key: RequiredSecret
    llm_primary_model: str

    # Optional: used when the primary is rate-limited or down.
    llm_fallback_base_url: str | None = None
    llm_fallback_api_key: SecretStr | None = None
    llm_fallback_model: str | None = None

    llm_timeout_seconds: float = 30.0
    # Retries on the same provider before failing over. 0 = fail over immediately,
    # which is what we want when the usual failure is a free-tier rate limit.
    llm_max_retries: int = 0
    llm_temperature: float = 0.1
    # Per-minute limits come back with a Retry-After of a few seconds: waiting beats a 503.
    # Longer waits (daily quotas) fail over instead, so a request is never stuck for minutes.
    llm_max_rate_limit_wait_seconds: float = 10.0

    superhero_api_token: RequiredSecret
    # The documented /api/... path now 302-redirects here; calling it directly saves a round trip.
    superhero_api_base_url: str = "https://www.superheroapi.com/api.php"
    superhero_timeout_seconds: float = 10.0
    superhero_cache_ttl_seconds: int = 3600

    # Each sub-folder with a collection.json is a document collection (data/mythology today).
    documents_dir: Path = Path("data")
    # 4, not 3: the right *article* was always in the top 3, but the passage holding the answer sometimes
    # ranked 4th ("Baldr killed": the mistletoe passages were #4 and #5), which cost answers in the benchmark.
    retrieval_top_k: int = 4
    max_tool_rounds: int = 3

    @field_validator("llm_fallback_base_url", "llm_fallback_api_key", "llm_fallback_model", mode="before")
    @classmethod
    def _empty_means_unset(cls, value: str | None) -> str | None:
        """Treat `LLM_FALLBACK_X=` (blank line in .env) as not configured."""
        return value or None

    @model_validator(mode="after")
    def _fallback_all_or_nothing(self) -> Settings:
        """Reject a half-configured fallback provider at startup rather than at first failover."""
        fallback_values = (self.llm_fallback_base_url, self.llm_fallback_api_key, self.llm_fallback_model)
        if any(fallback_values) and not all(fallback_values):
            raise ValueError("LLM_FALLBACK_* must be either all set or all empty")
        return self

    @property
    def providers(self) -> list[ProviderConfig]:
        """LLM providers in the order they should be tried: primary first, then the optional fallback."""
        providers = [ProviderConfig(
            name=f"{urlparse(self.llm_primary_base_url).hostname}/{self.llm_primary_model}",
            base_url=self.llm_primary_base_url,
            api_key=self.llm_primary_api_key,
            model=self.llm_primary_model,
        )]
        if self.llm_fallback_base_url:
            providers.append(ProviderConfig(
                name=f"{urlparse(self.llm_fallback_base_url).hostname}/{self.llm_fallback_model}",
                base_url=self.llm_fallback_base_url,
                api_key=self.llm_fallback_api_key,
                model=self.llm_fallback_model,
            ))
        return providers


class ConfigError(Exception):
    """Settings are missing or invalid; the message names the environment variables to fix."""


@lru_cache
def get_settings() -> Settings:
    """Load settings from the environment and .env once per process.

    Raises:
        ConfigError: With one readable line per problem, e.g. "SUPERHERO_API_TOKEN is empty".
    """
    try:
        return Settings()
    except ValidationError as error:
        problems = "; ".join(_describe(problem) for problem in error.errors())
        raise ConfigError(f"Invalid configuration: {problems} (check your .env, see .env.example)") from None


def _describe(problem: dict) -> str:
    """Turn one pydantic error into "ENV_VAR_NAME is not set" style text."""
    names = [str(part).upper() for part in problem["loc"]]
    if problem["type"] == "missing":
        return f"{names[0]} is not set"
    if problem["type"] == "too_short":
        return f"{names[0]} is empty"
    if not names:  # model-level rule, e.g. the all-or-nothing fallback check
        return problem["msg"].removeprefix("Value error, ")
    return f"{names[0]}: {problem['msg']}"
