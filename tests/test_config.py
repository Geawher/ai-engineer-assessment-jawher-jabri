from __future__ import annotations

import pytest

from app.config import ConfigError, get_settings


@pytest.fixture
def clean_settings(monkeypatch: pytest.MonkeyPatch):
    """Valid LLM settings, no .env file, and an empty settings cache before and after the test."""
    monkeypatch.setenv("LLM_PRIMARY_BASE_URL", "https://llm.test/v1")
    monkeypatch.setenv("LLM_PRIMARY_API_KEY", "key")
    monkeypatch.setenv("LLM_PRIMARY_MODEL", "model")
    monkeypatch.chdir("/")
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def test_empty_setting_is_named(clean_settings):
    """An empty required secret produces a message naming the variable to fix."""
    clean_settings.setenv("SUPERHERO_API_TOKEN", "")
    with pytest.raises(ConfigError, match="SUPERHERO_API_TOKEN is empty"):
        get_settings()


def test_half_configured_fallback_is_explained(clean_settings):
    """A fallback with some but not all LLM_FALLBACK_* values is rejected with the rule spelled out."""
    clean_settings.setenv("SUPERHERO_API_TOKEN", "token")
    clean_settings.setenv("LLM_FALLBACK_MODEL", "other-model")
    with pytest.raises(ConfigError, match="LLM_FALLBACK_\\* must be either all set or all empty"):
        get_settings()
