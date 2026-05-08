from __future__ import annotations

import os

from api.observability import configure_langsmith
from src.config import Settings


def _reset_langsmith_env(monkeypatch) -> None:
    for key in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_configure_langsmith_enables_tracing_with_key(monkeypatch) -> None:
    _reset_langsmith_env(monkeypatch)
    settings = Settings(
        openai_api_key="test-api-key",
        langsmith_tracing=True,
        langsmith_api_key="langsmith-key",
        langsmith_project="test-project",
    )

    configure_langsmith(settings)

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "langsmith-key"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"


def test_configure_langsmith_disables_tracing_without_key(monkeypatch) -> None:
    _reset_langsmith_env(monkeypatch)
    settings = Settings(
        openai_api_key="test-api-key",
        langsmith_tracing=True,
        langsmith_api_key=None,
    )

    configure_langsmith(settings)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_configure_langsmith_respects_disabled_tracing(monkeypatch) -> None:
    _reset_langsmith_env(monkeypatch)
    settings = Settings(
        openai_api_key="test-api-key",
        langsmith_tracing=False,
        langsmith_api_key="langsmith-key",
    )

    configure_langsmith(settings)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
