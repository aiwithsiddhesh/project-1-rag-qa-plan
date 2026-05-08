import importlib
import os
from pathlib import Path

from loguru import logger
import pytest
from pydantic import ValidationError

os.environ.setdefault("OPENAI_API_KEY", "test-api-key")

from src.config import Settings
from src.exceptions import (
    ChunkingError,
    DocumentLoadError,
    EmbeddingError,
    GenerationError,
    GenerationTimeoutError,
    RAGException,
    RetrievalError,
    VectorStoreCorruptError,
    VectorStoreError,
    VectorStoreNotFoundError,
)
from src.utils import timer_context, truncate_text


def test_settings_load_defaults_and_mask_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    settings = Settings(openai_api_key="test-api-key", _env_file=None)

    assert settings.openai_api_key.get_secret_value() == "test-api-key"
    assert "test-api-key" not in repr(settings)
    assert settings.openai_model == "gpt-3.5-turbo"
    assert settings.vector_store_path == Path("./data/vectorstore")
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.bm25_weight == 0.4
    assert settings.dense_weight == 0.6
    assert settings.langsmith_tracing is True
    assert settings.langsmith_api_key is None
    assert settings.langsmith_project == "project-1-rag-qa"
    assert settings.cors_origins == ["*"]


def test_settings_normalizes_empty_langsmith_api_key() -> None:
    settings = Settings(openai_api_key="test-api-key", langsmith_api_key="")

    assert settings.langsmith_api_key is None


@pytest.mark.parametrize("chunk_size", [99, 2001])
def test_settings_reject_invalid_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValidationError, match="chunk_size"):
        Settings(openai_api_key="test-api-key", chunk_size=chunk_size)


def test_settings_reject_chunk_overlap_greater_than_or_equal_to_size() -> None:
    with pytest.raises(
        ValidationError, match="chunk_overlap must be less than chunk_size"
    ):
        Settings(openai_api_key="test-api-key", chunk_size=500, chunk_overlap=500)


def test_settings_reject_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(
        ValidationError, match="bm25_weight and dense_weight must sum to 1.0"
    ):
        Settings(openai_api_key="test-api-key", bm25_weight=0.8, dense_weight=0.6)


def test_settings_requires_non_empty_openai_api_key() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY must not be empty"):
        Settings(openai_api_key="")


def test_module_settings_singleton_loads_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")

    config = importlib.import_module("src.config")
    reloaded_config = importlib.reload(config)

    assert reloaded_config.settings.openai_api_key.get_secret_value() == "env-api-key"


def test_rag_exception_stores_context_and_renders_details() -> None:
    original_error = ValueError("bad pdf")
    source_path = Path("data/sample_docs/bad.pdf")

    error = DocumentLoadError("Unable to load document", source_path, original_error)

    assert error.message == "Unable to load document"
    assert error.source_path == source_path
    assert error.original_error is original_error
    assert "Unable to load document" in str(error)
    assert "data" in str(error)
    assert "bad pdf" in str(error)


def test_exception_hierarchy_matches_phase_plan() -> None:
    assert issubclass(DocumentLoadError, RAGException)
    assert issubclass(ChunkingError, RAGException)
    assert issubclass(VectorStoreError, RAGException)
    assert issubclass(VectorStoreNotFoundError, VectorStoreError)
    assert issubclass(VectorStoreCorruptError, VectorStoreError)
    assert issubclass(EmbeddingError, RAGException)
    assert issubclass(RetrievalError, RAGException)
    assert issubclass(GenerationError, RAGException)
    assert issubclass(GenerationTimeoutError, GenerationError)


def test_truncate_text_normalizes_whitespace_and_truncates() -> None:
    text = "alpha\n\nbeta\tgamma delta"

    assert truncate_text(text, max_chars=16) == "alpha beta ga..."


def test_truncate_text_rejects_negative_length() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        truncate_text("text", max_chars=-1)


def test_settings_normalize_log_level_to_uppercase() -> None:
    settings = Settings(openai_api_key="test-api-key", log_level="debug")
    assert settings.log_level == "DEBUG"


def test_settings_reject_invalid_log_level() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        Settings(openai_api_key="test-api-key", log_level="VERBOSE")


def test_timer_context_logs_elapsed_time() -> None:
    messages: list[str] = []
    handler_id = logger.add(
        lambda msg: messages.append(str(msg)), level="INFO", format="{message}"
    )
    try:
        with timer_context("unit-test-operation"):
            pass
    finally:
        logger.remove(handler_id)
    assert any("unit-test-operation completed in" in m for m in messages)
