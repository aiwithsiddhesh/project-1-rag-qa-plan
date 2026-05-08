from __future__ import annotations

import os

from loguru import logger
from prometheus_client import Counter

from src.config import Settings


rag_chunks_retrieved_total = Counter(
    "rag_chunks_retrieved_total",
    "Total number of chunks retrieved by RAG queries.",
)
rag_empty_context_total = Counter(
    "rag_empty_context_total",
    "Total number of RAG queries that returned no usable context.",
)


def configure_langsmith(settings: Settings) -> None:
    """Enable LangSmith tracing only when requested and a key is available."""
    api_key = (
        settings.langsmith_api_key.get_secret_value()
        if settings.langsmith_api_key is not None
        else ""
    )

    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled")
        return

    if not api_key:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning("LangSmith tracing requested but LANGSMITH_API_KEY is not set")
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    logger.info(
        "LangSmith tracing enabled for project {project}",
        project=settings.langsmith_project,
    )
