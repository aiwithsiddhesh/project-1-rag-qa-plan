from __future__ import annotations

from typing import Any

import openai
from langchain_core.documents import Document
from loguru import logger
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.exceptions import GenerationError, GenerationTimeoutError

_CONTEXT_BUDGET_CHARS = 3000
_PROMPT_TEMPLATE = (
    "You are a document Q&A assistant. Answer the question using ONLY the context "
    "below. If the answer is not in the context, say "
    '"I could not find the answer in the provided documents."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def build_prompt(question: str, context_chunks: list[Document]) -> str:
    """Build a grounded prompt with source-annotated context and a 3000-char budget."""
    parts: list[str] = []
    for chunk in context_chunks:
        source_file = chunk.metadata.get("source_file", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", 0)
        parts.append(
            f"[Source: {source_file}, chunk {chunk_index}]\n{chunk.page_content}"
        )

    raw_context = "\n\n".join(parts)
    # Truncate from the end — reranker ensures most-relevant chunks are first.
    if len(raw_context) > _CONTEXT_BUDGET_CHARS:
        raw_context = raw_context[:_CONTEXT_BUDGET_CHARS]

    return _PROMPT_TEMPLATE.format(context=raw_context, question=question)


def call_llm_with_retry(prompt: str, llm: Any) -> str:
    """Call the LLM with tenacity exponential-backoff retry on RateLimitError.

    Raises GenerationTimeoutError after stop_after_attempt(3) exhausts retries.
    Raises GenerationError for all other LLM failures.
    """
    retrying = Retrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        reraise=True,
    )

    def _invoke() -> str:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    try:
        result: str = retrying(_invoke)
        return result
    except openai.RateLimitError as exc:
        logger.error("LLM rate-limited after 3 attempts")
        raise GenerationTimeoutError(
            "LLM generation failed after 3 attempts due to rate limiting",
            original_error=exc,
        ) from exc
    except Exception as exc:
        raise GenerationError("LLM call failed", original_error=exc) from exc


def extract_citations(answer: str, source_documents: list[Document]) -> list[str]:
    """Return unique source_file names that appear as a substring in the answer."""
    seen: set[str] = set()
    citations: list[str] = []
    for doc in source_documents:
        source_file = doc.metadata.get("source_file", "")
        if source_file and source_file not in seen and source_file in answer:
            seen.add(source_file)
            citations.append(source_file)
    return citations
