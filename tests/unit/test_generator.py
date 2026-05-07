from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.exceptions import GenerationError, GenerationTimeoutError
from src.generator import build_prompt, call_llm_with_retry, extract_citations


def _make_rate_limit_error() -> openai.RateLimitError:
    """Construct a minimal openai.RateLimitError for test assertions."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


@pytest.fixture
def chunks() -> list[Document]:
    return [
        Document(
            page_content="Artificial intelligence is transforming industries worldwide.",
            metadata={"source_file": "ai_report.pdf", "chunk_index": 0},
        ),
        Document(
            page_content="Machine learning enables pattern recognition at scale.",
            metadata={"source_file": "ml_guide.txt", "chunk_index": 1},
        ),
    ]


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_source_header_for_each_chunk(
        self, chunks: list[Document]
    ) -> None:
        prompt = build_prompt("What is AI?", chunks)
        assert "[Source: ai_report.pdf, chunk 0]" in prompt
        assert "[Source: ml_guide.txt, chunk 1]" in prompt

    def test_includes_chunk_content(self, chunks: list[Document]) -> None:
        prompt = build_prompt("What is AI?", chunks)
        assert "Artificial intelligence is transforming" in prompt
        assert "Machine learning enables" in prompt

    def test_includes_question_in_prompt(self, chunks: list[Document]) -> None:
        prompt = build_prompt("What is AI?", chunks)
        assert "What is AI?" in prompt

    def test_includes_answer_label(self, chunks: list[Document]) -> None:
        prompt = build_prompt("Q?", chunks)
        assert "Answer:" in prompt

    def test_empty_context_yields_valid_prompt(self) -> None:
        prompt = build_prompt("What is AI?", [])
        assert "What is AI?" in prompt
        assert "Answer:" in prompt

    def test_budget_truncation_caps_context_at_3000_chars(self) -> None:
        long_chunk = Document(
            page_content="x" * 4000,
            metadata={"source_file": "long.pdf", "chunk_index": 0},
        )
        prompt = build_prompt("Question?", [long_chunk])
        # Extract just the context section between "Context:\n" and "\n\nQuestion:"
        context_section = prompt.split("Context:\n")[1].split("\n\nQuestion:")[0]
        assert len(context_section) <= 3000

    def test_missing_metadata_falls_back_to_defaults(self) -> None:
        chunk = Document(page_content="Some content.", metadata={})
        prompt = build_prompt("Q?", [chunk])
        assert "[Source: unknown, chunk 0]" in prompt

    def test_most_relevant_chunk_preserved_when_truncating(self) -> None:
        # First chunk should survive truncation; last should be cut when budget is tight.
        first_chunk = Document(
            page_content="FIRST " + "a" * 500,
            metadata={"source_file": "first.pdf", "chunk_index": 0},
        )
        second_chunk = Document(
            page_content="SECOND " + "b" * 3000,
            metadata={"source_file": "second.pdf", "chunk_index": 1},
        )
        prompt = build_prompt("Q?", [first_chunk, second_chunk])
        assert "FIRST" in prompt


# ---------------------------------------------------------------------------
# call_llm_with_retry
# ---------------------------------------------------------------------------


class TestCallLlmWithRetry:
    def test_returns_llm_response_content(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="The answer is 42.")
        result = call_llm_with_retry("What is the answer?", llm)
        assert result == "The answer is 42."

    def test_calls_invoke_exactly_once_on_success(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="OK")
        call_llm_with_retry("Q", llm)
        assert llm.invoke.call_count == 1

    def test_handles_response_without_content_attribute(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = "plain string response"
        result = call_llm_with_retry("Q", llm)
        assert result == "plain string response"

    def test_retries_exactly_three_times_on_rate_limit(self) -> None:
        llm = MagicMock()
        llm.invoke.side_effect = _make_rate_limit_error()
        with patch("time.sleep"):
            with pytest.raises(GenerationTimeoutError):
                call_llm_with_retry("Q", llm)
        assert llm.invoke.call_count == 3

    def test_raises_generation_timeout_error_after_retries(self) -> None:
        llm = MagicMock()
        llm.invoke.side_effect = _make_rate_limit_error()
        with patch("time.sleep"):
            with pytest.raises(GenerationTimeoutError):
                call_llm_with_retry("Q", llm)

    def test_generation_timeout_error_wraps_original_exception(self) -> None:
        llm = MagicMock()
        llm.invoke.side_effect = _make_rate_limit_error()
        with patch("time.sleep"):
            with pytest.raises(GenerationTimeoutError) as exc_info:
                call_llm_with_retry("Q", llm)
        assert exc_info.value.original_error is not None

    def test_raises_generation_error_on_non_rate_limit_exception(self) -> None:
        llm = MagicMock()
        llm.invoke.side_effect = ValueError("unexpected LLM failure")
        with pytest.raises(GenerationError):
            call_llm_with_retry("Q", llm)

    def test_generation_error_wraps_original_non_rate_limit_exception(self) -> None:
        llm = MagicMock()
        original = ValueError("something went wrong")
        llm.invoke.side_effect = original
        with pytest.raises(GenerationError) as exc_info:
            call_llm_with_retry("Q", llm)
        assert exc_info.value.original_error is original

    def test_prompt_is_passed_to_llm_invoke(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="answer")
        call_llm_with_retry("specific prompt text", llm)
        llm.invoke.assert_called_once_with("specific prompt text")


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_extracts_matching_source_file(self, chunks: list[Document]) -> None:
        answer = "According to ai_report.pdf, AI is transforming industries."
        citations = extract_citations(answer, chunks)
        assert "ai_report.pdf" in citations

    def test_excludes_source_not_mentioned_in_answer(
        self, chunks: list[Document]
    ) -> None:
        answer = "AI is transforming industries."
        citations = extract_citations(answer, chunks)
        assert "ml_guide.txt" not in citations

    def test_extracts_multiple_citations(self, chunks: list[Document]) -> None:
        answer = "ai_report.pdf discusses AI while ml_guide.txt covers ML."
        citations = extract_citations(answer, chunks)
        assert "ai_report.pdf" in citations
        assert "ml_guide.txt" in citations

    def test_deduplicates_repeated_source(self) -> None:
        duplicate_chunks = [
            Document(
                page_content="Content A",
                metadata={"source_file": "report.pdf"},
            ),
            Document(
                page_content="Content B",
                metadata={"source_file": "report.pdf"},
            ),
        ]
        answer = "See report.pdf for details."
        citations = extract_citations(answer, duplicate_chunks)
        assert citations.count("report.pdf") == 1

    def test_empty_documents_returns_empty_list(self) -> None:
        assert extract_citations("Some answer.", []) == []

    def test_missing_source_file_metadata_is_skipped(self) -> None:
        chunk = Document(page_content="Content", metadata={})
        citations = extract_citations("Some answer.", [chunk])
        assert citations == []

    def test_empty_source_file_string_is_skipped(self) -> None:
        chunk = Document(page_content="Content", metadata={"source_file": ""})
        citations = extract_citations("Some answer.", [chunk])
        assert citations == []

    def test_preserves_insertion_order(self, chunks: list[Document]) -> None:
        answer = "See ai_report.pdf and ml_guide.txt for details."
        citations = extract_citations(answer, chunks)
        assert citations == ["ai_report.pdf", "ml_guide.txt"]
