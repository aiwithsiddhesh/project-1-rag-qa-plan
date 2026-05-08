from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from src.config import Settings
from src.contracts import NO_CONTEXT_PHRASE


class KeywordEmbeddings(Embeddings):
    """Small deterministic embeddings for local FAISS integration tests."""

    vocabulary = ["machine", "learning", "artificial", "intelligence", "zero"]

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [float(lower.count(token)) for token in self.vocabulary]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


class IdentityReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def rerank(
        self, query: str, documents: list[Document], top_n: int
    ) -> list[Document]:
        return documents[:top_n]


class PromptAwareLLM:
    def invoke(self, prompt: str) -> AIMessage:
        if "Question: How do sourdough starters work?" in prompt:
            return AIMessage(content=NO_CONTEXT_PHRASE)
        return AIMessage(
            content=(
                "Machine learning enables pattern recognition at scale "
                "according to ml_intro.txt."
            )
        )


def _build_test_vectorstore(path: Path, embeddings: KeywordEmbeddings) -> None:
    docs = [
        Document(
            page_content=(
                "Machine learning enables pattern recognition at scale and is "
                "a subset of artificial intelligence."
            ),
            metadata={
                "source_file": "ml_intro.txt",
                "chunk_index": 0,
                "total_chunks": 2,
            },
        ),
        Document(
            page_content=(
                "Zero trust architecture removes implicit trust from network "
                "access decisions."
            ),
            metadata={
                "source_file": "zero_trust.txt",
                "chunk_index": 1,
                "total_chunks": 2,
            },
        ),
    ]
    FAISS.from_documents(docs, embeddings).save_local(str(path))


def _make_pipeline(tmp_path: Path):
    embeddings = KeywordEmbeddings()
    vectorstore_path = tmp_path / "vectorstore"
    _build_test_vectorstore(vectorstore_path, embeddings)
    settings = Settings(
        openai_api_key="test-key",
        vector_store_path=vectorstore_path,
        top_k_results=1,
        fetch_k_multiplier=2,
    )

    with (
        patch("src.pipeline.get_embedding_model", return_value=embeddings),
        patch("src.pipeline.CrossEncoderReranker", IdentityReranker),
        patch("src.pipeline.ChatOpenAI", return_value=PromptAwareLLM()),
    ):
        from src.pipeline import RAGPipeline

        return RAGPipeline(settings)


def test_full_pipeline_returns_required_response_keys(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)

    result = pipeline.query("What is machine learning?")

    assert {
        "answer",
        "sources",
        "num_chunks_retrieved",
        "retrieval_scores",
        "contexts",
    } <= result.keys()
    assert result["num_chunks_retrieved"] > 0


def test_full_pipeline_retrieves_relevant_document(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)

    result = pipeline.query("What is machine learning?")

    assert "Machine learning enables pattern recognition" in result["contexts"][0]
    assert result["sources"] == ["ml_intro.txt"]


def test_full_pipeline_uses_fallback_for_unrelated_query(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)

    result = pipeline.query("How do sourdough starters work?")

    assert result["answer"] == NO_CONTEXT_PHRASE
    assert result["sources"] == []


def test_full_pipeline_accepts_600_character_question(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    question = "What is machine learning? " + "x" * 574

    result = pipeline.query(question)

    assert "Machine learning" in result["answer"]
