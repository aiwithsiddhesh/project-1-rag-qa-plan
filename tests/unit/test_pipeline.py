from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.config import Settings


@pytest.fixture
def pipeline_docs() -> list[Document]:
    return [
        Document(
            page_content=f"Pipeline test chunk {i}",
            metadata={
                "source_file": f"doc_{i}.txt",
                "chunk_index": i,
                "total_chunks": 3,
            },
        )
        for i in range(3)
    ]


@pytest.fixture
def pipeline_settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        top_k_results=2,
        fetch_k_multiplier=2,
        use_hyde=False,
    )


@pytest.fixture
def hyde_settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        top_k_results=2,
        fetch_k_multiplier=2,
        use_hyde=True,
    )


@pytest.fixture
def mock_vs(pipeline_docs: list[Document]) -> MagicMock:
    store = MagicMock()
    store.index_to_docstore_id = {i: str(i) for i in range(len(pipeline_docs))}
    doc_map = {str(i): doc for i, doc in enumerate(pipeline_docs)}
    store.docstore.search.side_effect = lambda doc_id: doc_map.get(doc_id)
    return store


@pytest.fixture
def mock_reranker(pipeline_docs: list[Document]) -> MagicMock:
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, docs, top_n: docs[:top_n]
    return reranker


@pytest.fixture
def mock_retriever(pipeline_docs: list[Document]) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve_hybrid.return_value = pipeline_docs
    retriever.expand_query_hyde.return_value = "expanded query"
    return retriever


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Test answer.")
    return llm


def make_pipeline(settings, mock_vs, mock_reranker, mock_retriever, mock_llm):
    with (
        patch("src.pipeline.get_embedding_model", return_value=MagicMock()),
        patch("src.pipeline.load_vectorstore", return_value=mock_vs),
        patch("src.pipeline.CrossEncoderReranker", return_value=mock_reranker),
        patch("src.pipeline.HybridRetriever", return_value=mock_retriever),
        patch("src.pipeline.ChatOpenAI", return_value=mock_llm),
    ):
        from src.pipeline import RAGPipeline

        return RAGPipeline(settings)


class TestRAGPipelineInit:
    def test_is_ready_after_successful_init(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        assert pipeline.is_ready() is True

    def test_init_extracts_chunks_from_vectorstore_docstore(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        with (
            patch("src.pipeline.get_embedding_model", return_value=MagicMock()),
            patch("src.pipeline.load_vectorstore", return_value=mock_vs),
            patch("src.pipeline.CrossEncoderReranker", return_value=mock_reranker),
            patch("src.pipeline.HybridRetriever") as MockHybrid,
            patch("src.pipeline.ChatOpenAI", return_value=mock_llm),
        ):
            from src.pipeline import RAGPipeline

            RAGPipeline(pipeline_settings)
            _, call_kwargs = MockHybrid.call_args
            passed_chunks = MockHybrid.call_args[0][1]
            assert len(passed_chunks) == 3

    def test_init_passes_settings_to_hybrid_retriever(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        with (
            patch("src.pipeline.get_embedding_model", return_value=MagicMock()),
            patch("src.pipeline.load_vectorstore", return_value=mock_vs),
            patch("src.pipeline.CrossEncoderReranker", return_value=mock_reranker),
            patch("src.pipeline.HybridRetriever") as MockHybrid,
            patch("src.pipeline.ChatOpenAI", return_value=mock_llm),
        ):
            from src.pipeline import RAGPipeline

            RAGPipeline(pipeline_settings)
            assert MockHybrid.call_args[0][2] is pipeline_settings


class TestRAGPipelineQuery:
    def test_query_returns_expected_keys(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        result = pipeline.query("What is the topic?")
        assert set(result.keys()) == {
            "answer",
            "sources",
            "num_chunks_retrieved",
            "retrieval_scores",
        }

    def test_query_answer_comes_from_llm(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        result = pipeline.query("What is the topic?")
        assert result["answer"] == "Test answer."

    def test_query_num_chunks_retrieved_equals_len_candidates(
        self,
        pipeline_settings,
        mock_vs,
        mock_reranker,
        mock_retriever,
        mock_llm,
        pipeline_docs,
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        result = pipeline.query("What is the topic?")
        # num_chunks_retrieved reflects the retrieval phase (pre-rerank) candidate count
        assert result["num_chunks_retrieved"] == len(pipeline_docs)

    def test_query_calls_reranker_with_candidates(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        pipeline.query("What is the topic?")
        mock_reranker.rerank.assert_called_once()
        _, call_kwargs = mock_reranker.rerank.call_args
        assert (
            mock_reranker.rerank.call_args[1]["top_n"]
            == pipeline_settings.top_k_results
        )

    def test_query_retrieves_top_k_times_fetch_k_multiplier_candidates(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        pipeline.query("What is the topic?")
        call_args = mock_retriever.retrieve_hybrid.call_args
        expected_k = (
            pipeline_settings.top_k_results * pipeline_settings.fetch_k_multiplier
        )
        assert call_args[1]["k"] == expected_k

    def test_query_too_short_raises_value_error(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        with pytest.raises(ValueError, match="at least 3"):
            pipeline.query("Hi")

    def test_query_too_long_raises_value_error(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        with pytest.raises(ValueError, match="at most 2000"):
            pipeline.query("x" * 2001)

    def test_query_with_hyde_calls_expand_query(
        self, hyde_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            hyde_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        pipeline.query("What is the topic?")
        mock_retriever.expand_query_hyde.assert_called_once()

    def test_query_without_hyde_skips_expand_query(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        pipeline.query("What is the topic?")
        mock_retriever.expand_query_hyde.assert_not_called()

    def test_query_retrieval_scores_is_list(
        self, pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
    ):
        pipeline = make_pipeline(
            pipeline_settings, mock_vs, mock_reranker, mock_retriever, mock_llm
        )
        result = pipeline.query("What is the topic?")
        assert isinstance(result["retrieval_scores"], list)
