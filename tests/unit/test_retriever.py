from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from src.config import Settings
from src.exceptions import RetrievalError
from src.retriever import HybridRetriever, _tokenize


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test-api-key")


@pytest.fixture
def biology_chunks() -> list[Document]:
    return [
        Document(
            page_content="The mitochondria is the powerhouse of the cell.",
            metadata={"source_file": "bio.txt", "chunk_index": 0, "total_chunks": 3},
        ),
        Document(
            page_content="Photosynthesis converts sunlight into energy.",
            metadata={"source_file": "bio.txt", "chunk_index": 1, "total_chunks": 3},
        ),
        Document(
            page_content="Mitochondria produce ATP through cellular respiration.",
            metadata={"source_file": "bio.txt", "chunk_index": 2, "total_chunks": 3},
        ),
    ]


class TestTokenize:
    def test_lowercases_and_splits(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert _tokenize("Python!") == ["python"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert _tokenize("") == []

    def test_same_input_produces_same_output(self) -> None:
        text = "Mitochondria produce ATP."
        assert _tokenize(text) == _tokenize(text)

    def test_period_attached_to_word_is_stripped(self) -> None:
        tokens = _tokenize("end.")
        assert tokens == ["end"]


class TestDenseRetrieval:
    def test_returns_doc_float_tuples(
        self,
        mock_vectorstore: MagicMock,
        sample_chunks: list[Document],
        settings: Settings,
    ) -> None:
        retriever = HybridRetriever(mock_vectorstore, sample_chunks, settings)
        results = retriever.retrieve_dense("test query", k=3, fetch_k=12)
        assert len(results) == 3
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_passes_mmr_lambda_from_settings(
        self,
        mock_vectorstore: MagicMock,
        sample_chunks: list[Document],
        settings: Settings,
    ) -> None:
        retriever = HybridRetriever(mock_vectorstore, sample_chunks, settings)
        retriever.retrieve_dense("query", k=2, fetch_k=8)
        _, kwargs = (
            mock_vectorstore.max_marginal_relevance_search_with_score_by_vector.call_args
        )
        assert kwargs["lambda_mult"] == settings.mmr_lambda

    def test_raises_retrieval_error_on_failure(
        self,
        mock_vectorstore: MagicMock,
        sample_chunks: list[Document],
        settings: Settings,
    ) -> None:
        mock_vectorstore.max_marginal_relevance_search_with_score_by_vector.side_effect = RuntimeError(
            "faiss internal error"
        )
        retriever = HybridRetriever(mock_vectorstore, sample_chunks, settings)
        with pytest.raises(RetrievalError, match="Dense retrieval failed"):
            retriever.retrieve_dense("query", k=3, fetch_k=12)


class TestBM25Retrieval:
    def test_keyword_match_ranks_in_top_results(
        self,
        mock_vectorstore: MagicMock,
        biology_chunks: list[Document],
        settings: Settings,
    ) -> None:
        retriever = HybridRetriever(mock_vectorstore, biology_chunks, settings)
        results = retriever.retrieve_bm25("mitochondria", k=2)
        top_contents = [doc.page_content for doc, _ in results]
        assert all("mitochondria" in content.lower() for content in top_contents)

    def test_returns_at_most_k_results(
        self,
        mock_vectorstore: MagicMock,
        biology_chunks: list[Document],
        settings: Settings,
    ) -> None:
        retriever = HybridRetriever(mock_vectorstore, biology_chunks, settings)
        results = retriever.retrieve_bm25("energy", k=2)
        assert len(results) <= 2

    def test_scores_are_floats(
        self,
        mock_vectorstore: MagicMock,
        biology_chunks: list[Document],
        settings: Settings,
    ) -> None:
        retriever = HybridRetriever(mock_vectorstore, biology_chunks, settings)
        for _, score in retriever.retrieve_bm25("cell", k=3):
            assert isinstance(score, float)

    def test_tokenization_is_consistent_at_index_and_query_time(
        self,
        mock_vectorstore: MagicMock,
        settings: Settings,
    ) -> None:
        chunks = [
            Document(page_content="Python programming language.", metadata={}),
            Document(page_content="Java programming language.", metadata={}),
        ]
        retriever = HybridRetriever(mock_vectorstore, chunks, settings)
        results = retriever.retrieve_bm25("Python!", k=2)
        assert "Python" in results[0][0].page_content


class TestHybridRetrieval:
    def test_deduplicates_doc_appearing_in_both_lists(
        self,
        settings: Settings,
    ) -> None:
        shared = Document(page_content="shared document content", metadata={})
        other = Document(page_content="unique dense-only content", metadata={})

        vs = MagicMock()
        vs.embedding_function.return_value = [0.1] * 10
        vs.max_marginal_relevance_search_with_score_by_vector.return_value = [
            (shared, 0.9),
            (other, 0.7),
        ]
        retriever = HybridRetriever(vs, [shared, other], settings)
        results = retriever.retrieve_hybrid("shared", k=5)

        assert [doc.page_content for doc in results].count(
            "shared document content"
        ) == 1

    def test_doc_in_both_lists_ranks_above_doc_in_one_list(
        self,
        mock_vectorstore: MagicMock,
        sample_chunks: list[Document],
        settings: Settings,
        mocker: MockerFixture,
    ) -> None:
        doc_both = Document(page_content="appears in both retrieval lists", metadata={})
        doc_dense = Document(page_content="appears only in dense results", metadata={})
        doc_bm25 = Document(page_content="appears only in bm25 results", metadata={})

        mocker.patch.object(
            HybridRetriever,
            "retrieve_dense",
            return_value=[(doc_both, 0.9), (doc_dense, 0.5)],
        )
        mocker.patch.object(
            HybridRetriever,
            "retrieve_bm25",
            return_value=[(doc_both, 10.0), (doc_bm25, 5.0)],
        )

        retriever = HybridRetriever(mock_vectorstore, sample_chunks, settings)
        results = retriever.retrieve_hybrid("query", k=3)

        # doc_both scores 1/60 + 1/60 ≈ 0.033; single-list docs score 1/61 ≈ 0.016
        assert results[0].page_content == "appears in both retrieval lists"

    def test_returns_at_most_k_documents(
        self,
        mock_vectorstore: MagicMock,
        sample_chunks: list[Document],
        settings: Settings,
    ) -> None:
        retriever = HybridRetriever(mock_vectorstore, sample_chunks, settings)
        results = retriever.retrieve_hybrid("query", k=2)
        assert len(results) <= 2


class TestHyDE:
    def test_calls_llm_exactly_once(
        self,
        mock_vectorstore: MagicMock,
        sample_chunks: list[Document],
        settings: Settings,
        mock_llm: MagicMock,
    ) -> None:
        retriever = HybridRetriever(mock_vectorstore, sample_chunks, settings)
        retriever.expand_query_hyde("What is machine learning?", mock_llm)
        mock_llm.invoke.assert_called_once()

    def test_returns_llm_response_content(
        self,
        mock_vectorstore: MagicMock,
        sample_chunks: list[Document],
        settings: Settings,
        mock_llm: MagicMock,
    ) -> None:
        retriever = HybridRetriever(mock_vectorstore, sample_chunks, settings)
        result = retriever.expand_query_hyde("What is AI?", mock_llm)
        assert result == mock_llm.invoke.return_value.content

    def test_falls_back_to_original_query_on_llm_failure(
        self,
        mock_vectorstore: MagicMock,
        sample_chunks: list[Document],
        settings: Settings,
        mock_llm: MagicMock,
    ) -> None:
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        retriever = HybridRetriever(mock_vectorstore, sample_chunks, settings)
        result = retriever.expand_query_hyde("original query text", mock_llm)
        assert result == "original query text"
