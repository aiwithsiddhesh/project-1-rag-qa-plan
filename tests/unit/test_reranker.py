import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from src.reranker import CrossEncoderReranker


@pytest.fixture
def mock_cross_encoder(mocker: MockerFixture) -> None:
    return mocker.patch("src.reranker.CrossEncoder")


class TestCrossEncoderReranker:
    def test_rerank_orders_documents_by_descending_score(
        self, mock_cross_encoder: MockerFixture
    ) -> None:
        mock_cross_encoder.return_value.predict.return_value = [0.2, 0.9, 0.5]
        docs = [
            Document(page_content="low relevance", metadata={}),
            Document(page_content="high relevance", metadata={}),
            Document(page_content="medium relevance", metadata={}),
        ]
        reranker = CrossEncoderReranker("test-model")
        results = reranker.rerank("test query", docs, top_n=3)

        assert results[0].page_content == "high relevance"
        assert results[1].page_content == "medium relevance"
        assert results[2].page_content == "low relevance"

    def test_rerank_respects_top_n(self, mock_cross_encoder: MockerFixture) -> None:
        mock_cross_encoder.return_value.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]
        docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(5)]
        reranker = CrossEncoderReranker("test-model")
        results = reranker.rerank("query", docs, top_n=2)

        assert len(results) == 2
        assert results[0].page_content == "doc 0"

    def test_rerank_fallback_preserves_original_order_on_failure(
        self, mock_cross_encoder: MockerFixture
    ) -> None:
        mock_cross_encoder.return_value.predict.side_effect = RuntimeError(
            "model failure"
        )
        docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(3)]
        reranker = CrossEncoderReranker("test-model")
        results = reranker.rerank("query", docs, top_n=3)

        assert [d.page_content for d in results] == ["doc 0", "doc 1", "doc 2"]

    def test_rerank_fallback_respects_top_n_on_failure(
        self, mock_cross_encoder: MockerFixture
    ) -> None:
        mock_cross_encoder.return_value.predict.side_effect = RuntimeError("failure")
        docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(5)]
        reranker = CrossEncoderReranker("test-model")
        results = reranker.rerank("query", docs, top_n=2)

        assert len(results) == 2
        assert results[0].page_content == "doc 0"

    def test_rerank_empty_documents_returns_empty_list(
        self, mock_cross_encoder: MockerFixture
    ) -> None:
        reranker = CrossEncoderReranker("test-model")
        results = reranker.rerank("query", [], top_n=5)
        assert results == []

    def test_rerank_calls_predict_with_query_doc_pairs(
        self, mock_cross_encoder: MockerFixture
    ) -> None:
        mock_model = mock_cross_encoder.return_value
        mock_model.predict.return_value = [0.8, 0.4]
        docs = [
            Document(page_content="first doc", metadata={}),
            Document(page_content="second doc", metadata={}),
        ]
        reranker = CrossEncoderReranker("test-model")
        reranker.rerank("my query", docs, top_n=2)

        expected_pairs = [("my query", "first doc"), ("my query", "second doc")]
        mock_model.predict.assert_called_once_with(expected_pairs)
