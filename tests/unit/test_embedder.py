from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

import src.embedder as embedder_module
from src.embedder import build_vectorstore, get_embedding_model, load_vectorstore
from src.exceptions import (
    EmbeddingError,
    VectorStoreCorruptError,
    VectorStoreError,
    VectorStoreNotFoundError,
)


@pytest.fixture(autouse=True)
def clear_embedding_cache() -> None:
    embedder_module._EMBEDDING_CACHE.clear()
    yield
    embedder_module._EMBEDDING_CACHE.clear()


class TestGetEmbeddingModel:
    def test_returns_same_instance_on_repeated_calls(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch("src.embedder.HuggingFaceEmbeddings", return_value=MagicMock())
        model_a = get_embedding_model("test-model")
        model_b = get_embedding_model("test-model")
        assert model_a is model_b

    def test_constructor_called_once_despite_two_calls(
        self, mocker: MockerFixture
    ) -> None:
        mock_cls = mocker.patch(
            "src.embedder.HuggingFaceEmbeddings", return_value=MagicMock()
        )
        get_embedding_model("cached-model")
        get_embedding_model("cached-model")
        mock_cls.assert_called_once()

    def test_different_model_names_return_different_instances(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "src.embedder.HuggingFaceEmbeddings",
            side_effect=[MagicMock(), MagicMock()],
        )
        model_a = get_embedding_model("model-a")
        model_b = get_embedding_model("model-b")
        assert model_a is not model_b

    def test_load_failure_raises_embedding_error(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.embedder.HuggingFaceEmbeddings",
            side_effect=RuntimeError("load failed"),
        )
        with pytest.raises(EmbeddingError, match="load failed"):
            get_embedding_model("bad-model")

    def test_normalize_embeddings_passed_to_constructor(
        self, mocker: MockerFixture
    ) -> None:
        mock_cls = mocker.patch(
            "src.embedder.HuggingFaceEmbeddings", return_value=MagicMock()
        )
        get_embedding_model("any-model")
        _, kwargs = mock_cls.call_args
        assert kwargs["encode_kwargs"]["normalize_embeddings"] is True


class TestBuildVectorstore:
    def test_build_saves_to_disk(
        self,
        tmp_path: Path,
        mock_embedding_model: MagicMock,
        sample_chunks: list[Document],
        mocker: MockerFixture,
    ) -> None:
        mock_faiss = MagicMock()
        mocker.patch("src.embedder.FAISS.from_documents", return_value=mock_faiss)

        save_path = tmp_path / "vectorstore"
        result = build_vectorstore(sample_chunks, mock_embedding_model, save_path)

        assert save_path.exists()
        mock_faiss.save_local.assert_called_once_with(str(save_path))
        assert result is mock_faiss

    def test_empty_chunks_raises_vector_store_error(
        self, tmp_path: Path, mock_embedding_model: MagicMock
    ) -> None:
        with pytest.raises(VectorStoreError, match="empty"):
            build_vectorstore([], mock_embedding_model, tmp_path / "vs")

    def test_creates_nested_parent_directories(
        self,
        tmp_path: Path,
        mock_embedding_model: MagicMock,
        sample_chunks: list[Document],
        mocker: MockerFixture,
    ) -> None:
        mocker.patch("src.embedder.FAISS.from_documents", return_value=MagicMock())
        save_path = tmp_path / "deep" / "nested" / "vectorstore"
        build_vectorstore(sample_chunks, mock_embedding_model, save_path)
        assert save_path.exists()

    def test_faiss_error_raises_vector_store_error(
        self,
        tmp_path: Path,
        mock_embedding_model: MagicMock,
        sample_chunks: list[Document],
        mocker: MockerFixture,
    ) -> None:
        mocker.patch(
            "src.embedder.FAISS.from_documents",
            side_effect=RuntimeError("FAISS internal error"),
        )
        with pytest.raises(VectorStoreError, match="Failed to build"):
            build_vectorstore(sample_chunks, mock_embedding_model, tmp_path / "vs")


class TestLoadVectorstore:
    def test_missing_path_raises_not_found_with_path_in_message(
        self, tmp_path: Path, mock_embedding_model: MagicMock
    ) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(VectorStoreNotFoundError, match=str(missing.name)):
            load_vectorstore(missing, mock_embedding_model)

    def test_missing_path_message_includes_make_ingest(
        self, tmp_path: Path, mock_embedding_model: MagicMock
    ) -> None:
        missing = tmp_path / "no_store"
        with pytest.raises(VectorStoreNotFoundError, match="make ingest"):
            load_vectorstore(missing, mock_embedding_model)

    def test_corrupt_index_raises_corrupt_error(
        self, tmp_path: Path, mock_embedding_model: MagicMock
    ) -> None:
        save_path = tmp_path / "corrupt_store"
        save_path.mkdir()
        (save_path / "index.pkl").write_bytes(b"not valid pickle data at all")
        (save_path / "index.faiss").write_bytes(b"not valid faiss data at all")

        with pytest.raises(VectorStoreCorruptError):
            load_vectorstore(save_path, mock_embedding_model)

    def test_load_success_returns_faiss_instance(
        self,
        tmp_path: Path,
        mock_embedding_model: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        save_path = tmp_path / "vectorstore"
        save_path.mkdir()
        mock_faiss = MagicMock()
        mocker.patch("src.embedder.FAISS.load_local", return_value=mock_faiss)

        result = load_vectorstore(save_path, mock_embedding_model)

        assert result is mock_faiss
