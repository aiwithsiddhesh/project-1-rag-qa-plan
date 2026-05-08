from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.exceptions import DocumentLoadError


def test_run_ingest_success_prints_summary(
    mocker, tmp_path: Path, sample_chunks
) -> None:
    import scripts.run_ingest as run_ingest

    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    vectorstore_path = tmp_path / "vectorstore"
    mocker.patch.object(
        run_ingest,
        "settings",
        MagicMock(
            chunk_size=500,
            chunk_overlap=50,
            log_level="INFO",
            embedding_model="test-embedding",
            vector_store_path=vectorstore_path,
        ),
    )
    mocker.patch.object(
        run_ingest,
        "_parse_args",
        return_value=MagicMock(
            docs_path=docs_path,
            chunk_size=500,
            chunk_overlap=50,
            force_rebuild=False,
        ),
    )
    mocker.patch.object(run_ingest, "setup_logging")
    mocker.patch.object(run_ingest, "load_documents", return_value=[MagicMock()])
    mocker.patch.object(run_ingest, "chunk_documents", return_value=sample_chunks)
    mocker.patch.object(run_ingest, "get_embedding_model", return_value=MagicMock())
    build = mocker.patch.object(run_ingest, "build_vectorstore")

    assert run_ingest.main() == 0
    build.assert_called_once()


def test_run_ingest_returns_one_on_rag_exception(mocker, tmp_path: Path) -> None:
    import scripts.run_ingest as run_ingest

    mocker.patch.object(
        run_ingest,
        "settings",
        MagicMock(
            chunk_size=500,
            chunk_overlap=50,
            log_level="INFO",
            embedding_model="test-embedding",
            vector_store_path=tmp_path / "vectorstore",
        ),
    )
    mocker.patch.object(
        run_ingest,
        "_parse_args",
        return_value=MagicMock(
            docs_path=tmp_path / "missing",
            chunk_size=500,
            chunk_overlap=50,
            force_rebuild=False,
        ),
    )
    mocker.patch.object(run_ingest, "setup_logging")
    mocker.patch.object(
        run_ingest,
        "load_documents",
        side_effect=DocumentLoadError("No documents found"),
    )

    assert run_ingest.main() == 1


def test_run_ingest_requires_force_rebuild_for_existing_vectorstore(
    mocker, tmp_path: Path
) -> None:
    import scripts.run_ingest as run_ingest

    vectorstore_path = tmp_path / "vectorstore"
    vectorstore_path.mkdir()
    (vectorstore_path / "index.faiss").write_text("existing index", encoding="utf-8")
    mocker.patch.object(
        run_ingest,
        "settings",
        MagicMock(
            chunk_size=500,
            chunk_overlap=50,
            log_level="INFO",
            embedding_model="test-embedding",
            vector_store_path=vectorstore_path,
        ),
    )
    mocker.patch.object(
        run_ingest,
        "_parse_args",
        return_value=MagicMock(
            docs_path=tmp_path / "docs",
            chunk_size=500,
            chunk_overlap=50,
            force_rebuild=False,
        ),
    )
    mocker.patch.object(run_ingest, "setup_logging")
    load_documents = mocker.patch.object(run_ingest, "load_documents")

    assert run_ingest.main() == 1
    load_documents.assert_not_called()


def test_run_ingest_rejects_file_at_vectorstore_path(mocker, tmp_path: Path) -> None:
    import scripts.run_ingest as run_ingest

    vectorstore_path = tmp_path / "vectorstore"
    vectorstore_path.write_text("not a directory", encoding="utf-8")
    mocker.patch.object(
        run_ingest,
        "settings",
        MagicMock(
            chunk_size=500,
            chunk_overlap=50,
            log_level="INFO",
            embedding_model="test-embedding",
            vector_store_path=vectorstore_path,
        ),
    )
    mocker.patch.object(
        run_ingest,
        "_parse_args",
        return_value=MagicMock(
            docs_path=tmp_path / "docs",
            chunk_size=500,
            chunk_overlap=50,
            force_rebuild=True,
        ),
    )
    mocker.patch.object(run_ingest, "setup_logging")
    load_documents = mocker.patch.object(run_ingest, "load_documents")

    assert run_ingest.main() == 1
    load_documents.assert_not_called()


def test_run_ingest_force_rebuild_removes_existing_vectorstore(
    mocker, tmp_path: Path, sample_chunks
) -> None:
    import scripts.run_ingest as run_ingest

    vectorstore_path = tmp_path / "vectorstore"
    vectorstore_path.mkdir()
    (vectorstore_path / ".gitignore").write_text("*", encoding="utf-8")
    index_path = vectorstore_path / "index.faiss"
    index_path.write_text("existing index", encoding="utf-8")
    mocker.patch.object(
        run_ingest,
        "settings",
        MagicMock(
            chunk_size=500,
            chunk_overlap=50,
            log_level="INFO",
            embedding_model="test-embedding",
            vector_store_path=vectorstore_path,
        ),
    )
    mocker.patch.object(
        run_ingest,
        "_parse_args",
        return_value=MagicMock(
            docs_path=tmp_path / "docs",
            chunk_size=500,
            chunk_overlap=50,
            force_rebuild=True,
        ),
    )
    mocker.patch.object(run_ingest, "setup_logging")
    mocker.patch.object(run_ingest, "load_documents", return_value=[MagicMock()])
    mocker.patch.object(run_ingest, "chunk_documents", return_value=sample_chunks)
    mocker.patch.object(run_ingest, "get_embedding_model", return_value=MagicMock())
    mocker.patch.object(run_ingest, "build_vectorstore")

    assert run_ingest.main() == 0
    assert not index_path.exists()
    assert (vectorstore_path / ".gitignore").exists()


def test_run_ingest_ignores_vectorstore_gitignore_on_fresh_build(
    mocker, tmp_path: Path, sample_chunks
) -> None:
    import scripts.run_ingest as run_ingest

    vectorstore_path = tmp_path / "vectorstore"
    vectorstore_path.mkdir()
    (vectorstore_path / ".gitignore").write_text("*", encoding="utf-8")
    mocker.patch.object(
        run_ingest,
        "settings",
        MagicMock(
            chunk_size=500,
            chunk_overlap=50,
            log_level="INFO",
            embedding_model="test-embedding",
            vector_store_path=vectorstore_path,
        ),
    )
    mocker.patch.object(
        run_ingest,
        "_parse_args",
        return_value=MagicMock(
            docs_path=tmp_path / "docs",
            chunk_size=500,
            chunk_overlap=50,
            force_rebuild=False,
        ),
    )
    mocker.patch.object(run_ingest, "setup_logging")
    mocker.patch.object(run_ingest, "load_documents", return_value=[MagicMock()])
    mocker.patch.object(run_ingest, "chunk_documents", return_value=sample_chunks)
    mocker.patch.object(run_ingest, "get_embedding_model", return_value=MagicMock())
    build = mocker.patch.object(run_ingest, "build_vectorstore")

    assert run_ingest.main() == 0
    build.assert_called_once()
