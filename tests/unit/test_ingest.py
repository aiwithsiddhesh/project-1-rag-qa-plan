from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.exceptions import ChunkingError, DocumentLoadError
from src.ingest import chunk_documents, load_documents


class TestLoadDocuments:
    def test_txt_loads_with_correct_metadata(self, sample_txt_path: Path) -> None:
        docs = load_documents(sample_txt_path.parent)
        assert len(docs) >= 1
        doc = docs[0]
        assert doc.metadata["source_file"] == "sample.txt"
        assert doc.metadata["file_type"] == "txt"
        assert doc.metadata["file_size_bytes"] > 0
        assert "ingested_at" in doc.metadata

    def test_pdf_loads_successfully(self, sample_pdf_path: Path) -> None:
        docs = load_documents(sample_pdf_path.parent)
        assert len(docs) >= 1
        assert any(d.metadata["source_file"] == "sample.pdf" for d in docs)

    def test_docx_loads_successfully(self, sample_docx_path: Path) -> None:
        docs = load_documents(sample_docx_path.parent)
        assert len(docs) >= 1
        assert any(d.metadata["source_file"] == "sample.docx" for d in docs)

    def test_unsupported_format_skipped_alongside_valid_file(
        self, sample_txt_path: Path
    ) -> None:
        (sample_txt_path.parent / "ignored.xyz").write_text("ignored content")
        docs = load_documents(sample_txt_path.parent)
        assert all(d.metadata["source_file"] != "ignored.xyz" for d in docs)
        assert len(docs) >= 1

    def test_empty_directory_raises(self, empty_docs_dir: Path) -> None:
        with pytest.raises(DocumentLoadError, match="No documents found"):
            load_documents(empty_docs_dir)

    def test_dir_with_only_unsupported_files_raises(self, tmp_path: Path) -> None:
        (tmp_path / "file.xyz").write_text("ignored")
        (tmp_path / "file.log").write_text("ignored")
        with pytest.raises(DocumentLoadError, match="No documents found"):
            load_documents(tmp_path)

    def test_corrupt_pdf_raises_document_load_error(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"NOT A VALID PDF CONTENT AT ALL GARBAGE DATA")
        with pytest.raises(DocumentLoadError, match="corrupt.pdf"):
            load_documents(tmp_path)

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        with pytest.raises(DocumentLoadError):
            load_documents(missing)

    def test_metadata_fields_all_present_on_every_document(
        self, sample_txt_path: Path
    ) -> None:
        docs = load_documents(sample_txt_path.parent)
        for doc in docs:
            assert "source_file" in doc.metadata
            assert "file_type" in doc.metadata
            assert "file_size_bytes" in doc.metadata
            assert "ingested_at" in doc.metadata

    def test_file_size_bytes_matches_actual_file_size(
        self, sample_txt_path: Path
    ) -> None:
        docs = load_documents(sample_txt_path.parent)
        expected_size = sample_txt_path.stat().st_size
        assert docs[0].metadata["file_size_bytes"] == expected_size


class TestChunkDocuments:
    def test_basic_chunking_produces_multiple_chunks(self) -> None:
        doc = Document(
            page_content="word " * 200,
            metadata={"source_file": "test.txt"},
        )
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 2

    def test_chunk_index_sequential_from_zero(self) -> None:
        doc = Document(
            page_content="word " * 200,
            metadata={"source_file": "test.txt"},
        )
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_total_chunks_matches_actual_count(self) -> None:
        doc = Document(
            page_content="word " * 200,
            metadata={"source_file": "test.txt"},
        )
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
        expected_total = len(chunks)
        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == expected_total

    def test_chunk_index_resets_per_source_document(self) -> None:
        docs = [
            Document(page_content="word " * 100, metadata={"source_file": "a.txt"}),
            Document(page_content="word " * 100, metadata={"source_file": "b.txt"}),
        ]
        chunks = chunk_documents(docs, chunk_size=80, chunk_overlap=10)
        a_chunks = [c for c in chunks if c.metadata["source_file"] == "a.txt"]
        b_chunks = [c for c in chunks if c.metadata["source_file"] == "b.txt"]
        assert a_chunks[0].metadata["chunk_index"] == 0
        assert b_chunks[0].metadata["chunk_index"] == 0
        assert a_chunks[-1].metadata["total_chunks"] == len(a_chunks)
        assert b_chunks[-1].metadata["total_chunks"] == len(b_chunks)

    def test_overlap_produces_extra_chars_vs_original(self) -> None:
        content = ("word " * 60).strip()
        doc = Document(page_content=content, metadata={"source_file": "test.txt"})
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=30)
        assert len(chunks) >= 2
        total_chars = sum(len(c.page_content) for c in chunks)
        assert total_chars > len(content)

    def test_empty_input_raises_chunking_error(self) -> None:
        with pytest.raises(ChunkingError, match="No documents"):
            chunk_documents([])

    def test_source_metadata_preserved_through_chunking(
        self, sample_txt_path: Path
    ) -> None:
        docs = load_documents(sample_txt_path.parent)
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "sample.txt"
            assert "file_type" in chunk.metadata
            assert "ingested_at" in chunk.metadata
