from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.exceptions import ChunkingError, DocumentLoadError


_LOADER_REGISTRY: dict[str, type] = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def load_documents(docs_path: Path) -> list[Document]:
    """Load supported documents from a directory with enriched file metadata."""
    if not docs_path.exists():
        raise DocumentLoadError(
            f"Directory not found: {docs_path}", source_path=docs_path
        )
    if not docs_path.is_dir():
        raise DocumentLoadError(
            f"Path is not a directory: {docs_path}", source_path=docs_path
        )

    documents: list[Document] = []

    for file_path in sorted(docs_path.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        loader_cls = _LOADER_REGISTRY.get(suffix)

        if loader_cls is None:
            logger.warning("Skipping unsupported file: {name}", name=file_path.name)
            continue

        logger.info("Loading: {name}", name=file_path.name)
        try:
            loader = loader_cls(str(file_path))
            docs = loader.load()
        except Exception as exc:
            raise DocumentLoadError(
                f"Failed to load {file_path.name}",
                source_path=file_path,
                original_error=exc,
            ) from exc

        ingested_at = datetime.now(timezone.utc).isoformat()
        file_size = file_path.stat().st_size
        for doc in docs:
            doc.metadata.update(
                {
                    "source_file": file_path.name,
                    "file_type": suffix.lstrip("."),
                    "file_size_bytes": file_size,
                    "ingested_at": ingested_at,
                }
            )
        documents.extend(docs)

    if not documents:
        raise DocumentLoadError("No documents found", source_path=docs_path)

    logger.info(
        "Loaded {count} page(s) from {path}",
        count=len(documents),
        path=docs_path,
    )
    return documents


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split documents into overlapping chunks with positional metadata."""
    if not docs:
        raise ChunkingError("No documents provided for chunking")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    for doc in docs:
        doc_chunks = splitter.split_documents([doc])
        total = len(doc_chunks)
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total
        all_chunks.extend(doc_chunks)

    logger.info(
        "Produced {chunk_count} chunk(s) from {doc_count} document(s)",
        chunk_count=len(all_chunks),
        doc_count=len(docs),
    )
    return all_chunks
