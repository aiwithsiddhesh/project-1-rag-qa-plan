from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from src.exceptions import (
    EmbeddingError,
    VectorStoreCorruptError,
    VectorStoreError,
    VectorStoreNotFoundError,
)


# IndexFlatIP (exact brute-force) — correct for corpora <10K chunks.
# At 50K+ chunks, switch to IndexIVFFlat or HNSW to reduce search latency.
_EMBEDDING_CACHE: dict[str, HuggingFaceEmbeddings] = {}


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model.

    normalize_embeddings=True is required for cosine similarity with IndexFlatIP.
    """
    if model_name not in _EMBEDDING_CACHE:
        logger.info("Loading embedding model: {model}", model=model_name)
        try:
            _EMBEDDING_CACHE[model_name] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model: {model_name}",
                original_error=exc,
            ) from exc
    return _EMBEDDING_CACHE[model_name]


def build_vectorstore(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    save_path: Path,
) -> FAISS:
    """Build a FAISS index from document chunks and persist it to disk."""
    if not chunks:
        raise VectorStoreError("Cannot build vectorstore from empty chunk list")

    save_path.mkdir(parents=True, exist_ok=True)
    logger.info("Building FAISS index from {n} chunks", n=len(chunks))

    try:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local(str(save_path))
        logger.info("Vectorstore saved to {path}", path=save_path)
    except Exception as exc:
        raise VectorStoreError(
            "Failed to build or save vectorstore",
            source_path=save_path,
            original_error=exc,
        ) from exc

    return vectorstore


def load_vectorstore(save_path: Path, embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """Load a previously persisted FAISS vectorstore from disk."""
    if not save_path.exists():
        raise VectorStoreNotFoundError(
            f"Vectorstore not found at {save_path}. Run `make ingest` first.",
            source_path=save_path,
        )

    logger.info("Loading vectorstore from {path}", path=save_path)
    try:
        return FAISS.load_local(
            str(save_path),
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        raise VectorStoreCorruptError(
            f"Vectorstore at {save_path} is corrupt or incompatible.",
            source_path=save_path,
            original_error=exc,
        ) from exc
