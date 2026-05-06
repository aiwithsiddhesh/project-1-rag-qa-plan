from pathlib import Path


class RAGException(Exception):
    """Base exception for recoverable RAG system failures."""

    def __init__(
        self,
        message: str,
        source_path: Path | None = None,
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.source_path = source_path
        self.original_error = original_error
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        details = self.message
        if self.source_path is not None:
            details = f"{details} [source_path={self.source_path}]"
        if self.original_error is not None:
            details = f"{details} [original_error={self.original_error}]"
        return details


class DocumentLoadError(RAGException):
    """Raised when a document cannot be loaded or has an unsupported format."""


class ChunkingError(RAGException):
    """Raised when document chunking fails."""


class VectorStoreError(RAGException):
    """Raised when vector store build or load operations fail."""


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when a requested vector store path does not exist."""


class VectorStoreCorruptError(VectorStoreError):
    """Raised when a vector store exists but cannot be deserialized."""


class EmbeddingError(RAGException):
    """Raised when embedding model operations fail."""


class RetrievalError(RAGException):
    """Raised when retrieval fails."""


class GenerationError(RAGException):
    """Raised when answer generation fails."""


class GenerationTimeoutError(GenerationError):
    """Raised after generation retries are exhausted."""
