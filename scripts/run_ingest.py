from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from time import perf_counter

from src.config import settings
from src.embedder import build_vectorstore, get_embedding_model
from src.exceptions import RAGException
from src.ingest import chunk_documents, load_documents
from src.utils import setup_logging


def _vectorstore_exists(path: Path) -> bool:
    return (path / "index.faiss").exists() or (path / "index.pkl").exists()


def _clear_vectorstore(path: Path) -> None:
    if not path.exists():
        return

    for child in path.iterdir():
        if child.name == ".gitignore":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the FAISS vectorstore from local sample documents."
    )
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=Path("data/sample_docs"),
        help="Directory containing PDF, DOCX, or TXT documents.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Chunk size passed to the text splitter.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Chunk overlap passed to the text splitter.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Remove an existing vectorstore before rebuilding it.",
    )
    return parser.parse_args()


def _print_summary(
    *,
    docs_path: Path,
    vectorstore_path: Path,
    document_count: int,
    chunk_count: int,
    elapsed_seconds: float,
) -> None:
    rows = [
        ("Docs path", str(docs_path)),
        ("Vectorstore path", str(vectorstore_path)),
        ("Loaded documents", str(document_count)),
        ("Generated chunks", str(chunk_count)),
        ("Elapsed seconds", f"{elapsed_seconds:.2f}"),
    ]
    width = max(len(label) for label, _ in rows)
    print("Ingestion summary")
    print("-" * (width + 24))
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


def main() -> int:
    setup_logging(settings.log_level)
    args = _parse_args()
    start = perf_counter()

    try:
        if _vectorstore_exists(settings.vector_store_path):
            if not args.force_rebuild:
                raise RAGException(
                    "Vectorstore already exists. Use --force-rebuild to rebuild it.",
                    source_path=settings.vector_store_path,
                )
            _clear_vectorstore(settings.vector_store_path)

        documents = load_documents(args.docs_path)
        chunks = chunk_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        embedding_model = get_embedding_model(settings.embedding_model)
        build_vectorstore(chunks, embedding_model, settings.vector_store_path)
    except RAGException as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(
        docs_path=args.docs_path,
        vectorstore_path=settings.vector_store_path,
        document_count=len(documents),
        chunk_count=len(chunks),
        elapsed_seconds=perf_counter() - start,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
