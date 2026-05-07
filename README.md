# Document Q&A RAG System

This repository is being built phase by phase from `project-1-rag-qa-plan.md`.

Current status: Phase 3 complete — Embedding + Vector Store. Phase 4 (Advanced Retrieval) next.

## Implemented

**Phase 1 — Foundation**
- Pydantic settings with validators in `src/config.py`
- RAG exception hierarchy (`DocumentLoadError`, `VectorStoreNotFoundError`, etc.) in `src/exceptions.py`
- Logging (`setup_logging`), text truncation, and `timer_context` in `src/utils.py`

**Phase 2 — Ingestion Layer**
- `load_documents(docs_path)` in `src/ingest.py` — loads PDF/DOCX/TXT with metadata (`source_file`, `file_type`, `file_size_bytes`, `ingested_at`)
- `chunk_documents(docs, chunk_size, chunk_overlap)` — `RecursiveCharacterTextSplitter` with `chunk_index` / `total_chunks` metadata

**Phase 3 — Embedding + Vector Store**
- `get_embedding_model(model_name)` in `src/embedder.py` — loads `HuggingFaceEmbeddings` with `normalize_embeddings=True`; module-level cache prevents repeated loading
- `build_vectorstore(chunks, embedding_model, save_path)` — builds FAISS index from chunks, creates parent dirs, persists to disk
- `load_vectorstore(save_path, embedding_model)` — loads persisted index; raises `VectorStoreNotFoundError` (with "Run `make ingest` first" hint) if missing, `VectorStoreCorruptError` on deserialization failure

## Tests

- `tests/unit/test_foundation.py` — Phase 1 unit tests
- `tests/unit/test_ingest.py` — 17 tests covering load, chunking, metadata, and error cases
- `tests/unit/test_embedder.py` — 13 tests covering model caching, build, and load error paths
- `tests/conftest.py` — shared fixtures (sample TXT/PDF/DOCX, empty dir, `sample_chunks`, `mock_embedding_model`, `mock_vectorstore`)

Run tests:

```
make test
```

## Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install all dependencies |
| `make ingest` | Build FAISS vectorstore from `data/sample_docs/` |
| `make serve` | Start FastAPI server on port 8000 |
| `make ui` | Start Streamlit UI on port 8501 |
| `make test` | Run unit + integration tests |
| `make eval` | Run RAGAS evaluation (slow, requires API key) |
| `make coverage` | Run tests and generate coverage report |
| `make lint` | Run ruff and mypy checks |
| `make docker-up` | Start API and UI services via docker-compose |
| `make docker-down` | Stop docker-compose services |
| `make clean` | Remove temporary files and caches |
