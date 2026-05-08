# Document Q&A RAG System

This repository is being built phase by phase from `project-1-rag-qa-plan.md`.

Current status: Phase 11 complete — Observability. Phase 12 (Exploration Notebook) next.

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

**Phase 4 — Advanced Retrieval**
- `HybridRetriever` in `src/retriever.py` — BM25 + dense MMR retrieval fused via Reciprocal Rank Fusion (RRF k=60); deduplicates by `page_content`; configurable `mmr_lambda` (default 0.7)
- `retrieve_dense(query, k, fetch_k)` — FAISS MMR search; `lambda_mult` from `settings.mmr_lambda`
- `retrieve_bm25(query, k)` — BM25Okapi over all chunks; tokenization consistent at index and query time via shared `_tokenize` helper
- `retrieve_hybrid(query, k)` — fuses both retrieval paths with RRF; docs appearing in both lists rank higher than docs in one
- `expand_query_hyde(query, llm)` — generates a hypothetical answer to close the query-document distribution gap; falls back to original query on failure
- `CrossEncoderReranker` in `src/reranker.py` — batch `CrossEncoder.predict` on `(query, doc)` pairs; sorted descending; falls back to original order with logged warning on failure

**Phase 5 — Generation**
- `build_prompt(question, context_chunks)` in `src/generator.py` — formats each chunk as `[Source: {source_file}, chunk {chunk_index}]`, applies a 3000-char context budget (truncates from the end so most-relevant chunks are preserved), embeds in a hardcoded grounded-answer template
- `call_llm_with_retry(prompt, llm)` — tenacity retry with `stop_after_attempt(3)` and `wait_exponential(min=1, max=10)` on `openai.RateLimitError`; raises `GenerationTimeoutError` after retries are exhausted; raises `GenerationError` for all other failures
- `extract_citations(answer, source_documents)` — returns unique `source_file` names whose filename appears as a substring in the answer text

**Phase 6 — Pipeline Orchestration**
- `RAGPipeline` in `src/pipeline.py` — wires all components: loads embedding model → loads vectorstore → extracts chunks → initializes `CrossEncoderReranker` → initializes `HybridRetriever` → initializes `ChatOpenAI`
- `query(question, use_hyde=None, top_k=None)` — validates length (3–2000 chars), optional HyDE expansion (overridable per-call), optional per-query `top_k` override (1–20), hybrid retrieval of `top_k × fetch_k_multiplier` candidates, reranks to `top_k`, builds prompt, calls LLM with retry, extracts citations; wrapped in `timer_context` for latency logging; returns `{"answer", "sources", "num_chunks_retrieved", "retrieval_scores", "contexts", "retrieval_strategy"}`
- `is_ready()` — returns `True` when vectorstore loaded; used by the `/readiness` endpoint

**Phase 7 — API Layer**
- `api/schemas.py` — `QueryRequest` (question 3–2000 chars, `use_hyde: bool`, optional `top_k` 1–20), `QueryResponse`, `HealthResponse`, `ErrorResponse`
- `api/middleware.py` — `RequestLoggingMiddleware` (JSON logs: request_id, method, path, status, latency_ms, question_preview, num_chunks_retrieved, retrieval_strategy; injects `X-Request-ID` UUID response header), `setup_cors`, `limiter` (slowapi, 60 req/min on `/query`)
- `api/main.py` — FastAPI app with lifespan (startup builds `RAGPipeline`, shutdown clears it); `POST /query` (async via `asyncio.to_thread`, supports per-query `use_hyde` and `top_k`, maps `RAGException` subclasses to HTTP 503/504/422); `GET /health` (liveness, always 200); `GET /readiness` (503 if pipeline not ready — Kubernetes uses this before routing traffic); `GET /metrics` (Prometheus via `prometheus-fastapi-instrumentator` plus RAG custom counters)

**Phase 8 — Streamlit UI**
- `app/streamlit_app.py` — chat UI backed by the FastAPI service; sidebar controls for API URL, API health/readiness, `top_k`, and HyDE
- Assistant responses display answer text, collapsed source expander, and query stats footer (`Retrieved N chunks | X.Xs`)
- UI distinguishes unreachable API errors, API/RAG service errors, and grounded no-context fallback responses

**Phase 9 — Scripts + RAGAS Eval**
- `scripts/run_ingest.py` — CLI for building the FAISS vectorstore from `data/sample_docs/`; supports `--docs-path`, `--chunk-size`, `--chunk-overlap`, and `--force-rebuild`
- `eval/rag_eval.py` — helpers to load eval JSON records, run RAGAS metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`), and generate a Markdown score report
- `data/eval_dataset.json` — 10 grounded Q&A records based on the committed NIST sample documents

**Phase 10 — Full Test Suite**
- `tests/integration/test_pipeline_integration.py` — local FAISS + deterministic embedding integration tests with mocked LLM and reranker
- `tests/eval/test_rag_quality.py` — slow/on-demand eval quality checks for metric thresholds, adversarial prompt injection, and out-of-domain fallback behavior
- `pytest.ini` — registers the `slow` marker so default tests stay clean and eval tests remain opt-in

**Phase 11 — Observability**
- `api/observability.py` — LangSmith tracing setup and Prometheus custom counters
- LangSmith tracing defaults to requested in config, but activates only when `LANGSMITH_API_KEY` is configured; missing keys log a warning and keep tracing disabled
- `/metrics` exposes `rag_chunks_retrieved_total` and `rag_empty_context_total`
- Request logs include `retrieval_strategy` (`hybrid` or `hybrid+hyde`) alongside request and retrieval metadata

## Tests

- `tests/unit/test_foundation.py` — Phase 1 unit tests
- `tests/unit/test_ingest.py` — 17 tests covering load, chunking, metadata, and error cases
- `tests/unit/test_embedder.py` — 13 tests covering model caching, build, and load error paths
- `tests/unit/test_retriever.py` — tests covering dense/BM25/hybrid retrieval, RRF ordering, deduplication, and HyDE fallback
- `tests/unit/test_reranker.py` — tests covering relevance ordering, top_n, CrossEncoder failure fallback
- `tests/unit/test_generator.py` — tests covering prompt construction, source headers, context budget truncation, tenacity retry behaviour, GenerationTimeoutError, GenerationError, and citation extraction
- `tests/unit/test_pipeline.py` — tests covering RAGPipeline init wiring, is_ready, query happy path, HyDE toggle (settings and per-call override), `top_k` override, question length validation, reranker call, and retrieval candidate count
- `tests/unit/test_streamlit_app.py` — tests covering Streamlit API helper behavior, health/readiness handling, API-down errors, `top_k`/HyDE payloads, and no-context detection
- `tests/unit/test_run_ingest.py` — tests covering ingestion CLI success, RAGException exit handling, and `--force-rebuild` behavior
- `tests/unit/test_rag_eval.py` — tests covering eval dataset validation, report generation, and RAGAS invocation with mocked dependencies
- `tests/unit/test_observability.py` — tests covering LangSmith tracing enable/disable behavior without real keys
- `tests/integration/test_pipeline_integration.py` — integration tests covering full pipeline response shape, relevant retrieval, no-context fallback, and 600-char question handling with local FAISS and mocked paid components
- `tests/integration/test_api_integration.py` — integration tests covering /health, /readiness (200 and 503), POST /query (200, 422 for short/long input and invalid `top_k`, 503 when not ready), X-Request-ID header, use_hyde/top_k passthrough, and rate-limit 429
- `tests/eval/test_rag_quality.py` — `@pytest.mark.slow` eval tests for RAGAS-style thresholds, prompt-injection behavior, and out-of-domain fallback
- `tests/conftest.py` — shared fixtures (sample TXT/PDF/DOCX, empty dir, `sample_chunks`, `mock_embedding_model`, `mock_vectorstore`, `mock_llm`)

Run tests:

```
make test
```

## Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install all dependencies |
| `make ingest` | Build FAISS vectorstore from `data/sample_docs/`; use `python scripts/run_ingest.py --force-rebuild` to overwrite an existing index |
| `make serve` | Start FastAPI server on port 8000 |
| `make ui` | Start Streamlit UI on port 8501 |
| `make test` | Run unit + integration tests |
| `make eval` | Run RAGAS evaluation (slow, requires API key) |
| `make coverage` | Run tests and generate coverage report |
| `make lint` | Run ruff and mypy checks |
| `make docker-up` | Start API and UI services via docker-compose |
| `make docker-down` | Stop docker-compose services |
| `make clean` | Remove temporary files and caches |
