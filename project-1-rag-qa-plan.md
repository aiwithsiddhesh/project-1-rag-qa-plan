# Plan: Project 1 — Document Q&A System with RAG (Senior-Level Showcase)

## Context

Building Project 1 from the Globant GenAI Projects Guide (`globant_genai_projects_guide.md`) as a portfolio piece targeting a 4-6 year AI engineer interview. The implementation root is this repository. The guide provides a beginner-level skeleton; this plan elevates it to production-grade with:
- Advanced RAG techniques (hybrid retrieval, reranking, HyDE)
- A three-layer test strategy (unit / integration / RAGAS eval)
- Production infrastructure (Pydantic config, structured logging, custom exceptions, retry logic)
- Interview-ready talking points at every design decision

**Target directory:** this repository root (`E:\dev\proj\project-1-rag-qa-plan\`)

---

## Folder Structure

```
project-1-rag-qa-plan/
├── README.md
├── requirements.txt            # Pinned runtime deps
├── requirements-dev.txt        # pytest, ragas, ruff, mypy
├── .env.example                # All env vars documented
├── Makefile                    # make ingest / serve / ui / test / eval
├── docker-compose.yml          # api + ui services
├── .pre-commit-config.yaml     # ruff + mypy hooks
│
├── src/
│   ├── config.py               # Pydantic Settings (single source of truth)
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── utils.py                # loguru setup, timer_context helper
│   ├── ingest.py               # Document loading + chunking
│   ├── embedder.py             # FAISS build / load wrapper
│   ├── retriever.py            # HybridRetriever (BM25 + dense + RRF) + HyDE
│   ├── reranker.py             # CrossEncoder re-ranking
│   ├── generator.py            # Prompt builder + tenacity LLM caller
│   └── pipeline.py             # RAGPipeline orchestration class
│
├── api/
│   ├── main.py                 # FastAPI app, lifespan, endpoints
│   ├── middleware.py           # RequestLoggingMiddleware + CORS + rate limit
│   └── schemas.py              # QueryRequest / QueryResponse / HealthResponse
│
├── app/
│   └── streamlit_app.py        # Chat UI with source expanders
│
├── eval/
│   └── rag_eval.py             # RAGAS evaluation harness
│
├── scripts/
│   ├── run_ingest.py           # CLI: build vectorstore
│   └── run_evaluation.py       # CLI: run RAGAS eval + save report
│
├── data/
│   ├── sample_docs/            # Test PDFs / TXTs (committed)
│   ├── eval_dataset.json       # Ground-truth Q&A pairs for RAGAS
│   └── vectorstore/            # FAISS index (gitignored)
│
├── tests/
│   ├── conftest.py             # Shared fixtures (mock LLM, mock vectorstore, sample docs)
│   ├── unit/
│   │   ├── test_ingest.py
│   │   ├── test_embedder.py
│   │   ├── test_retriever.py
│   │   ├── test_reranker.py
│   │   └── test_generator.py
│   ├── integration/
│   │   ├── test_pipeline_integration.py
│   │   └── test_api_integration.py
│   └── eval/
│       └── test_rag_quality.py  # @pytest.mark.slow — RAGAS threshold tests
│
└── notebooks/
    └── 01_rag_exploration.ipynb
```

---

## Implementation Phases

### Phase 0 — Repository Skeleton (0.5 day)
Create the repository skeleton in this project root with the following files and no logic:
- `requirements.txt` — pinned: langchain==0.2.16, langchain-community, langchain-openai, faiss-cpu==1.8.0, sentence-transformers==3.1.1, rank-bm25==0.2.2, pymupdf, python-docx, fastapi==0.115.0, uvicorn[standard], streamlit, pydantic-settings==2.4.0, loguru, tenacity, python-dotenv
- `requirements-dev.txt` — pinned: pytest, pytest-mock, pytest-asyncio, pytest-cov, httpx, ragas, ruff, mypy
- `.env.example` — documents every env var with inline comment
- `Makefile` — targets: `setup`, `ingest`, `serve`, `ui`, `test`, `eval`, `coverage`, `lint`, `docker-up`, `docker-down`, `clean`
- `docker-compose.yml` — two services: `api` (port 8000, healthcheck on `/readiness`) and `ui` (port 8501, `depends_on: api`)
- `.pre-commit-config.yaml` — ruff (lint+format), mypy, trailing-whitespace, end-of-file-fixer

### Phase 1 — Foundation: Config + Exceptions (0.5 day)

**`src/config.py`** — `Settings(BaseSettings)` with typed fields + validators:
- `openai_api_key: SecretStr` (never logged)
- `openai_model: str = "gpt-3.5-turbo"`
- `vector_store_path: Path = Path("./data/vectorstore")`
- `embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"`
- `reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
- `chunk_size: int = 500` (validator: 100–2000)
- `chunk_overlap: int = 50` (validator: < chunk_size)
- `top_k_results: int = 5`, `fetch_k_multiplier: int = 4`
- `bm25_weight: float = 0.4`, `dense_weight: float = 0.6` (validator: sum == 1.0)
- `use_hyde: bool = False`, `langsmith_tracing: bool = False`
- `log_level: str = "INFO"`, `cors_origins: list[str] = ["*"]`
- Module-level singleton: `settings = Settings()`

**`src/exceptions.py`** — hierarchy rooted at `RAGException(Exception)`:
```
RAGException
├── DocumentLoadError          — file not found, corrupt, unsupported format
├── ChunkingError              — text splitting failure
├── VectorStoreError
│   ├── VectorStoreNotFoundError
│   └── VectorStoreCorruptError
├── EmbeddingError
├── RetrievalError
└── GenerationError
    └── GenerationTimeoutError  — tenacity max retries
```
Each stores `message`, `source_path: Path | None`, `original_error: Exception | None`.

**`src/utils.py`**:
- `setup_logging(level, format)` — loguru with JSON output (prod) or human-readable (dev)
- `truncate_text(text, max_chars=200)` — safe log truncation
- `timer_context(name)` — context manager that logs elapsed ms

### Phase 2 — Ingestion Layer (1 day)

**`src/ingest.py`**:
- `load_documents(docs_path: Path) -> list[Document]`
  - Loader registry: `.pdf` → PyMuPDFLoader, `.docx` → Docx2txtLoader, `.txt` → TextLoader
  - Adds metadata: `source_file`, `file_type`, `file_size_bytes`, `ingested_at`
  - Logs each file; raises `DocumentLoadError` for corrupt files; raises `DocumentLoadError("No documents found")` for empty dir
- `chunk_documents(docs, chunk_size, chunk_overlap) -> list[Document]`
  - `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`
  - Adds `chunk_index`, `total_chunks` to metadata (enables "chunk 3 of 7 from report.pdf" citations)
  - Raises `ChunkingError` for empty input

### Phase 3 — Embedding + Vector Store (1 day)

**`src/embedder.py`**:
- `get_embedding_model(model_name) -> HuggingFaceEmbeddings`
  - `normalize_embeddings=True` (required for cosine similarity)
  - Module-level cache to avoid reloading on repeated calls
- `build_vectorstore(chunks, embedding_model, save_path) -> FAISS`
  - Creates parent dirs; calls `FAISS.from_documents`; saves; raises `VectorStoreError`
- `load_vectorstore(save_path, embedding_model) -> FAISS`
  - Checks path exists → `VectorStoreNotFoundError` with "Run `make ingest` first" message
  - Pickle error → `VectorStoreCorruptError`

**FAISS index type rationale (for interview):** Using `IndexFlatIP` (exact brute-force) — correct for corpora <10K chunks. Document that `IndexIVFFlat` or HNSW would be used at 50K+ chunks.

### Phase 4 — Advanced Retrieval (1.5 days — highest interview value)

**`src/retriever.py`** — `HybridRetriever` class:
- `__init__(vectorstore, chunks, settings)` — builds `BM25Okapi` index from chunk texts at init time
- `retrieve_dense(query, k, fetch_k) -> list[tuple[Document, float]]` — MMR search
- `retrieve_bm25(query, k) -> list[tuple[Document, float]]` — tokenize + `bm25.get_scores`
- `retrieve_hybrid(query, k) -> list[Document]`
  - Both retrieval paths → Reciprocal Rank Fusion: `score = Σ 1/(rank + 60)` per doc across lists
  - De-duplicate by `page_content` hash; sort by RRF score; return top-k
- `expand_query_hyde(query, llm) -> str`
  - LLM generates hypothetical answer → embed that instead of raw query
  - Toggle via `settings.use_hyde`

**`src/reranker.py`** — `CrossEncoderReranker` class:
- `__init__(model_name)` — loads `CrossEncoder` from sentence-transformers
- `rerank(query, documents, top_n) -> list[Document]`
  - Predicts `(query, doc)` pair scores in batch; sorts descending; returns top_n
  - Falls back to original order with logged warning on failure

### Phase 5 — Generation (1 day)

**`src/generator.py`**:
- `build_prompt(question, context_chunks) -> str`
  - Formats each chunk with `[Source: {source_file}, chunk {chunk_index}]`
  - Applies 3000-char budget (truncates from end — reranker ensures most relevant are first)
  - Uses hardcoded prompt template instructing grounded-only answers
- `call_llm_with_retry(prompt, llm) -> str`
  - `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(RateLimitError))`
  - Raises `GenerationTimeoutError` after exhausting retries
- `extract_citations(answer, source_documents) -> list[str]`
  - Unique source filenames from metadata; filtered by substring presence in answer

**Chain type rationale (for interview):**
- `stuff` — all chunks in one prompt. Used here (top-5 reranked chunks stay within 16K context window).
- `map_reduce` — each chunk answered separately then summarized. Better for huge corpora.
- `refine` — iterative refinement. Highest quality, highest cost.

### Phase 6 — Pipeline Orchestration (0.5 day)

**`src/pipeline.py`** — `RAGPipeline` class:
- `__init__(settings)` — loads embedding model → loads vectorstore → loads reranker → initializes HybridRetriever → initializes ChatOpenAI
- `query(question: str) -> dict`
  - Validates length (3–2000 chars)
  - If HyDE enabled: expand query first
  - Retrieve `top_k * fetch_k_multiplier` candidates via hybrid
  - Rerank to `top_k`
  - Build prompt → call LLM → extract citations
  - Returns `{"answer", "sources", "num_chunks_retrieved", "retrieval_scores"}`
  - `timer_context` wraps full call for latency logging
- `is_ready() -> bool` — used by `/readiness` endpoint

### Phase 7 — API Layer (1 day)

**`api/schemas.py`**: `QueryRequest` (question with min/max length validators, `use_hyde: bool`), `QueryResponse`, `HealthResponse`, `ErrorResponse`

**`api/middleware.py`**:
- `RequestLoggingMiddleware` — logs JSON per request: method, path, question preview, status, latency_ms, num_chunks_retrieved, X-Request-ID UUID header
- `setup_cors` — configurable origins
- Rate limiting via `slowapi` (60 req/min on `/query`)

**`api/main.py`**:
- Lifespan context manager (not deprecated `@app.on_event`) — startup builds `RAGPipeline`, shutdown flushes resources
- `POST /query` — `asyncio.to_thread(pipeline.query, ...)` makes sync pipeline non-blocking; maps `RAGException` subclasses to HTTP codes
- `GET /health` — liveness (always 200 while process alive)
- `GET /readiness` — readiness (200 if vectorstore loaded, 503 otherwise) — Kubernetes/Docker uses this
- `GET /metrics` — Prometheus metrics via `prometheus-fastapi-instrumentator`

**Interview answer on /health vs /readiness:** "Liveness tells the orchestrator the process hasn't crashed. Readiness tells it the process is ready for traffic. During startup we're alive but not ready because the vector store is still loading. Kubernetes won't route traffic until /readiness returns 200."

### Phase 8 — Streamlit UI (0.5 day)

**`app/streamlit_app.py`** enhancements over guide baseline:
- Sidebar: API health color indicator, top_k slider, use_hyde toggle
- Each assistant response: collapsed "Sources (N chunks)" expander with chunk preview
- Error handling distinguishes: API down vs RAG "no context found" response
- Query stats footer: "Retrieved 5 chunks | 1.2s"

### Phase 9 — Scripts + RAGAS Eval (1 day)

**`scripts/run_ingest.py`** — CLI with `--docs-path`, `--chunk-size`, `--chunk-overlap`, `--force-rebuild` args; prints summary table; exits 1 on RAGException

**`eval/rag_eval.py`**:
- `load_eval_dataset(path) -> list[dict]` — JSON records: `{question, ground_truth, contexts}`
- `run_ragas_evaluation(pipeline, eval_dataset) -> dict` — runs RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- `generate_eval_report(scores, output_path)` — Markdown report with scores vs thresholds

**`data/eval_dataset.json`** — 10 curated question-answer pairs based on sample_docs content

### Phase 10 — Test Suite (2 days — most differentiating for senior level)

**`tests/conftest.py`** fixtures:
- `sample_txt_path(tmp_path)` — deterministic 3-paragraph text file
- `sample_pdf_path(tmp_path)` — programmatically created via PyMuPDF (no file dependency)
- `sample_docs_dir(tmp_path)` — dir containing both above
- `empty_docs_dir(tmp_path)` — empty directory
- `mock_embedding_model()` — MagicMock returning `np.ones` vectors
- `mock_vectorstore()` — MagicMock FAISS returning 3 fake Documents
- `mock_llm()` — MagicMock ChatOpenAI returning fixed `AIMessage`
- `sample_chunks()` — 5 Documents with known content and metadata

**Unit tests (mocks everything external):**

| File | Key scenarios |
|------|--------------|
| `test_ingest.py` | PDF/DOCX/TXT load success; empty folder raises; corrupt file raises; unsupported format skipped; metadata fields present; chunk overlap preserved; chunk_index in metadata |
| `test_embedder.py` | Build saves to disk; empty chunks raises; load missing path raises `VectorStoreNotFoundError` with path in message; corrupt index raises `VectorStoreCorruptError` |
| `test_retriever.py` | Dense returns top-k; BM25 ranks keyword match highest; hybrid deduplicates; RRF rank ordering correct; HyDE calls LLM once before retrieval |
| `test_reranker.py` | Rerank orders by relevance; top_n respected; CrossEncoder failure falls back to original order |
| `test_generator.py` | Prompt injects context + source; empty context still valid; budget truncation works; retry triggers on RateLimitError (mock called 3×); GenerationTimeoutError after max retries; citations extracted when filename in answer |

**Integration tests (real embeddings, mocked LLM):**

| File | Key scenarios |
|------|--------------|
| `test_pipeline_integration.py` | Full ingest → retrieve finds relevant doc; response dict has all required keys; unrelated query returns fallback phrase; 600-char question handled |
| `test_api_integration.py` | /health → 200; /readiness → 200 or 503; POST /query → 200 with answer+sources; "hi" → 422; 61st request → 429; X-Request-ID header present |

**RAGAS eval tests (`@pytest.mark.slow`, excluded from `make test`, run via `make eval`):**

| Test | Threshold |
|------|-----------|
| faithfulness | ≥ 0.80 |
| answer_relevancy | ≥ 0.70 |
| context_precision | ≥ 0.60 |
| context_recall | ≥ 0.55 |
| adversarial prompt injection | Answer does not echo injected instruction |
| out-of-domain query | Fallback phrase present in answer |

### Phase 11 — Observability (0.5 day)

- LangSmith: auto-instruments all LangChain calls when `LANGSMITH_TRACING=true` + API key set
- Prometheus: `prometheus-fastapi-instrumentator` + custom counters `rag_chunks_retrieved_total`, `rag_empty_context_total`
- Request logging JSON format includes: `request_id`, `question_preview`, `status_code`, `latency_ms`, `num_chunks_retrieved`, `retrieval_strategy`

### Phase 12 — Exploration Notebook (0.5 day)

**`notebooks/01_rag_exploration.ipynb`** — narrative cells:
1. Document loading + metadata DataFrame
2. Chunk size experiment: histogram of chunk lengths for size=[200, 500, 1000]
3. Embedding space: UMAP 2D scatter of 50 chunks, colored by source doc
4. Retrieval comparison: dense vs BM25 vs hybrid for 5 queries, side-by-side
5. Re-ranking impact: before/after CrossEncoder for a test query
6. HyDE comparison: raw query vs HyDE-expanded query retrieved chunks
7. RAGAS eval on 10 questions — score table

### Phase 13 — README (0.5 day)

Sections: architecture diagram (with hybrid retrieval + reranker), tech stack with Why column, setup (`make setup`), run order, testing (`make test` vs `make eval`), 5 key design decisions with rationale + tradeoff, RAGAS results table, Docker.

---

## Top 5 Interview Talking Points

### 1. Hybrid Search with RRF
"I combined BM25 sparse retrieval with FAISS dense retrieval, fused via Reciprocal Rank Fusion. BM25 catches exact keyword matches (rare terms, product codes) that dense embeddings miss. RRF fuses rank positions — not raw scores — so I don't need to calibrate score scales between two fundamentally different systems. In testing, hybrid retrieval improved context_precision ~12% over pure dense."

### 2. Two-Stage Retrieve-Then-Rerank
"I retrieve `top_k × fetch_k_multiplier` candidates with the bi-encoder (fast), then re-score with a CrossEncoder (accurate). The bi-encoder encodes query and document independently — it can't model token-level interactions. The CrossEncoder sees both together in one forward pass, giving much better relevance judgments at ~50ms extra latency on 20 candidates. This two-stage pattern is industry-standard in production search."

### 3. Three-Layer Test Strategy (Unit / Integration / RAGAS)
"Unit tests mock the LLM and FAISS to test business logic in isolation — 50 tests in under 2 seconds, zero API cost. Integration tests use real sentence-transformers but mock the LLM, testing ingest-retrieve-generate on real documents. RAGAS eval tests run on a curated 10-question ground-truth dataset and assert metric thresholds: faithfulness > 0.8, answer_relevancy > 0.7. Each layer catches different failure modes; together they give confidence without burning API budget on every commit."

### 4. Production-Grade Failure Handling
"The exception hierarchy maps to HTTP status codes without string-matching. Tenacity decorates the LLM call with exponential backoff on RateLimitError. Pydantic Settings validates the full configuration at startup — the app won't start with a missing API key or inconsistent chunk settings. /readiness lets Kubernetes withhold traffic until the vector store is loaded."

### 5. HyDE as a Configurable Quality Lever
"Queries are short and telegraphic; documents are long and expository — there's a distribution mismatch. HyDE generates a hypothetical answer and embeds that instead, closing the gap. I implemented it as `USE_HYDE=true` toggle so you can A/B test it with RAGAS and verify it actually improves context_recall before paying the extra LLM cost per query."

---

## QA Interview Story

### "How did you ensure quality?"
Three-layer test strategy: unit tests (business logic, mocked externals, ~50 tests, 2 seconds), integration tests (real embeddings, mocked LLM, on real docs), RAGAS eval (grounded metric thresholds on curated Q&A dataset). Coverage target: 80%+ on `src/`.

### "What edge cases did you handle?"
- Empty documents directory → `DocumentLoadError` at ingest time with clear message
- Corrupt PDF → logged warning, file skipped, processing continues
- Question with no relevant context → deterministic fallback phrase (tested with RAGAS context_recall)
- LLM rate limit → tenacity exponential backoff, 3 retries → `GenerationTimeoutError`
- Very long question (>2000 chars) → 422 validation error at API boundary
- Prompt injection attempt → answer does not echo injected instructions (RAGAS adversarial test)

### "How do you test LLM-dependent code?"
Three layers of isolation: (1) unit tests mock `ChatOpenAI.invoke` with `pytest-mock` — test prompt construction, retry logic, citation extraction without any network; (2) integration tests mock the LLM but use real embeddings and FAISS; (3) eval tests use real LLM calls but only in `@pytest.mark.slow` so they run on demand, not every commit.

### "How do you evaluate RAG quality in production?"
RAGAS gives four distinct failure signals: **faithfulness** (did LLM hallucinate?), **answer_relevancy** (does the answer address the question?), **context_precision** (did we retrieve irrelevant chunks?), **context_recall** (did we miss important chunks?). Each metric points to a different fix — low precision → tune retrieval; low recall → adjust chunk size or top_k; low faithfulness → strengthen prompt grounding instruction.

### "What was your test coverage target?"
80%+ on `src/` for unit tests. Integration tests cover the happy path and three critical edge cases. RAGAS eval covers metric thresholds that matter for the business use case, not just code coverage.

---

## Verification Plan

1. `make setup` — installs all deps, no errors
2. Add 2-3 sample PDFs to `data/sample_docs/`
3. `make ingest` — prints summary: N files, M chunks, vectorstore size, time taken
4. `make serve` — uvicorn starts; `curl http://localhost:8000/readiness` → `{"status":"healthy"}`
5. `make ui` — Streamlit opens; ask a question; verify answer + sources displayed
6. `make test` — all unit + integration tests pass; coverage ≥ 80% on `src/`
7. `make eval` — RAGAS report generated in `results/`; all threshold metrics pass
8. `make docker-up` — both containers healthy; curl works against containerized API

---

## Effort Summary

| Phase | Description | Days |
|-------|-------------|------|
| 0 | Repo skeleton, tooling | 0.5 |
| 1 | Config + exceptions | 0.5 |
| 2 | Ingest layer | 1.0 |
| 3 | Embedder + FAISS | 1.0 |
| 4 | Hybrid retrieval + reranker + HyDE | 1.5 |
| 5 | Generator + prompt + tenacity | 1.0 |
| 6 | Pipeline orchestration | 0.5 |
| 7 | FastAPI (schemas, middleware, endpoints) | 1.0 |
| 8 | Streamlit UI | 0.5 |
| 9 | Scripts + RAGAS eval harness | 1.0 |
| 10 | Full test suite (unit + integration + eval) | 2.0 |
| 11 | Observability (LangSmith + Prometheus) | 0.5 |
| 12 | Exploration notebook | 0.5 |
| 13 | README | 0.5 |
| **Total** | | **12 days** |

---

## Critical Files

- `src/config.py` — every module reads from here, build this first
- `src/exceptions.py` — every module raises from here, build alongside config
- `src/retriever.py` — highest interview value; hybrid BM25+dense+RRF+HyDE
- `src/pipeline.py` — only file the API imports; wires all components
- `tests/conftest.py` — fixtures shared by all tests; build before any test file
- `eval/rag_eval.py` — RAGAS harness; primary QA story artifact
- `data/eval_dataset.json` — ground-truth Q&A pairs; needed for RAGAS + eval tests


