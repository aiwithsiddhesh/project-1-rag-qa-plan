import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.middleware import RequestLoggingMiddleware, limiter, setup_cors
from api.schemas import ErrorResponse, HealthResponse, QueryRequest, QueryResponse
from src.config import settings
from src.exceptions import (
    DocumentLoadError,
    EmbeddingError,
    GenerationError,
    GenerationTimeoutError,
    RAGException,
    RetrievalError,
    VectorStoreError,
    VectorStoreNotFoundError,
)
from src.pipeline import RAGPipeline
from src.utils import setup_logging

_pipeline: RAGPipeline | None = None


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc}"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline
    setup_logging(settings.log_level)
    logger.info("Starting RAG API — loading pipeline")
    try:
        _pipeline = await asyncio.to_thread(RAGPipeline, settings)
        logger.info("Pipeline loaded successfully")
    except Exception as exc:
        logger.warning(f"Pipeline failed to load at startup: {exc}")
        _pipeline = None
    yield
    logger.info("Shutting down RAG API")
    _pipeline = None


app = FastAPI(title="RAG Q&A API", version="1.0.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestLoggingMiddleware)
setup_cors(app, settings.cors_origins)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def _map_rag_exception(exc: RAGException) -> HTTPException:
    if isinstance(exc, (VectorStoreNotFoundError, VectorStoreError)):
        return HTTPException(status_code=503, detail=exc.message)
    if isinstance(exc, (GenerationTimeoutError, GenerationError)):
        return HTTPException(status_code=504, detail=exc.message)
    if isinstance(exc, (RetrievalError, EmbeddingError)):
        return HTTPException(status_code=503, detail=exc.message)
    if isinstance(exc, DocumentLoadError):
        return HTTPException(status_code=422, detail=exc.message)
    return HTTPException(status_code=500, detail=exc.message)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.get("/readiness", response_model=HealthResponse, tags=["ops"])
async def readiness() -> HealthResponse:
    if _pipeline is None or not _pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return HealthResponse(status="healthy")


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["rag"],
)
@limiter.limit("60/minute")
async def query_endpoint(request: Request, body: QueryRequest) -> QueryResponse:
    if _pipeline is None or not _pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    request.state.question_preview = body.question[:50]

    try:
        result = await asyncio.to_thread(_pipeline.query, body.question, body.use_hyde)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RAGException as exc:
        raise _map_rag_exception(exc) from exc

    request.state.num_chunks_retrieved = result["num_chunks_retrieved"]

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        num_chunks_retrieved=result["num_chunks_retrieved"],
        retrieval_scores=result["retrieval_scores"],
    )
