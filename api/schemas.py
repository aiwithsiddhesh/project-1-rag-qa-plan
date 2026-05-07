from pydantic import BaseModel, Field

from src.config import TOP_K_MAX, TOP_K_MIN


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    use_hyde: bool | None = None
    top_k: int | None = Field(default=None, ge=TOP_K_MIN, le=TOP_K_MAX)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    num_chunks_retrieved: int
    retrieval_scores: list[float]


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
