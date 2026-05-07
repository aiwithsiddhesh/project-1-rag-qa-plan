from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    use_hyde: bool | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    num_chunks_retrieved: int
    retrieval_scores: list[float]


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
