from pathlib import Path

from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr
    openai_model: str = "gpt-3.5-turbo"
    vector_store_path: Path = Path("./data/vectorstore")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = 50
    top_k_results: int = Field(default=5, gt=0)
    fetch_k_multiplier: int = Field(default=4, gt=0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    use_hyde: bool = False
    langsmith_tracing: bool = False
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, value: SecretStr) -> SecretStr:
        secret_val = value.get_secret_value().strip()
        if not secret_val:
            raise ValueError("OPENAI_API_KEY must not be empty")
        return SecretStr(secret_val)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("chunk_size")
        if value < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        allowed_levels = {
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }
        normalized = value.upper()
        if normalized not in allowed_levels:
            raise ValueError(
                f"log_level must be one of: {', '.join(sorted(allowed_levels))}"
            )
        return normalized

    @model_validator(mode="after")
    def validate_retrieval_weights(self) -> "Settings":
        if abs((self.bm25_weight + self.dense_weight) - 1.0) > 1e-6:
            raise ValueError("bm25_weight and dense_weight must sum to 1.0")
        return self


settings = Settings()  # type: ignore[call-arg]
