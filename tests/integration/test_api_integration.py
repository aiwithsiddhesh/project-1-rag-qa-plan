"""Integration tests for the FastAPI API layer (Phase 7)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app

_MOCK_RESULT = {
    "answer": "Machine learning is a subset of artificial intelligence.",
    "sources": ["ml_intro.txt"],
    "num_chunks_retrieved": 5,
    "retrieval_scores": [],
}


@pytest.fixture()
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.is_ready.return_value = True
    pipeline.query.return_value = _MOCK_RESULT
    return pipeline


@pytest.fixture()
def client(mock_pipeline: MagicMock):
    """TestClient backed by a mocked, ready pipeline."""
    with TestClient(app, raise_server_exceptions=False) as c:
        api_main._pipeline = mock_pipeline
        yield c
    api_main._pipeline = None


@pytest.fixture()
def not_ready_client():
    """TestClient with no pipeline (simulates failed startup)."""
    with TestClient(app, raise_server_exceptions=False) as c:
        api_main._pipeline = None
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_health_returns_200_even_without_pipeline(
        self, not_ready_client: TestClient
    ) -> None:
        r = not_ready_client.get("/health")
        assert r.status_code == 200


class TestReadinessEndpoint:
    def test_readiness_returns_200_when_pipeline_ready(
        self, client: TestClient
    ) -> None:
        r = client.get("/readiness")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_readiness_returns_503_when_pipeline_not_ready(
        self, not_ready_client: TestClient
    ) -> None:
        r = not_ready_client.get("/readiness")
        assert r.status_code == 503


class TestQueryEndpoint:
    def test_query_returns_200_with_valid_question(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "What is machine learning?"})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "sources" in body
        assert "num_chunks_retrieved" in body
        assert "retrieval_scores" in body

    def test_query_returns_422_for_question_too_short(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "hi"})
        assert r.status_code == 422

    def test_query_returns_422_for_question_too_long(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "x" * 2001})
        assert r.status_code == 422

    def test_query_returns_503_when_pipeline_not_ready(
        self, not_ready_client: TestClient
    ) -> None:
        r = not_ready_client.post(
            "/query", json={"question": "What is machine learning?"}
        )
        assert r.status_code == 503

    def test_query_passes_use_hyde_to_pipeline(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        client.post("/query", json={"question": "What is ML?", "use_hyde": True})
        mock_pipeline.query.assert_called_once_with("What is ML?", True, None)

    def test_query_default_use_hyde_is_none(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        client.post("/query", json={"question": "What is ML?"})
        mock_pipeline.query.assert_called_once_with("What is ML?", None, None)

    def test_query_passes_top_k_to_pipeline(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        r = client.post(
            "/query", json={"question": "What is ML?", "use_hyde": False, "top_k": 3}
        )
        assert r.status_code == 200
        mock_pipeline.query.assert_called_once_with("What is ML?", False, 3)

    def test_query_rejects_top_k_below_min(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "What is ML?", "top_k": 0})
        assert r.status_code == 422

    def test_query_rejects_top_k_above_max(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "What is ML?", "top_k": 21})
        assert r.status_code == 422


class TestMiddleware:
    def test_x_request_id_header_present_on_health(self, client: TestClient) -> None:
        r = client.get("/health")
        assert "x-request-id" in r.headers

    def test_x_request_id_header_present_on_query(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "What is machine learning?"})
        assert "x-request-id" in r.headers

    def test_x_request_id_is_uuid_format(self, client: TestClient) -> None:
        r = client.get("/health")
        request_id = r.headers.get("x-request-id", "")
        assert len(request_id) == 36
        assert request_id.count("-") == 4


class TestRateLimit:
    def test_61st_request_returns_429(self, mock_pipeline: MagicMock) -> None:
        from api.middleware import limiter

        # slowapi.Limiter exposes no public counter-reset API; _storage.reset()
        # is the only way to clear the in-memory window without recreating the
        # whole app. Documented as a known exception to the private-attribute
        # rule — wrapped in try/skip to guard against future library changes.
        try:
            limiter._storage.reset()
        except Exception:
            pytest.skip("Cannot reset limiter storage — skipping rate limit test")

        with TestClient(app, raise_server_exceptions=False) as c:
            api_main._pipeline = mock_pipeline
            payload = {"question": "What is machine learning?"}
            for i in range(60):
                r = c.post("/query", json=payload)
                assert r.status_code == 200, f"Request {i + 1} returned {r.status_code}"
            r = c.post("/query", json=payload)
            assert r.status_code == 429
        api_main._pipeline = None
