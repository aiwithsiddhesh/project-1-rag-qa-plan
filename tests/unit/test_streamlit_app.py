from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from unittest.mock import MagicMock

import pytest

from app.streamlit_app import (
    ApiRequestError,
    REQUEST_TIMEOUT_SECONDS,
    get_api_status,
    is_no_context_answer,
    normalize_api_base_url,
    query_api,
    request_json,
)


@pytest.fixture(autouse=True)
def clear_status_cache() -> None:
    get_api_status.clear()


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_normalize_api_base_url_trims_trailing_slash() -> None:
    assert normalize_api_base_url(" http://localhost:8000/ ") == "http://localhost:8000"


def test_request_json_posts_payload(mocker) -> None:
    opener = mocker.patch(
        "app.streamlit_app.urlopen", return_value=FakeResponse({"answer": "ok"})
    )

    result = request_json(
        "POST", "http://api/query", payload={"question": "What is ML?"}
    )

    assert result == {"answer": "ok"}
    request = opener.call_args.args[0]
    assert request.method == "POST"
    assert json.loads(request.data.decode("utf-8")) == {"question": "What is ML?"}


def test_request_json_uses_rag_friendly_default_timeout(mocker) -> None:
    opener = mocker.patch("app.streamlit_app.urlopen", return_value=FakeResponse({}))

    request_json("GET", "http://api/health")

    assert REQUEST_TIMEOUT_SECONDS == 30.0
    assert opener.call_args.kwargs["timeout"] == 30.0


def test_request_json_raises_for_http_error_detail(mocker) -> None:
    error = HTTPError(
        url="http://api/query",
        code=503,
        msg="Service Unavailable",
        hdrs=None,
        fp=MagicMock(read=lambda: b'{"detail": "Pipeline not ready"}'),
    )
    mocker.patch("app.streamlit_app.urlopen", side_effect=error)

    with pytest.raises(ApiRequestError) as exc_info:
        request_json("POST", "http://api/query", payload={"question": "What is ML?"})

    assert exc_info.value.status_code == 503
    assert exc_info.value.message == "Pipeline not ready"


def test_request_json_raises_api_down_for_url_error(mocker) -> None:
    mocker.patch("app.streamlit_app.urlopen", side_effect=URLError("refused"))

    with pytest.raises(ApiRequestError, match="API is unreachable"):
        request_json("GET", "http://api/health")


def test_request_json_raises_timeout_separately(mocker) -> None:
    mocker.patch("app.streamlit_app.urlopen", side_effect=TimeoutError("slow"))

    with pytest.raises(ApiRequestError, match="API request timed out"):
        request_json("GET", "http://api/health")


def test_get_api_status_ready(mocker) -> None:
    mocker.patch(
        "app.streamlit_app.request_json",
        side_effect=[{"status": "healthy"}, {"status": "healthy"}],
    )

    status = get_api_status("http://api")

    assert status.is_up is True
    assert status.is_ready is True
    assert status.detail == "API ready"


def test_get_api_status_down(mocker) -> None:
    mocker.patch(
        "app.streamlit_app.request_json",
        side_effect=ApiRequestError("API is unreachable."),
    )

    status = get_api_status("http://api")

    assert status.is_up is False
    assert status.is_ready is False


def test_get_api_status_not_ready(mocker) -> None:
    mocker.patch(
        "app.streamlit_app.request_json",
        side_effect=[
            {"status": "healthy"},
            ApiRequestError("Pipeline not ready", status_code=503),
        ],
    )

    status = get_api_status("http://api")

    assert status.is_up is True
    assert status.is_ready is False
    assert status.detail == "Pipeline not ready"


def test_query_api_sends_top_k_and_hyde(mocker) -> None:
    request = mocker.patch(
        "app.streamlit_app.request_json",
        return_value={
            "answer": "Answer",
            "sources": ["doc.txt"],
            "num_chunks_retrieved": 5,
            "retrieval_scores": [],
        },
    )
    mocker.patch("app.streamlit_app.time.perf_counter", side_effect=[10.0, 11.25])

    result, elapsed = query_api("http://api/", "What is ML?", top_k=3, use_hyde=True)

    assert elapsed == 1.25
    assert result["answer"] == "Answer"
    request.assert_called_once_with(
        "POST",
        "http://api/query",
        payload={"question": "What is ML?", "top_k": 3, "use_hyde": True},
    )


def test_is_no_context_answer_detects_fallback_phrase() -> None:
    assert is_no_context_answer(
        "I could not find the answer in the provided documents."
    )


def test_render_message_uses_warning_for_historical_no_context_answer(mocker) -> None:
    chat_message = mocker.patch("app.streamlit_app.st.chat_message")
    warning = mocker.patch("app.streamlit_app.st.warning")
    markdown = mocker.patch("app.streamlit_app.st.markdown")
    mocker.patch("app.streamlit_app._render_sources")

    from app.streamlit_app import _render_message

    _render_message(
        {
            "role": "assistant",
            "content": "I could not find the answer in the provided documents.",
            "sources": [],
            "num_chunks_retrieved": 0,
        }
    )

    chat_message.assert_called_once_with("assistant")
    warning.assert_called_once_with(
        "I could not find the answer in the provided documents."
    )
    markdown.assert_not_called()
