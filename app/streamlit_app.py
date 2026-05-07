from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from src.contracts import DEFAULT_TOP_K, NO_CONTEXT_PHRASE, TOP_K_MAX, TOP_K_MIN

DEFAULT_API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class ApiStatus:
    is_up: bool
    is_ready: bool
    detail: str


class ApiRequestError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def normalize_api_base_url(value: str) -> str:
    return value.strip().rstrip("/") or DEFAULT_API_BASE_URL


def _decode_response(raw_body: bytes) -> dict[str, Any]:
    if not raw_body:
        return {}
    decoded = raw_body.decode("utf-8")
    data = json.loads(decoded)
    if not isinstance(data, dict):
        raise ApiRequestError("API returned an unexpected response.")
    return data


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return _decode_response(response.read())
    except HTTPError as exc:
        response_body = exc.read()
        detail = f"API request failed with status {exc.code}."
        try:
            parsed = _decode_response(response_body)
            detail = str(parsed.get("detail") or detail)
        except (ApiRequestError, json.JSONDecodeError, UnicodeDecodeError):
            pass
        raise ApiRequestError(detail, status_code=exc.code) from exc
    except TimeoutError as exc:
        raise ApiRequestError("API request timed out.") from exc
    except (URLError, OSError) as exc:
        raise ApiRequestError("API is unreachable.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApiRequestError("API returned invalid JSON.") from exc


@st.cache_data(ttl=10)
def get_api_status(api_base_url: str) -> ApiStatus:
    base_url = normalize_api_base_url(api_base_url)
    try:
        request_json("GET", f"{base_url}/health")
    except ApiRequestError as exc:
        return ApiStatus(is_up=False, is_ready=False, detail=exc.message)

    try:
        request_json("GET", f"{base_url}/readiness")
    except ApiRequestError as exc:
        return ApiStatus(is_up=True, is_ready=False, detail=exc.message)

    return ApiStatus(is_up=True, is_ready=True, detail="API ready")


def query_api(
    api_base_url: str, question: str, top_k: int, use_hyde: bool
) -> tuple[dict[str, Any], float]:
    base_url = normalize_api_base_url(api_base_url)
    payload = {"question": question, "top_k": top_k, "use_hyde": use_hyde}
    started = time.perf_counter()
    result = request_json("POST", f"{base_url}/query", payload=payload)
    elapsed_seconds = time.perf_counter() - started
    return result, elapsed_seconds


def is_no_context_answer(answer: str) -> bool:
    return NO_CONTEXT_PHRASE.lower() in answer.lower()


def _render_status(status: ApiStatus) -> None:
    if not status.is_up:
        st.sidebar.error("API down")
    elif not status.is_ready:
        st.sidebar.warning("API starting")
    else:
        st.sidebar.success("API ready")
    st.sidebar.caption(status.detail)


def _render_sources(sources: list[str], num_chunks_retrieved: int) -> None:
    label = f"Sources ({num_chunks_retrieved} chunks)"
    with st.expander(label, expanded=False):
        if not sources:
            st.caption("No cited sources returned.")
            return
        for source in sources:
            st.markdown(f"- `{source}`")


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        content = str(message["content"])
        if message["role"] == "assistant" and is_no_context_answer(content):
            st.warning(content)
        else:
            st.markdown(content)
        if message["role"] == "assistant":
            sources = list(message.get("sources", []))
            num_chunks = int(message.get("num_chunks_retrieved", 0))
            _render_sources(sources, num_chunks)
            elapsed_seconds = message.get("elapsed_seconds")
            if elapsed_seconds is not None:
                st.caption(f"Retrieved {num_chunks} chunks | {elapsed_seconds:.1f}s")


def main() -> None:
    st.set_page_config(page_title="Document Q&A", page_icon="?", layout="wide")
    st.title("Document Q&A")

    with st.sidebar:
        st.header("Controls")
        api_base_url = st.text_input("API URL", value=DEFAULT_API_BASE_URL)
        top_k = st.slider(
            "Top K", min_value=TOP_K_MIN, max_value=TOP_K_MAX, value=DEFAULT_TOP_K
        )
        use_hyde = st.toggle("HyDE", value=False)
        status = get_api_status(api_base_url)
        _render_status(status)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        _render_message(message)

    question = st.chat_input("Ask about your documents")
    if not question:
        return

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant"):
        try:
            result, elapsed_seconds = query_api(api_base_url, question, top_k, use_hyde)
            answer = str(result.get("answer", ""))
            sources = list(result.get("sources", []))
            num_chunks = int(result.get("num_chunks_retrieved", 0))

            if is_no_context_answer(answer):
                st.warning(answer)
            else:
                st.markdown(answer)
            _render_sources(sources, num_chunks)
            st.caption(f"Retrieved {num_chunks} chunks | {elapsed_seconds:.1f}s")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "num_chunks_retrieved": num_chunks,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
        except ApiRequestError as exc:
            if exc.status_code is None:
                st.error("API is down. Start the API service and try again.")
            else:
                st.error(exc.message)


if __name__ == "__main__":
    main()
