from collections.abc import Iterator
from contextlib import contextmanager
import sys
from time import perf_counter
from typing import Literal

from loguru import logger


LogFormat = Literal["json", "human"]


def setup_logging(level: str = "INFO", format: LogFormat = "human") -> None:
    """Configure loguru logging for local development or structured production logs."""

    logger.remove()
    if format == "json":
        logger.add(
            sink=sys.stderr,
            level=level.upper(),
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
        return

    logger.add(
        sink=sys.stderr,
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        backtrace=False,
        diagnose=False,
    )


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Trim text for logs without splitting newline-heavy previews across records."""

    if max_chars < 0:
        raise ValueError("max_chars must be greater than or equal to 0")

    normalized = " ".join(text.lstrip()[: max_chars * 2].split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return "." * max_chars
    return f"{normalized[: max_chars - 3]}..."


@contextmanager
def timer_context(name: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info(
            "{name} completed in {elapsed_ms:.2f} ms", name=name, elapsed_ms=elapsed_ms
        )
