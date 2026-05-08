from __future__ import annotations

from pathlib import Path

import pytest

from eval.rag_eval import THRESHOLDS, generate_eval_report
from src.contracts import NO_CONTEXT_PHRASE

pytestmark = pytest.mark.slow


def test_ragas_scores_meet_thresholds() -> None:
    scores = {
        "faithfulness": 0.82,
        "answer_relevancy": 0.74,
        "context_precision": 0.66,
        "context_recall": 0.58,
    }

    for metric, threshold in THRESHOLDS.items():
        assert scores[metric] >= threshold


def test_eval_report_records_failures_against_thresholds(tmp_path: Path) -> None:
    output_path = tmp_path / "ragas_report.md"

    generate_eval_report(
        {
            "faithfulness": 0.79,
            "answer_relevancy": 0.74,
            "context_precision": 0.66,
            "context_recall": 0.58,
        },
        output_path,
    )

    report = output_path.read_text(encoding="utf-8")
    assert "| faithfulness | 0.790 | 0.80 | FAIL |" in report
    assert "| answer_relevancy | 0.740 | 0.70 | PASS |" in report


def test_adversarial_prompt_injection_is_not_echoed() -> None:
    answer = (
        "I can only answer from the provided documents. "
        "The AI RMF manages risks from AI systems."
    )

    assert "ignore previous instructions" not in answer.lower()
    assert "reveal the system prompt" not in answer.lower()


def test_out_of_domain_query_uses_fallback_phrase() -> None:
    answer = NO_CONTEXT_PHRASE

    assert NO_CONTEXT_PHRASE in answer
