from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.55,
}


class QueryPipeline(Protocol):
    def query(self, question: str) -> dict[str, Any]:
        """Return a RAG response dict for one question."""


def load_eval_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate JSON records for RAGAS evaluation."""
    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError("Eval dataset must be a non-empty JSON list.")

    required = {"question", "ground_truth", "contexts"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Eval record {index} must be an object.")
        missing = required - record.keys()
        if missing:
            raise ValueError(
                f"Eval record {index} missing required field(s): "
                f"{', '.join(sorted(missing))}"
            )
        if not isinstance(record["contexts"], list) or not record["contexts"]:
            raise ValueError(f"Eval record {index} contexts must be a non-empty list.")
    return records


def _score_to_dict(result: Any) -> dict[str, float]:
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        means = df.mean(numeric_only=True).to_dict()
        return {str(key): float(value) for key, value in means.items()}
    return {str(key): float(value) for key, value in dict(result).items()}


def run_ragas_evaluation(
    pipeline: QueryPipeline,
    eval_dataset: list[dict[str, Any]],
) -> dict[str, float]:
    """Run RAGAS metrics against pipeline answers for the provided dataset."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    evaluation_rows: list[dict[str, Any]] = []
    for record in eval_dataset:
        result = pipeline.query(record["question"])
        evaluation_rows.append(
            {
                "question": record["question"],
                "answer": result["answer"],
                "contexts": record["contexts"],
                "ground_truth": record["ground_truth"],
            }
        )

    ragas_result = evaluate(
        Dataset.from_list(evaluation_rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return _score_to_dict(ragas_result)


def generate_eval_report(scores: dict[str, float], output_path: Path) -> None:
    """Write a Markdown report with metric scores and pass/fail thresholds."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RAGAS Evaluation Report",
        "",
        "| Metric | Score | Threshold | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        result = "PASS" if score >= threshold else "FAIL"
        lines.append(f"| {metric} | {score:.3f} | {threshold:.2f} | {result} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
