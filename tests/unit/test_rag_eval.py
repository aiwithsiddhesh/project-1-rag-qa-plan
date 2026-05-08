from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest

from eval.rag_eval import (
    generate_eval_report,
    load_eval_dataset,
    run_ragas_evaluation,
)


def test_load_eval_dataset_validates_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"question": "What?"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="ground_truth"):
        load_eval_dataset(path)


def test_load_eval_dataset_returns_records(tmp_path: Path) -> None:
    records = [
        {
            "question": "What is zero trust?",
            "ground_truth": "A security model.",
            "contexts": ["Zero trust removes implicit trust."],
        }
    ]
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    assert load_eval_dataset(path) == records


def test_generate_eval_report_writes_threshold_table(tmp_path: Path) -> None:
    output_path = tmp_path / "report.md"

    generate_eval_report(
        {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        },
        output_path,
    )

    content = output_path.read_text(encoding="utf-8")
    assert "# RAGAS Evaluation Report" in content
    assert "| faithfulness | 0.900 | 0.80 | PASS |" in content


def test_run_ragas_evaluation_calls_pipeline_and_ragas(monkeypatch) -> None:
    captured_rows = {}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            captured_rows["rows"] = rows
            return rows

    class FakeResult:
        scores = {
            "faithfulness": 0.91,
            "answer_relevancy": 0.82,
            "context_precision": 0.73,
            "context_recall": 0.64,
        }

        def to_pandas(self):
            class FakeMean:
                def to_dict(self):
                    return {
                        "faithfulness": 0.91,
                        "answer_relevancy": 0.82,
                        "context_precision": 0.73,
                        "context_recall": 0.64,
                    }

            class FakeFrame:
                def mean(self, numeric_only=True):
                    assert numeric_only is True
                    return FakeMean()

            return FakeFrame()

    def fake_evaluate(dataset, metrics):
        assert dataset == captured_rows["rows"]
        assert len(metrics) == 4
        return FakeResult()

    datasets_module = types.SimpleNamespace(Dataset=FakeDataset)
    ragas_module = types.SimpleNamespace(evaluate=fake_evaluate)
    metrics_module = types.SimpleNamespace(
        faithfulness=object(),
        answer_relevancy=object(),
        context_precision=object(),
        context_recall=object(),
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)

    class FakePipeline:
        def query(self, question: str) -> dict:
            return {
                "answer": f"answer for {question}",
                "contexts": [f"retrieved context for {question}"],
            }

    scores = run_ragas_evaluation(
        FakePipeline(),
        [
            {
                "question": "What is AI RMF?",
                "ground_truth": "A framework.",
                "contexts": ["AI RMF manages AI risk."],
            }
        ],
    )

    assert scores["faithfulness"] == 0.91
    assert captured_rows["rows"][0]["answer"] == "answer for What is AI RMF?"
    assert captured_rows["rows"][0]["contexts"] == [
        "retrieved context for What is AI RMF?"
    ]


def test_run_ragas_evaluation_requires_pipeline_contexts() -> None:
    class FakePipeline:
        def query(self, question: str) -> dict:
            return {"answer": f"answer for {question}"}

    with pytest.raises(ValueError, match="contexts"):
        run_ragas_evaluation(
            FakePipeline(),
            [
                {
                    "question": "What is AI RMF?",
                    "ground_truth": "A framework.",
                    "contexts": ["Reference context only."],
                }
            ],
        )
