"""Optional RAGAS evaluation for the RAG workflow.

This module is intentionally separate from the simple CSV evaluator. RAGAS is
useful for deeper inspection, but it adds heavier dependencies and uses LLM
judges, so it should not be required for running the application.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.app.evaluation.evaluator import (
    DEFAULT_TEST_QUESTIONS_PATH,
    PROJECT_ROOT,
)
from backend.app.graph.workflow import run_rag_workflow_state


DEFAULT_RAGAS_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "ragas_results.csv"


def parse_expected_contexts(raw_contexts: str | None) -> list[str]:
    """Parse optional expected contexts from a CSV cell.

    Use `||` as a delimiter so commas and sentence punctuation can remain in
    the context text.
    """
    if not raw_contexts:
        return []
    return [context.strip() for context in raw_contexts.split("||") if context.strip()]


def build_ragas_rows(questions_path: Path, top_k: int = 5) -> list[dict[str, Any]]:
    """Run the RAG workflow and build rows in a RAGAS-compatible shape."""
    rows: list[dict[str, Any]] = []

    with questions_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            question = row["question"]
            result = run_rag_workflow_state(question=question, top_k=top_k)
            retrieved_chunks = result.get("retrieved_chunks", [])
            contexts = [chunk["chunk_text"] for chunk in retrieved_chunks]
            reference = row.get("expected_answer", "").strip()
            expected_contexts = parse_expected_contexts(row.get("expected_contexts"))

            ragas_row: dict[str, Any] = {
                "question": question,
                "answer": result["final_answer"],
                "contexts": contexts,
                "ground_truth": reference,
                "user_input": question,
                "response": result["final_answer"],
                "retrieved_contexts": contexts,
                "reference": reference,
            }
            if expected_contexts:
                ragas_row["reference_contexts"] = expected_contexts
            rows.append(ragas_row)

    return rows


def get_ragas_metrics(include_context_recall: bool) -> list[Any]:
    """Return RAGAS metrics while tolerating common version differences."""
    try:
        from ragas.metrics import answer_relevancy
    except ImportError:
        from ragas.metrics import answer_relevance as answer_relevancy

    from ragas.metrics import context_precision, faithfulness

    metrics: list[Any] = [faithfulness, answer_relevancy, context_precision]

    if include_context_recall:
        from ragas.metrics import context_recall

        metrics.append(context_recall)

    return metrics


def save_ragas_results(result: Any, results_path: Path) -> None:
    """Save RAGAS scores to CSV.

    RAGAS result objects differ by version. Newer versions expose `to_pandas`;
    older versions can usually be converted through `dict(result)`.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(result, "to_pandas"):
        result.to_pandas().to_csv(results_path, index=False)
        return

    scores = dict(result)
    with results_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(scores.keys()))
        writer.writeheader()
        writer.writerow(scores)


def run_ragas_evaluation(
    questions_path: Path = DEFAULT_TEST_QUESTIONS_PATH,
    results_path: Path = DEFAULT_RAGAS_RESULTS_PATH,
    top_k: int = 5,
    include_context_recall: bool = False,
) -> dict[str, Any]:
    """Run optional RAGAS evaluation and save metric results.

    Raises:
        RuntimeError: If optional RAGAS dependencies are not installed.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation dependencies are not installed. "
            "Install them with: pip install -r requirements-ragas.txt"
        ) from exc

    rows = build_ragas_rows(questions_path=questions_path, top_k=top_k)
    dataset = Dataset.from_list(rows)
    metrics = get_ragas_metrics(include_context_recall=include_context_recall)

    result = evaluate(dataset, metrics=metrics)
    save_ragas_results(result=result, results_path=results_path)

    return {
        "rows": len(rows),
        "metrics": [getattr(metric, "name", metric.__class__.__name__) for metric in metrics],
        "results_path": str(results_path),
    }
