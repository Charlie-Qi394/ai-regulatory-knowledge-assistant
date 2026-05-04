"""Simple CSV-based evaluation for the RAG workflow.

This is intentionally lightweight. It checks whether the generated answer
contains expected keywords and records the retrieved sources for inspection.
"""

from __future__ import annotations

import csv
from pathlib import Path

from backend.app.graph.workflow import run_rag_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEST_QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "test_questions.csv"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.csv"


def parse_expected_keywords(raw_keywords: str) -> list[str]:
    """Parse pipe-separated expected keyword groups from the CSV."""
    return [keyword.strip().lower() for keyword in raw_keywords.split("|") if keyword.strip()]


def answer_contains_expected_info(answer: str, expected_keywords: list[str]) -> bool:
    """Return True when all expected keyword groups appear in the answer."""
    normalized_answer = answer.lower()
    return all(keyword in normalized_answer for keyword in expected_keywords)


def format_sources(sources: list[dict[str, object]]) -> str:
    """Create a compact source summary for CSV output."""
    formatted_sources: list[str] = []
    for source in sources:
        formatted_sources.append(
            f"[Source {source['source_id']}] {source['filename']} "
            f"chunk={source['chunk_index']} similarity={float(source['similarity']):.4f}"
        )
    return "; ".join(formatted_sources)


def run_evaluation(
    questions_path: Path = DEFAULT_TEST_QUESTIONS_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
) -> dict[str, int]:
    """Run the LangGraph RAG workflow against CSV test questions."""
    results_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    passed = 0

    with questions_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    with results_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = [
            "question",
            "expected_answer",
            "generated_answer",
            "retrieved_sources",
            "answer_contains_expected_info",
            "notes",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            total += 1
            question = row["question"]
            expected_answer = row["expected_answer"]
            expected_keywords = parse_expected_keywords(row.get("expected_keywords", ""))
            notes = ""
            generated_answer = ""
            retrieved_sources = ""
            contains_expected_info = False

            try:
                result = run_rag_workflow(question)
                generated_answer = result["answer"]
                retrieved_sources = format_sources(result["sources"])
                contains_expected_info = answer_contains_expected_info(
                    generated_answer,
                    expected_keywords,
                )
                if contains_expected_info:
                    passed += 1
                else:
                    notes = "Generated answer did not contain all expected keyword groups."
            except Exception as exc:  # noqa: BLE001 - evaluation should continue across rows.
                notes = f"Evaluation error: {exc}"

            writer.writerow(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": generated_answer,
                    "retrieved_sources": retrieved_sources,
                    "answer_contains_expected_info": str(contains_expected_info),
                    "notes": notes,
                }
            )

            print(f"Evaluated {total}/{len(rows)}: {question}")

    print(f"Evaluation complete. Passed: {passed}/{total}. Results: {results_path}")
    return {"total": total, "passed": passed, "failed": total - passed}
