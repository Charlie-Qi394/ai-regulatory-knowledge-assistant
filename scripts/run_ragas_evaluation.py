"""CLI script for optional RAGAS evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.evaluation.ragas_evaluator import run_ragas_evaluation


def main() -> int:
    """Run optional RAGAS evaluation and write metric results."""
    parser = argparse.ArgumentParser(description="Run optional RAGAS evaluation.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "test_questions.csv",
        help="Path to the evaluation questions CSV.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "ragas_results.csv",
        help="Path where RAGAS results should be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for each question.",
    )
    parser.add_argument(
        "--include-context-recall",
        action="store_true",
        help="Include context recall when reference answers or expected contexts are available.",
    )
    args = parser.parse_args()

    try:
        summary = run_ragas_evaluation(
            questions_path=args.questions,
            results_path=args.results,
            top_k=args.top_k,
            include_context_recall=args.include_context_recall,
        )
    except RuntimeError as exc:
        print(exc)
        return 1

    print(
        "RAGAS evaluation complete. "
        f"rows={summary['rows']} "
        f"metrics={', '.join(summary['metrics'])} "
        f"results={summary['results_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
