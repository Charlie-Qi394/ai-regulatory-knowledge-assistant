"""CLI script for running the simple RAG evaluation."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.evaluation.evaluator import run_evaluation


def main() -> int:
    """Run evaluation and write a CSV results file."""
    parser = argparse.ArgumentParser(description="Run the simple RAG evaluation.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path to the evaluation questions CSV.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path where evaluation results should be written.",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.questions is not None:
        kwargs["questions_path"] = args.questions
    if args.results is not None:
        kwargs["results_path"] = args.results

    summary = run_evaluation(**kwargs)
    print(
        "Summary: "
        f"total={summary['total']} "
        f"passed={summary['passed']} "
        f"failed={summary['failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
