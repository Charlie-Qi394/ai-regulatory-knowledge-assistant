"""CLI script for asking a question through the LangGraph workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.graph.workflow import run_rag_workflow


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ask a question using the LangGraph RAG workflow.")
    parser.add_argument("question", help="Question to answer from retrieved context.")
    return parser.parse_args()


def main() -> int:
    """Run the graph and print the final answer with sources."""
    args = parse_args()
    try:
        result = run_rag_workflow(args.question)
    except (RuntimeError, ValueError) as exc:
        print(f"Graph RAG error: {exc}", file=sys.stderr)
        return 1

    print("Answer:")
    print(result["answer"])

    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(
                f"[Source {source['source_id']}] {source['filename']} "
                f"| chunk {source['chunk_index']} "
                f"| similarity={source['similarity']:.4f}"
            )
    else:
        print("\nSources: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
