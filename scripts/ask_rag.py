"""CLI script for asking a basic RAG question."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.rag.generator import answer_question


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ask a question using basic RAG.")
    parser.add_argument("question", help="Question to answer from retrieved context.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    return parser.parse_args()


def main() -> int:
    """Run retrieval plus answer generation."""
    args = parse_args()
    try:
        result = answer_question(args.question, top_k=args.top_k)
    except (RuntimeError, ValueError) as exc:
        print(f"RAG error: {exc}", file=sys.stderr)
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
            print(f"  excerpt={source['excerpt']}")
    else:
        print("\nSources: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
