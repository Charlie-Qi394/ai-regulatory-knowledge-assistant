"""CLI script for testing vector retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.rag.retriever import retrieve_relevant_chunks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrieve relevant document chunks for a sample question."
    )
    parser.add_argument("question", help="Question to search for.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    return parser.parse_args()


def main() -> int:
    """Run retrieval and print readable results."""
    args = parse_args()
    try:
        results = retrieve_relevant_chunks(args.question, top_k=args.top_k)
    except (RuntimeError, ValueError) as exc:
        print(f"Retrieval error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No matching chunks found. Generate embeddings first with scripts/embed_chunks.py.")
        return 0

    print(f"Top {len(results)} result(s):")
    for rank, result in enumerate(results, start=1):
        preview = result["chunk_text"][:240].replace("\n", " ")
        print(f"\n{rank}. {result['filename']} | chunk {result['chunk_index']}")
        print(f"   distance={result['distance']:.4f} similarity={result['similarity']:.4f}")
        if result["page_number"] is not None:
            print(f"   page={result['page_number']}")
        print(f"   preview={preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
