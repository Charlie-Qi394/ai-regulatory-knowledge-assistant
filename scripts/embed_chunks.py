"""CLI script for generating OpenAI embeddings for document chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.rag.embeddings import embed_unembedded_chunks


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for document chunks where embedding is NULL."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of chunks to embed.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate embeddings for unembedded chunks."""
    args = parse_args()
    try:
        embed_unembedded_chunks(limit=args.limit)
    except RuntimeError as exc:
        print(f"Embedding setup error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
