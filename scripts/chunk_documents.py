"""CLI script for chunking ingested sample documents."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ingestion.service import chunk_ingested_documents


def main() -> int:
    """Create chunks for documents already saved in the database."""
    result = chunk_ingested_documents()
    if result["documents"] == 0:
        print("No documents found. Run scripts/ingest_documents.py first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
