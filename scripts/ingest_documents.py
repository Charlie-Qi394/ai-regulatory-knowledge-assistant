"""CLI script for ingesting supported sample documents."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ingestion.service import ingest_documents


def main() -> int:
    """Run document metadata ingestion for `data/sample_docs/`."""
    result = ingest_documents()
    if result["found"] == 0:
        print("No supported files found. Add .txt, .pdf, or .docx files to data/sample_docs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
