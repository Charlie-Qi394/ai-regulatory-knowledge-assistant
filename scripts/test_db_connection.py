"""Command-line database connection check."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database.connection import test_connection as check_database_connection


def main() -> int:
    """Run the database connection check and print a short status report."""
    result = check_database_connection()
    print("Database connection OK")
    print(f"Database: {result['database']}")
    print(f"User: {result['user']}")
    print(f"pgvector: {result['pgvector']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
