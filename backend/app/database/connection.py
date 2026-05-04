"""PostgreSQL connection helpers for the application.

This module keeps the first database layer intentionally small. Later stages can
add pooling or repository classes when the application needs them.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import connection as PsycopgConnection


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def get_database_url() -> str:
    """Return DATABASE_URL from the environment.

    Raises:
        RuntimeError: If DATABASE_URL is not configured.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and configure the database URL."
        )
    return database_url


def get_connection() -> PsycopgConnection:
    """Create and return a psycopg2 PostgreSQL connection.

    The caller is responsible for closing the returned connection.
    """
    return psycopg2.connect(get_database_url())


def test_connection() -> dict[str, str]:
    """Check database connectivity and pgvector availability."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user;")
            database_name, user_name = cursor.fetchone()

            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            vector_extension = cursor.fetchone()

    return {
        "database": str(database_name),
        "user": str(user_name),
        "pgvector": "enabled" if vector_extension else "missing",
    }
