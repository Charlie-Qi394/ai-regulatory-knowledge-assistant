"""OpenAI embedding helpers and chunk embedding storage.

Embeddings turn text into numeric vectors. In this project, each document chunk
will receive a vector so PostgreSQL + pgvector can later find chunks that are
semantically similar to a user's question.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from backend.app.database.connection import get_connection


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_openai_api_key() -> str:
    """Return the OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add your key to .env before generating embeddings."
        )
    return api_key


def get_embedding_model() -> str:
    """Return the configured embedding model name."""
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def create_embedding(text: str) -> list[float]:
    """Create one embedding vector for a piece of text.

    Args:
        text: Input text to embed.

    Returns:
        A list of floating-point numbers suitable for the pgvector column.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Cannot create an embedding for empty text.")

    client = OpenAI(api_key=get_openai_api_key())
    response = client.embeddings.create(
        model=get_embedding_model(),
        input=cleaned_text,
    )
    return response.data[0].embedding


def get_unembedded_chunks(limit: int | None = None) -> list[tuple[int, str]]:
    """Return document chunks that do not yet have embeddings."""
    query = """
        SELECT id, chunk_text
        FROM document_chunks
        WHERE embedding IS NULL
        ORDER BY id
    """
    params: tuple[int, ...] = ()
    if limit is not None:
        query += " LIMIT %s"
        params = (limit,)

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return [(int(row[0]), str(row[1])) for row in cursor.fetchall()]


def format_vector_literal(embedding: list[float]) -> str:
    """Format a Python list as a pgvector-compatible vector literal."""
    return "[" + ",".join(str(value) for value in embedding) + "]"


def store_chunk_embedding(chunk_id: int, embedding: list[float]) -> None:
    """Store an embedding vector for one document chunk."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE document_chunks
                SET embedding = %s::vector
                WHERE id = %s;
                """,
                (format_vector_literal(embedding), chunk_id),
            )
        conn.commit()


def embed_unembedded_chunks(limit: int | None = None) -> dict[str, int]:
    """Generate and store embeddings for chunks with no embedding yet."""
    get_openai_api_key()
    chunks = get_unembedded_chunks(limit=limit)
    print(f"Found {len(chunks)} chunk(s) without embeddings.")

    embedded = 0
    for chunk_id, chunk_text in chunks:
        embedding = create_embedding(chunk_text)
        store_chunk_embedding(chunk_id, embedding)
        embedded += 1
        print(f"Stored embedding for chunk id={chunk_id}.")

    print(f"Embedding generation complete. Embedded: {embedded}.")
    return {"found": len(chunks), "embedded": embedded}
