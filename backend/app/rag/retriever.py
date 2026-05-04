"""Vector retrieval for document chunks.

Retrieval is the "R" in RAG. It finds the most relevant stored chunks for a
question before a later stage sends those chunks to an answer-generation model.
"""

from __future__ import annotations

from typing import TypedDict

from backend.app.database.connection import get_connection
from backend.app.rag.embeddings import create_embedding, format_vector_literal


class RetrievedChunk(TypedDict):
    """A chunk returned from vector search."""

    chunk_text: str
    filename: str
    chunk_index: int
    page_number: int | None
    distance: float
    similarity: float


def retrieve_relevant_chunks(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Retrieve the most relevant embedded chunks for a user question.

    Args:
        question: User question to search for.
        top_k: Maximum number of chunks to return.

    Returns:
        Ranked chunks with source metadata and distance/similarity scores.
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    question_embedding = create_embedding(cleaned_question)
    question_vector = format_vector_literal(question_embedding)

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    dc.chunk_text,
                    d.filename,
                    dc.chunk_index,
                    dc.page_number,
                    dc.embedding <=> %s::vector AS distance
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.embedding IS NOT NULL
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s;
                """,
                (question_vector, question_vector, top_k),
            )
            rows = cursor.fetchall()

    results: list[RetrievedChunk] = []
    for chunk_text, filename, chunk_index, page_number, distance in rows:
        distance_value = float(distance)
        results.append(
            {
                "chunk_text": str(chunk_text),
                "filename": str(filename),
                "chunk_index": int(chunk_index),
                "page_number": int(page_number) if page_number is not None else None,
                "distance": distance_value,
                "similarity": 1.0 - distance_value,
            }
        )

    return results
