"""Service functions used by FastAPI route handlers."""

from __future__ import annotations

from backend.app.database.connection import get_connection
from backend.app.graph.workflow import run_rag_workflow
from backend.app.rag.generator import RagAnswer


def save_query_history(question: str, answer: str) -> int:
    """Save a question and answer to the query_history table."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO query_history (question, answer)
                VALUES (%s, %s)
                RETURNING id;
                """,
                (question, answer),
            )
            history_id = cursor.fetchone()[0]
        conn.commit()

    return int(history_id)


def ask_and_save(question: str) -> RagAnswer:
    """Generate a RAG answer and save the query history."""
    result = run_rag_workflow(question)
    save_query_history(question=question, answer=result["answer"])
    return result


def get_recent_history(limit: int = 20) -> list[dict[str, object]]:
    """Return recent query history records."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, question, answer, created_at
                FROM query_history
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cursor.fetchall()

    return [
        {
            "id": int(row[0]),
            "question": str(row[1]),
            "answer": str(row[2]),
            "created_at": row[3],
        }
        for row in rows
    ]
