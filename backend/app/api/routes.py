"""FastAPI routes for the AI Regulatory Knowledge Assistant."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from psycopg2 import Error as PsycopgError

from backend.app.api.schemas import AskRequest, AskResponse, HistoryResponse
from backend.app.api.service import ask_and_save, get_recent_history


router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok", "service": "ai-regulatory-knowledge-assistant"}


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question using the RAG pipeline."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = ask_and_save(question)
    except PsycopgError as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        question=question,
        answer=result["answer"],
        sources=result["sources"],
    )


@router.get("/history", response_model=HistoryResponse)
def history() -> HistoryResponse:
    """Return recent query history."""
    try:
        return HistoryResponse(history=get_recent_history())
    except PsycopgError as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
