"""FastAPI routes for the AI Regulatory Knowledge Assistant."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from psycopg2 import Error as PsycopgError

from backend.app.api.schemas import (
    AiComplianceCheckResponse,
    AskRequest,
    AskResponse,
    ComplianceCheckResponse,
    HistoryResponse,
)
from backend.app.api.service import ask_and_save, get_recent_history
from backend.app.compliance.ai_excel_reviewer import ai_review_excel_workbook
from backend.app.compliance.excel_checker import check_excel_workbook


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


@router.post("/check-excel", response_model=ComplianceCheckResponse)
async def check_excel(file: UploadFile = File(...)) -> ComplianceCheckResponse:
    """Check uploaded Excel product values against selected deterministic rules."""
    filename = file.filename or "uploaded.xlsx"
    if not filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Please upload a .xlsx file.")

    try:
        content = await file.read()
        result = check_excel_workbook(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ComplianceCheckResponse(
        filename=filename,
        summary=result["summary"],
        results=result["results"],
    )


@router.post("/review-excel-ai", response_model=AiComplianceCheckResponse)
async def review_excel_ai(file: UploadFile = File(...)) -> AiComplianceCheckResponse:
    """AI-review uploaded Excel product values against retrieved regulatory context."""
    filename = file.filename or "uploaded.xlsx"
    if not filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Please upload a .xlsx file.")

    try:
        content = await file.read()
        result = ai_review_excel_workbook(content)
    except PsycopgError as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AiComplianceCheckResponse(
        filename=filename,
        summary=result["summary"],
        results=result["results"],
    )
