"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for asking the RAG assistant a question."""

    question: str = Field(..., min_length=1, description="Question to answer.")


class SourceResponse(BaseModel):
    """Citation source returned by the RAG assistant."""

    source_id: int
    filename: str
    chunk_index: int
    page_number: Optional[int]
    distance: float
    similarity: float
    excerpt: str


class AskResponse(BaseModel):
    """Response returned by POST /ask."""

    question: str
    answer: str
    sources: list[SourceResponse]


class HistoryItem(BaseModel):
    """One query history record."""

    id: int
    question: str
    answer: str
    created_at: datetime


class HistoryResponse(BaseModel):
    """Response returned by GET /history."""

    history: list[HistoryItem]
