"""State definition for the LangGraph RAG workflow."""

from __future__ import annotations

from typing import TypedDict

from backend.app.rag.generator import Source
from backend.app.rag.retriever import RetrievedChunk


class VerificationResult(TypedDict):
    """Result of the lightweight answer verification step."""

    passed: bool
    reason: str


class RagGraphState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    question: str
    top_k: int
    retrieved_chunks: list[RetrievedChunk]
    context_is_sufficient: bool
    draft_answer: str
    verification_result: VerificationResult
    final_answer: str
    sources: list[Source]
