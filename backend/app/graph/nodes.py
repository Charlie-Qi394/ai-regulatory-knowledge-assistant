"""Node functions for the simple LangGraph RAG workflow."""

from __future__ import annotations

import re

from backend.app.graph.state import RagGraphState, VerificationResult
from backend.app.rag.generator import (
    INSUFFICIENT_CONTEXT_RESPONSE,
    build_sources,
    generate_answer_from_context,
)
from backend.app.rag.retriever import retrieve_relevant_chunks


MINIMUM_SIMILARITY = 0.2


def retrieve_context(state: RagGraphState) -> RagGraphState:
    """Retrieve document chunks that are relevant to the question."""
    question = state["question"]
    top_k = state.get("top_k", 5)
    chunks = retrieve_relevant_chunks(question=question, top_k=top_k)
    return {
        "retrieved_chunks": chunks,
        "sources": build_sources(chunks),
    }


def check_context_sufficiency(state: RagGraphState) -> RagGraphState:
    """Decide whether retrieved context is strong enough to answer."""
    chunks = state.get("retrieved_chunks", [])
    has_context = len(chunks) > 0
    top_similarity = chunks[0]["similarity"] if has_context else 0.0

    return {
        "context_is_sufficient": has_context and top_similarity >= MINIMUM_SIMILARITY,
    }


def generate_answer(state: RagGraphState) -> RagGraphState:
    """Generate a draft answer from the retrieved chunks."""
    result = generate_answer_from_context(
        question=state["question"],
        chunks=state.get("retrieved_chunks", []),
    )
    return {
        "draft_answer": result["answer"],
        "sources": result["sources"],
    }


def verify_answer(state: RagGraphState) -> RagGraphState:
    """Run a lightweight verification check before returning the answer.

    This is deliberately rule-based. It checks that an answer exists, is not the
    fallback message, and includes at least one source citation.
    """
    draft_answer = state.get("draft_answer", "").strip()
    if not draft_answer:
        result: VerificationResult = {"passed": False, "reason": "Draft answer is empty."}
    elif draft_answer == INSUFFICIENT_CONTEXT_RESPONSE:
        result = {"passed": False, "reason": "Draft answer used the fallback response."}
    elif not re.search(r"\[Source \d+\]", draft_answer):
        result = {"passed": False, "reason": "Draft answer does not include a source citation."}
    else:
        result = {"passed": True, "reason": "Draft answer includes a source citation."}

    return {"verification_result": result}


def return_final_answer(state: RagGraphState) -> RagGraphState:
    """Return either the verified draft answer or a cautious fallback."""
    context_is_sufficient = state.get("context_is_sufficient", False)
    verification_result = state.get(
        "verification_result",
        {"passed": False, "reason": "Answer was not verified."},
    )

    if not context_is_sufficient or not verification_result["passed"]:
        return {"final_answer": INSUFFICIENT_CONTEXT_RESPONSE}

    return {"final_answer": state.get("draft_answer", INSUFFICIENT_CONTEXT_RESPONSE)}


def route_after_context_check(state: RagGraphState) -> str:
    """Route to answer generation only when context is sufficient."""
    return "generate_answer" if state.get("context_is_sufficient", False) else "return_final_answer"
