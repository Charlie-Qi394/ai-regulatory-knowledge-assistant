"""Tests for the simple LangGraph RAG workflow nodes."""

from backend.app.graph.nodes import (
    check_context_sufficiency,
    retrieve_context,
    return_final_answer,
    route_after_context_check,
    verify_answer,
)
from backend.app.rag.generator import INSUFFICIENT_CONTEXT_RESPONSE


def test_insufficient_context_routes_to_final_answer() -> None:
    state = {"question": "What is the policy?", "retrieved_chunks": []}

    updated_state = check_context_sufficiency(state)
    route = route_after_context_check(updated_state)

    assert updated_state["context_is_sufficient"] is False
    assert route == "return_final_answer"


def test_sufficient_context_routes_to_generation() -> None:
    state = {
        "question": "What is the policy?",
        "retrieved_chunks": [
            {
                "chunk_text": "Relevant context",
                "filename": "sample.txt",
                "chunk_index": 0,
                "page_number": None,
                "distance": 0.1,
                "similarity": 0.9,
            }
        ],
    }

    updated_state = check_context_sufficiency(state)
    route = route_after_context_check(updated_state)

    assert updated_state["context_is_sufficient"] is True
    assert route == "generate_answer"


def test_retrieve_context_uses_top_k_from_state(monkeypatch) -> None:
    observed = {}

    def fake_retrieve_relevant_chunks(question: str, top_k: int):
        observed["question"] = question
        observed["top_k"] = top_k
        return []

    monkeypatch.setattr(
        "backend.app.graph.nodes.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks,
    )

    retrieve_context({"question": "What is regulated?", "top_k": 3})

    assert observed == {"question": "What is regulated?", "top_k": 3}


def test_verification_fails_without_citation() -> None:
    state = {"draft_answer": "This answer has no citation."}

    updated_state = verify_answer(state)

    assert updated_state["verification_result"]["passed"] is False
    assert "source citation" in updated_state["verification_result"]["reason"]


def test_verification_passes_with_source_citation() -> None:
    state = {"draft_answer": "The label should be reviewed before approval [Source 1]."}

    updated_state = verify_answer(state)

    assert updated_state["verification_result"]["passed"] is True


def test_return_final_answer_uses_fallback_when_verification_fails() -> None:
    state = {
        "context_is_sufficient": True,
        "draft_answer": "Unsupported answer.",
        "verification_result": {"passed": False, "reason": "No citation."},
    }

    updated_state = return_final_answer(state)

    assert updated_state["final_answer"] == INSUFFICIENT_CONTEXT_RESPONSE


def test_run_rag_workflow_with_mocked_nodes(monkeypatch) -> None:
    from backend.app.graph import workflow

    def fake_retrieve_context(state):
        return {
            "retrieved_chunks": [
                {
                    "chunk_text": "Label artwork should be checked before approval.",
                    "filename": "sample.txt",
                    "chunk_index": 0,
                    "page_number": None,
                    "distance": 0.1,
                    "similarity": 0.9,
                }
            ],
            "sources": [
                {
                    "source_id": 1,
                    "filename": "sample.txt",
                    "chunk_index": 0,
                    "page_number": None,
                    "distance": 0.1,
                    "similarity": 0.9,
                    "excerpt": "Label artwork should be checked before approval.",
                }
            ],
        }

    def fake_generate_answer(state):
        return {"draft_answer": "Check label artwork before approval [Source 1]."}

    monkeypatch.setattr(workflow, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(workflow, "generate_answer", fake_generate_answer)

    result = workflow.run_rag_workflow("What should be checked?")

    assert result["answer"] == "Check label artwork before approval [Source 1]."
    assert result["sources"][0]["filename"] == "sample.txt"
