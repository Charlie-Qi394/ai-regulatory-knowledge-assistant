"""LangGraph workflow for the simple RAG assistant.

The graph is intentionally small: retrieve context, check sufficiency, generate
an answer, verify citation grounding, and return the final answer.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.app.graph.nodes import (
    check_context_sufficiency,
    generate_answer,
    retrieve_context,
    return_final_answer,
    route_after_context_check,
    verify_answer,
)
from backend.app.graph.state import RagGraphState
from backend.app.rag.generator import RagAnswer


RAG_GRAPH = None


def build_rag_graph():
    """Build and compile the LangGraph RAG workflow."""
    graph = StateGraph(RagGraphState)

    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("check_context_sufficiency", check_context_sufficiency)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_answer", verify_answer)
    graph.add_node("return_final_answer", return_final_answer)

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "check_context_sufficiency")
    graph.add_conditional_edges(
        "check_context_sufficiency",
        route_after_context_check,
        {
            "generate_answer": "generate_answer",
            "return_final_answer": "return_final_answer",
        },
    )
    graph.add_edge("generate_answer", "verify_answer")
    graph.add_edge("verify_answer", "return_final_answer")
    graph.add_edge("return_final_answer", END)

    return graph.compile()


def get_rag_graph():
    """Return a compiled LangGraph workflow."""
    global RAG_GRAPH
    if RAG_GRAPH is None:
        RAG_GRAPH = build_rag_graph()
    return RAG_GRAPH


def run_rag_workflow_state(question: str, top_k: int = 5) -> RagGraphState:
    """Run the LangGraph RAG workflow and return the full final state."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    app = get_rag_graph()
    return app.invoke({"question": question, "top_k": top_k})


def run_rag_workflow(question: str, top_k: int = 5) -> RagAnswer:
    """Run the LangGraph RAG workflow and return API-compatible output."""
    final_state = run_rag_workflow_state(question=question, top_k=top_k)

    return {
        "answer": final_state["final_answer"],
        "sources": final_state.get("sources", []),
    }
