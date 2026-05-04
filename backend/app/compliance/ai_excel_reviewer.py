"""AI-assisted Excel review using RAG context.

This module is intentionally separate from deterministic checks. It uses
retrieved regulatory document chunks plus an LLM to screen workbook rows that
may not have explicit Python rules yet.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from backend.app.compliance.excel_checker import ProductValue, load_product_values
from backend.app.rag.generator import build_sources, format_context, get_chat_model
from backend.app.rag.embeddings import get_openai_api_key
from backend.app.rag.retriever import RetrievedChunk, retrieve_relevant_chunks


AI_REVIEW_SYSTEM_PROMPT = """You are an AI-assisted regulatory screening reviewer.
Assess one Excel row using only the provided regulatory context.
Return JSON only.

Allowed status values:
- PASS: the value clearly meets a requirement in the context.
- FAIL: the value clearly does not meet a requirement in the context.
- NEEDS_REVIEW: the context is relevant but the decision requires human review, missing product details, or non-trivial interpretation.
- INSUFFICIENT_CONTEXT: the context does not contain enough information to assess the row.

Rules:
- Do not use outside knowledge.
- Do not invent thresholds or formulas.
- If unit conversion or calculation is required but supporting values are missing, use NEEDS_REVIEW or INSUFFICIENT_CONTEXT.
- Cite sources using labels like [Source 1].
- Keep reasoning concise.

JSON schema:
{
  "status": "PASS | FAIL | NEEDS_REVIEW | INSUFFICIENT_CONTEXT",
  "requirement": "short requirement found in context, or empty string",
  "reasoning": "short explanation with citations where available",
  "citations": ["[Source 1]"]
}
"""

FALLBACK_REVIEW = {
    "status": "NEEDS_REVIEW",
    "requirement": "",
    "reasoning": "The AI review could not produce a structured assessment. Human review is required.",
    "citations": [],
}


def build_row_question(row: ProductValue) -> str:
    """Create a retrieval question for one workbook row."""
    parts = [
        "Regulatory requirement or limit for",
        row.parameter,
        f"value {row.value}" if row.value is not None else "missing value",
        row.unit,
    ]
    if row.category:
        parts.append(f"category {row.category}")
    if row.notes:
        parts.append(f"notes {row.notes}")
    return " ".join(parts)


def normalize_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize model JSON into the API response shape."""
    status = str(payload.get("status", "NEEDS_REVIEW")).strip().upper()
    if status not in {"PASS", "FAIL", "NEEDS_REVIEW", "INSUFFICIENT_CONTEXT"}:
        status = "NEEDS_REVIEW"

    citations = payload.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    return {
        "status": status,
        "requirement": str(payload.get("requirement", "") or "").strip(),
        "reasoning": str(payload.get("reasoning", "") or "").strip(),
        "citations": [str(citation) for citation in citations],
    }


def parse_review_json(content: str | None) -> dict[str, Any]:
    """Parse and normalize model JSON output."""
    if not content:
        return FALLBACK_REVIEW.copy()

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return FALLBACK_REVIEW.copy()

    if not isinstance(payload, dict):
        return FALLBACK_REVIEW.copy()

    return normalize_review_payload(payload)


def review_row_with_context(row: ProductValue, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    """Use the chat model to assess one workbook row against retrieved context."""
    if not chunks:
        return {
            "status": "INSUFFICIENT_CONTEXT",
            "requirement": "",
            "reasoning": "No relevant document context was retrieved for this row.",
            "citations": [],
        }

    context = format_context(chunks)
    client = OpenAI(api_key=get_openai_api_key())
    response = client.chat.completions.create(
        model=get_chat_model(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": AI_REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Excel row:\n"
                    f"parameter: {row.parameter}\n"
                    f"value: {row.value}\n"
                    f"unit: {row.unit}\n"
                    f"category: {row.category}\n"
                    f"notes: {row.notes}\n\n"
                    "Regulatory context:\n"
                    f"{context}\n\n"
                    "Assess this row against the context and return JSON only."
                ),
            },
        ],
    )

    return parse_review_json(response.choices[0].message.content)


def ai_review_product_values(rows: list[ProductValue], top_k: int = 4) -> dict[str, Any]:
    """Review workbook rows using RAG retrieval and the chat model."""
    results: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        question = build_row_question(row)
        chunks = retrieve_relevant_chunks(question=question, top_k=top_k)
        review = review_row_with_context(row=row, chunks=chunks)
        sources = build_sources(chunks)

        results.append(
            {
                "row_index": row_index,
                "parameter": row.parameter,
                "input_value": row.value,
                "input_unit": row.unit,
                "category": row.category,
                "status": review["status"],
                "requirement": review["requirement"],
                "reasoning": review["reasoning"],
                "citations": review["citations"],
                "sources": sources,
            }
        )

    summary = {
        "total": len(results),
        "passed": sum(1 for row in results if row["status"] == "PASS"),
        "failed": sum(1 for row in results if row["status"] == "FAIL"),
        "needs_review": sum(1 for row in results if row["status"] == "NEEDS_REVIEW"),
        "insufficient_context": sum(1 for row in results if row["status"] == "INSUFFICIENT_CONTEXT"),
    }
    return {"summary": summary, "results": results}


def ai_review_excel_workbook(workbook_bytes: bytes, top_k: int = 4) -> dict[str, Any]:
    """Parse and AI-review an uploaded Excel workbook."""
    rows = load_product_values(workbook_bytes)
    return ai_review_product_values(rows=rows, top_k=top_k)
