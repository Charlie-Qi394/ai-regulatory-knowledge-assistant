"""AI-assisted Excel review using RAG context.

This module is intentionally separate from deterministic checks. It uses
retrieved regulatory document chunks plus an LLM to screen workbook rows that
may not have explicit Python rules yet.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from backend.app.compliance.excel_checker import (
    ProductValue,
    check_product_values,
    load_product_values,
    normalize_text,
)
from backend.app.rag.generator import build_sources, format_context, get_chat_model
from backend.app.rag.embeddings import get_openai_api_key
from backend.app.rag.retriever import RetrievedChunk, retrieve_relevant_chunks


AI_REVIEW_SYSTEM_PROMPT = """You are an AI-assisted regulatory screening reviewer.
Assess one Excel row using only the provided regulatory context and the normalized workbook context.
Return JSON only.

Allowed status values:
- PASS: the value clearly meets a requirement in the context.
- FAIL: the value clearly does not meet a requirement in the context.
- NEEDS_REVIEW: the context is relevant but the decision requires human review, missing product details, or non-trivial interpretation.
- INSUFFICIENT_CONTEXT: the context does not contain enough information to assess the row.

Rules:
- Do not use outside knowledge.
- Do not invent thresholds or formulas.
- You may use supporting values from the normalized workbook context for calculations.
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

BATCH_AI_REVIEW_SYSTEM_PROMPT = """You are an AI-assisted regulatory screening reviewer.
Assess normalized Excel workbook rows using only the provided regulatory context and workbook context.
Return JSON only.

Allowed status values:
- PASS: the value clearly meets a requirement in the context.
- FAIL: the value clearly does not meet a requirement in the context.
- NEEDS_REVIEW: the context is relevant but the decision requires human review, missing product details, or non-trivial interpretation.
- INSUFFICIENT_CONTEXT: the context does not contain enough information to assess the row.

Rules:
- Do not use outside knowledge.
- Do not invent thresholds or formulas.
- You may use supporting values from other workbook rows for calculations.
- Prioritize the user's focus instructions when provided, but still return one result for every input row.
- If unit conversion or calculation is required but supporting values are missing, use NEEDS_REVIEW or INSUFFICIENT_CONTEXT.
- Cite sources using labels like [Source 1].
- Keep reasoning concise.
- Return one result for every input row.

JSON schema:
{
  "results": [
    {
      "row_index": 1,
      "status": "PASS | FAIL | NEEDS_REVIEW | INSUFFICIENT_CONTEXT",
      "requirement": "short requirement found in context, or empty string",
      "reasoning": "short explanation with citations where available",
      "citations": ["[Source 1]"]
    }
  ]
}
"""

FALLBACK_REVIEW = {
    "status": "NEEDS_REVIEW",
    "requirement": "",
    "reasoning": "The AI review could not produce a structured assessment. Human review is required.",
    "citations": [],
}
MAX_AI_REVIEW_ROWS = 120


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


def build_workbook_question(rows: list[ProductValue], focus_instructions: str = "") -> str:
    """Create one retrieval query covering the normalized workbook."""
    parameters = ", ".join(row.parameter for row in rows[:MAX_AI_REVIEW_ROWS] if row.parameter)
    units = ", ".join(sorted({row.unit for row in rows[:MAX_AI_REVIEW_ROWS] if row.unit}))
    focus = focus_instructions.strip()
    focus_text = f" User focus instructions: {focus}." if focus else ""
    return (
        "Regulatory requirements, nutrient limits, product composition limits, unit conversions, "
        f"and calculation rules for these product values: {parameters}. Units: {units}.{focus_text}"
    )


def focus_category_terms(focus_instructions: str) -> list[str]:
    """Extract category/column terms that should narrow workbook review rows."""
    focus = normalize_text(focus_instructions).replace("/", " / ")
    terms: list[str] = []

    if "spec min" in focus:
        terms.append("spec min")
    if "spec max" in focus:
        terms.append("spec max")
    if "can / nip" in focus or "cannip" in focus.replace(" ", ""):
        terms.append("can / nip")
    if "old fsanz min" in focus:
        terms.append("old fsanz min")
    if "old fsanz max" in focus:
        terms.append("old fsanz max")
    if "review target" in focus:
        terms.append("review target")
    if "can target" in focus:
        terms.append("can target")

    return terms


def row_matches_focus(row: ProductValue, category_terms: list[str]) -> bool:
    """Return whether a normalized row matches category focus terms."""
    if not category_terms:
        return True

    category = normalize_text(row.category).replace("/", " / ")
    compact_category = category.replace(" ", "")

    has_specific_limit = any(term in {"spec min", "spec max", "old fsanz min", "old fsanz max"} for term in category_terms)
    for term in category_terms:
        if term == "can / nip":
            if "can / nip" in category or "cannip" in compact_category:
                if not has_specific_limit:
                    return True
                continue
        elif term in category:
            return True

    return False


def select_rows_for_focus(rows: list[ProductValue], focus_instructions: str) -> list[ProductValue]:
    """Use user focus instructions to narrow rows before AI review."""
    category_terms = focus_category_terms(focus_instructions)
    if not category_terms:
        return rows

    focused_rows = [row for row in rows if row_matches_focus(row, category_terms)]
    return focused_rows or rows


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


def parse_batch_review_json(content: str | None) -> dict[int, dict[str, Any]]:
    """Parse model JSON output for a batch review."""
    if not content:
        return {}

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}

    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        return {}

    parsed: dict[int, dict[str, Any]] = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        try:
            row_index = int(raw_result.get("row_index"))
        except (TypeError, ValueError):
            continue
        parsed[row_index] = normalize_review_payload(raw_result)
    return parsed


def format_workbook_context(rows: list[ProductValue], max_rows: int = 80) -> str:
    """Format normalized workbook rows so the model can use supporting values."""
    lines: list[str] = []
    for index, row in enumerate(rows[:max_rows], start=1):
        lines.append(
            f"Row {index}: parameter={row.parameter}; value={row.value}; "
            f"unit={row.unit}; category={row.category}; notes={row.notes}"
        )
    if len(rows) > max_rows:
        lines.append(f"... {len(rows) - max_rows} additional normalized rows omitted.")
    return "\n".join(lines)


def review_rows_with_context(
    rows: list[ProductValue],
    chunks: list[RetrievedChunk],
    focus_instructions: str = "",
) -> dict[int, dict[str, Any]]:
    """Use one chat-model call to assess multiple workbook rows."""
    if not chunks:
        return {}

    rows_to_review = rows[:MAX_AI_REVIEW_ROWS]
    workbook_context = format_workbook_context(rows_to_review, max_rows=MAX_AI_REVIEW_ROWS)
    regulatory_context = format_context(chunks)
    client = OpenAI(api_key=get_openai_api_key())

    response = client.chat.completions.create(
        model=get_chat_model(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": BATCH_AI_REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Normalized workbook rows to review:\n"
                    f"{workbook_context}\n\n"
                    "User focus instructions:\n"
                    f"{focus_instructions.strip() or 'No specific focus instructions provided.'}\n\n"
                    "Regulatory context:\n"
                    f"{regulatory_context}\n\n"
                    "Review every input row, but prioritize the user's focus instructions. "
                    "Use workbook rows only as supporting product values. "
                    "Use regulatory context as the source of requirements. Return JSON only."
                ),
            },
        ],
    )

    return parse_batch_review_json(response.choices[0].message.content)


def review_row_with_context(
    row: ProductValue,
    chunks: list[RetrievedChunk],
    workbook_context: str = "",
) -> dict[str, Any]:
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
                    "Normalized workbook context:\n"
                    f"{workbook_context}\n\n"
                    "Regulatory context:\n"
                    f"{context}\n\n"
                    "Assess this row against the regulatory context. Use normalized workbook context only "
                    "for supporting product values needed for calculations. Return JSON only."
                ),
            },
        ],
    )

    return parse_review_json(response.choices[0].message.content)


def ai_review_product_values(
    rows: list[ProductValue],
    top_k: int = 4,
    focus_instructions: str = "",
) -> dict[str, Any]:
    """Review workbook rows using RAG retrieval and the chat model."""
    results: list[dict[str, Any]] = []
    focused_rows = select_rows_for_focus(rows, focus_instructions)
    rows_to_review = focused_rows[:MAX_AI_REVIEW_ROWS]
    coded_results = check_product_values(rows_to_review)["results"]
    question = build_workbook_question(rows_to_review, focus_instructions=focus_instructions)
    chunks = retrieve_relevant_chunks(question=question, top_k=max(top_k, 10))
    batch_reviews = review_rows_with_context(
        rows=rows_to_review,
        chunks=chunks,
        focus_instructions=focus_instructions,
    )
    sources = build_sources(chunks)

    for row_index, row in enumerate(rows_to_review, start=1):
        coded_result = coded_results[row_index - 1]
        if coded_result["status"] in {"PASS", "FAIL"}:
            review = {
                "status": coded_result["status"],
                "requirement": coded_result["requirement"],
                "reasoning": (
                    "Coded calculation/check applied before AI screening. "
                    f"{coded_result['notes']}".strip()
                ),
                "citations": [coded_result["source"]] if coded_result["source"] else [],
            }
            row_sources = []
        elif chunks:
            review = batch_reviews.get(
                row_index,
                {
                    "status": "NEEDS_REVIEW",
                    "requirement": "",
                    "reasoning": "The AI review did not return a structured assessment for this row.",
                    "citations": [],
                },
            )
            row_sources = sources
        else:
            review = {
                "status": "INSUFFICIENT_CONTEXT",
                "requirement": "",
                "reasoning": "No relevant document context was retrieved for this workbook.",
                "citations": [],
            }
            row_sources = []

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
                "sources": row_sources,
            }
        )

    for row_index, row in enumerate(rows[MAX_AI_REVIEW_ROWS:], start=MAX_AI_REVIEW_ROWS + 1):
        results.append(
            {
                "row_index": row_index,
                "parameter": row.parameter,
                "input_value": row.value,
                "input_unit": row.unit,
                "category": row.category,
                "status": "NEEDS_REVIEW",
                "requirement": "",
                "reasoning": (
                    f"AI review is limited to the first {MAX_AI_REVIEW_ROWS} normalized rows per upload "
                    "to keep the request responsive."
                ),
                "citations": [],
                "sources": [],
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


def ai_review_excel_workbook(
    workbook_bytes: bytes,
    top_k: int = 4,
    focus_instructions: str = "",
) -> dict[str, Any]:
    """Parse and AI-review an uploaded Excel workbook."""
    rows = load_product_values(workbook_bytes)
    return ai_review_product_values(rows=rows, top_k=top_k, focus_instructions=focus_instructions)
