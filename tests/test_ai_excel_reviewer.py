"""Tests for AI-assisted Excel review plumbing."""

from backend.app.compliance.ai_excel_reviewer import (
    ai_review_product_values,
    build_row_question,
    format_workbook_context,
    parse_review_json,
)
from backend.app.compliance.excel_checker import ProductValue


def test_build_row_question_includes_key_row_fields() -> None:
    row = ProductValue("Vitamin A", 80, "ug RE/100 kJ", "infant formula", "label target")

    question = build_row_question(row)

    assert "Vitamin A" in question
    assert "80" in question
    assert "ug RE/100 kJ" in question
    assert "infant formula" in question
    assert "label target" in question


def test_parse_review_json_normalizes_valid_payload() -> None:
    result = parse_review_json(
        '{"status": "pass", "requirement": "Not more than 12 mg/100 kJ", '
        '"reasoning": "The value meets the limit [Source 1].", "citations": ["[Source 1]"]}'
    )

    assert result["status"] == "PASS"
    assert result["requirement"] == "Not more than 12 mg/100 kJ"
    assert result["citations"] == ["[Source 1]"]


def test_parse_review_json_falls_back_for_invalid_payload() -> None:
    result = parse_review_json("not json")

    assert result["status"] == "NEEDS_REVIEW"
    assert "structured assessment" in result["reasoning"]


def test_format_workbook_context_includes_supporting_rows() -> None:
    rows = [
        ProductValue("Energy", 2720, "kJ/L", "Nutrition", ""),
        ProductValue("Protein", 15, "g/L", "Nutrition", "Inferred row"),
    ]

    context = format_workbook_context(rows)

    assert "parameter=Energy" in context
    assert "value=2720" in context
    assert "parameter=Protein" in context


def test_ai_review_product_values_summarizes_mocked_reviews(monkeypatch) -> None:
    rows = [
        ProductValue("Vitamin A", 80, "ug RE/100 kJ", "", ""),
        ProductValue("Unknown nutrient", 1, "mg/100 kJ", "", ""),
    ]
    chunks = [
        {
            "chunk_text": "Vitamin A requirements are listed in Schedule 29.",
            "filename": "FSANZ Schedule 29.pdf",
            "chunk_index": 3,
            "page_number": 10,
            "distance": 0.1,
            "similarity": 0.9,
        }
    ]

    def fake_retrieve_relevant_chunks(question: str, top_k: int):
        if "Unknown nutrient" in question:
            return []
        return chunks

    def fake_review_row_with_context(row, chunks, workbook_context=""):
        assert "parameter=Vitamin A" in workbook_context
        return {
            "status": "PASS",
            "requirement": "Vitamin A requirement from context.",
            "reasoning": "The value meets the retrieved requirement [Source 1].",
            "citations": ["[Source 1]"],
        }

    monkeypatch.setattr(
        "backend.app.compliance.ai_excel_reviewer.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks,
    )
    monkeypatch.setattr(
        "backend.app.compliance.ai_excel_reviewer.review_row_with_context",
        fake_review_row_with_context,
    )

    result = ai_review_product_values(rows)

    assert result["summary"] == {
        "total": 2,
        "passed": 2,
        "failed": 0,
        "needs_review": 0,
        "insufficient_context": 0,
    }
    assert result["results"][0]["sources"][0]["filename"] == "FSANZ Schedule 29.pdf"
