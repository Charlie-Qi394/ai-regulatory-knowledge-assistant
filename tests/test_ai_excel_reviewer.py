"""Tests for AI-assisted Excel review plumbing."""

from backend.app.compliance.ai_excel_reviewer import (
    ai_review_product_values,
    build_workbook_question,
    build_row_question,
    format_workbook_context,
    parse_batch_review_json,
    parse_review_json,
    select_rows_for_focus,
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


def test_build_workbook_question_includes_focus_instructions() -> None:
    rows = [ProductValue("Protein", 15, "g/L", "milk-based", "")]

    question = build_workbook_question(rows, focus_instructions="Focus on protein conversion.")

    assert "Protein" in question
    assert "Focus on protein conversion." in question


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


def test_parse_batch_review_json_maps_results_by_row_index() -> None:
    result = parse_batch_review_json(
        '{"results": [{"row_index": 2, "status": "fail", "requirement": "Limit", '
        '"reasoning": "Above limit [Source 1].", "citations": ["[Source 1]"]}]}'
    )

    assert result[2]["status"] == "FAIL"
    assert result[2]["requirement"] == "Limit"


def test_format_workbook_context_includes_supporting_rows() -> None:
    rows = [
        ProductValue("Energy", 2720, "kJ/L", "Nutrition", ""),
        ProductValue("Protein", 15, "g/L", "Nutrition", "Inferred row"),
    ]

    context = format_workbook_context(rows)

    assert "parameter=Energy" in context
    assert "value=2720" in context
    assert "parameter=Protein" in context


def test_select_rows_for_focus_filters_spec_min_and_max_can_nip_rows() -> None:
    rows = [
        ProductValue("Protein", 11.8, "g", "Can Target (= Total x 0.976)", ""),
        ProductValue("Protein", 10.4, "g", "Spec Min (Can / NIP)", ""),
        ProductValue("Protein", 13.8, "g", "Spec Max (Can / NIP)", ""),
        ProductValue("Protein", 9.51, "g", "Old FSANZ Min (per 100g)", ""),
    ]

    selected = select_rows_for_focus(
        rows,
        "Please check the spec min (can/NIP) and spec Max (can/NIP) for each nutrient.",
    )

    assert [(row.parameter, row.category) for row in selected] == [
        ("Protein", "Spec Min (Can / NIP)"),
        ("Protein", "Spec Max (Can / NIP)"),
    ]


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
        assert "Vitamin A" in question
        assert "Unknown nutrient" in question
        return chunks

    def fake_review_rows_with_context(rows, chunks, focus_instructions=""):
        assert rows[0].parameter == "Vitamin A"
        assert focus_instructions == "Focus on vitamin A."
        return {
            1: {
                "status": "PASS",
                "requirement": "Vitamin A requirement from context.",
                "reasoning": "The value meets the retrieved requirement [Source 1].",
                "citations": ["[Source 1]"],
            },
            2: {
                "status": "NEEDS_REVIEW",
                "requirement": "",
                "reasoning": "No clear threshold was found [Source 1].",
                "citations": ["[Source 1]"],
            },
        }

    monkeypatch.setattr(
        "backend.app.compliance.ai_excel_reviewer.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks,
    )
    monkeypatch.setattr(
        "backend.app.compliance.ai_excel_reviewer.review_rows_with_context",
        fake_review_rows_with_context,
    )

    result = ai_review_product_values(rows, focus_instructions="Focus on vitamin A.")

    assert result["summary"] == {
        "total": 2,
        "passed": 1,
        "failed": 0,
        "needs_review": 1,
        "insufficient_context": 0,
    }
    assert result["results"][0]["sources"][0]["filename"] == "FSANZ Schedule 29.pdf"


def test_ai_review_product_values_uses_coded_result_for_known_rules(monkeypatch) -> None:
    rows = [ProductValue("Docosahexaenoic acid", 13, "mg/100 kJ", "", "")]

    def fake_retrieve_relevant_chunks(question: str, top_k: int):
        return []

    monkeypatch.setattr(
        "backend.app.compliance.ai_excel_reviewer.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks,
    )

    result = ai_review_product_values(rows)

    assert result["summary"]["failed"] == 1
    assert result["results"][0]["status"] == "FAIL"
    assert result["results"][0]["requirement"] == "<= 12 mg/100 kJ"
    assert result["results"][0]["citations"] == ["FSANZ Schedule 29 section S29-4"]
