"""Tests for deterministic Excel compliance checks."""

from io import BytesIO

from fastapi.testclient import TestClient
from openpyxl import Workbook

from backend.app.compliance.excel_checker import ProductValue, check_excel_workbook, check_product_values
from backend.main import app


def build_workbook(rows: list[tuple[object, ...]]) -> bytes:
    """Create an in-memory workbook for tests."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["parameter", "value", "unit", "category"])
    for row in rows:
        sheet.append(row)

    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def test_check_product_values_converts_protein_from_g_per_l() -> None:
    rows = [
        ProductValue("Energy", 2720, "kJ/L", "", ""),
        ProductValue("Protein", 15, "g/L", "milk-based", ""),
    ]

    result = check_product_values(rows)

    protein_result = result["results"][1]
    assert protein_result["status"] == "PASS"
    assert protein_result["converted_unit"] == "g/100 kJ"
    assert protein_result["converted_value"] == 0.551471
    assert "Converted from g/L" in protein_result["notes"]


def test_check_product_values_flags_failures_and_review_items() -> None:
    rows = [
        ProductValue("Docosahexaenoic acid", 13, "mg/100 kJ", "", ""),
        ProductValue("Protein", 12, "g/L", "", ""),
        ProductValue("Unknown nutrient", 1, "g/100 kJ", "", ""),
    ]

    result = check_product_values(rows)

    assert result["summary"] == {"total": 3, "passed": 0, "failed": 1, "needs_review": 2}
    assert result["results"][0]["status"] == "FAIL"
    assert result["results"][1]["status"] == "NEEDS_REVIEW"
    assert result["results"][2]["status"] == "NEEDS_REVIEW"


def test_check_excel_workbook_requires_expected_columns() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["parameter", "value"])
    sheet.append(["Energy", 2720])
    output = BytesIO()
    workbook.save(output)

    try:
        check_excel_workbook(output.getvalue())
    except ValueError as exc:
        assert "unit" in str(exc)
    else:
        raise AssertionError("Expected missing column ValueError")


def test_check_excel_endpoint_accepts_xlsx_upload() -> None:
    workbook_bytes = build_workbook(
        [
            ("Energy", 2720, "kJ/L", ""),
            ("Protein", 15, "g/L", "milk-based"),
        ]
    )

    client = TestClient(app)
    response = client.post(
        "/check-excel",
        files={
            "file": (
                "product_check.xlsx",
                workbook_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "product_check.xlsx"
    assert body["summary"]["passed"] == 2
    assert body["results"][1]["source"] == "FSANZ Standard 2.9.1 section 2.9.1-6"


def test_check_excel_endpoint_rejects_non_xlsx_upload() -> None:
    client = TestClient(app)
    response = client.post(
        "/check-excel",
        files={"file": ("product_check.csv", b"parameter,value,unit\nEnergy,2720,kJ/L\n", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please upload a .xlsx file."


def test_ai_review_excel_endpoint_accepts_xlsx_upload(monkeypatch) -> None:
    workbook_bytes = build_workbook([("Vitamin A", 80, "ug RE/100 kJ", "infant formula")])

    def fake_ai_review_excel_workbook(content: bytes) -> dict[str, object]:
        assert content == workbook_bytes
        return {
            "summary": {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "needs_review": 0,
                "insufficient_context": 0,
            },
            "results": [
                {
                    "row_index": 1,
                    "parameter": "Vitamin A",
                    "input_value": 80.0,
                    "input_unit": "ug RE/100 kJ",
                    "category": "infant formula",
                    "status": "PASS",
                    "requirement": "Example requirement.",
                    "reasoning": "The value meets the requirement [Source 1].",
                    "citations": ["[Source 1]"],
                    "sources": [
                        {
                            "source_id": 1,
                            "filename": "FSANZ Schedule 29.pdf",
                            "chunk_index": 4,
                            "page_number": 12,
                            "distance": 0.1,
                            "similarity": 0.9,
                            "excerpt": "Example requirement.",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr("backend.app.api.routes.ai_review_excel_workbook", fake_ai_review_excel_workbook)

    client = TestClient(app)
    response = client.post(
        "/review-excel-ai",
        files={
            "file": (
                "product_check.xlsx",
                workbook_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "product_check.xlsx"
    assert body["summary"]["passed"] == 1
    assert body["results"][0]["citations"] == ["[Source 1]"]
