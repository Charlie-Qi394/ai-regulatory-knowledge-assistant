"""Deterministic Excel compliance checks for selected infant-formula rules.

The LLM should not perform regulatory calculations. This module parses a simple
Excel input, applies explicit Python calculations, and returns auditable
pass/fail/needs-review rows with source notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

from openpyxl import load_workbook


REQUIRED_COLUMNS = {"parameter", "value", "unit"}


@dataclass(frozen=True)
class ProductValue:
    """One row extracted from the uploaded workbook."""

    parameter: str
    value: float | None
    unit: str
    category: str
    notes: str


@dataclass(frozen=True)
class Rule:
    """One deterministic regulatory check."""

    key: str
    label: str
    min_value: float | None
    max_value: float | None
    unit: str
    source: str


RULES = {
    "energy": Rule(
        key="energy",
        label="Energy content",
        min_value=2510,
        max_value=2930,
        unit="kJ/L",
        source="FSANZ Standard 2.9.1 section 2.9.1-5",
    ),
    "protein_milk_based": Rule(
        key="protein_milk_based",
        label="Protein content, milk-based infant formula",
        min_value=0.43,
        max_value=0.72,
        unit="g/100 kJ",
        source="FSANZ Standard 2.9.1 section 2.9.1-6",
    ),
    "protein_non_milk_based": Rule(
        key="protein_non_milk_based",
        label="Protein content, non-milk-based infant formula",
        min_value=0.54,
        max_value=0.72,
        unit="g/100 kJ",
        source="FSANZ Standard 2.9.1 section 2.9.1-6",
    ),
    "dha": Rule(
        key="dha",
        label="Docosahexaenoic acid",
        min_value=None,
        max_value=12,
        unit="mg/100 kJ",
        source="FSANZ Schedule 29 section S29-4",
    ),
    "total_trans_fatty_acids": Rule(
        key="total_trans_fatty_acids",
        label="Total trans fatty acids",
        min_value=None,
        max_value=4,
        unit="% of total fatty acids",
        source="FSANZ Schedule 29 section S29-4",
    ),
}


def normalize_text(value: object) -> str:
    """Normalize text for loose matching."""
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def normalize_unit(value: object) -> str:
    """Normalize common unit spellings."""
    unit = normalize_text(value).replace(" ", "")
    aliases = {
        "kj/l": "kJ/L",
        "kilojoules/l": "kJ/L",
        "kilojoulesperlitre": "kJ/L",
        "g/100kj": "g/100 kJ",
        "gper100kj": "g/100 kJ",
        "g/l": "g/L",
        "gperlitre": "g/L",
        "mg/100kj": "mg/100 kJ",
        "mgper100kj": "mg/100 kJ",
        "%": "% of total fatty acids",
        "%totalfattyacids": "% of total fatty acids",
        "%oftotalfattyacids": "% of total fatty acids",
        "percentoftotalfattyacids": "% of total fatty acids",
    }
    return aliases.get(unit, str(value or "").strip())


def parse_float(value: object) -> float | None:
    """Parse numeric cell values while tolerating percentage strings."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def load_product_values(workbook_bytes: bytes) -> list[ProductValue]:
    """Load product values from the first sheet of an uploaded workbook."""
    workbook = load_workbook(BytesIO(workbook_bytes), data_only=True)
    sheet = workbook.active

    header_row = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), None)
    if not header_row:
        raise ValueError("Workbook must include a header row.")

    headers = [normalize_text(header) for header in header_row]
    header_map = {header: index for index, header in enumerate(headers) if header}
    missing = REQUIRED_COLUMNS - set(header_map)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Workbook is missing required column(s): {missing_columns}.")

    values: list[ProductValue] = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if not any(cell not in (None, "") for cell in row):
            continue

        def get_cell(column: str) -> object:
            index = header_map.get(column)
            return row[index] if index is not None and index < len(row) else ""

        values.append(
            ProductValue(
                parameter=str(get_cell("parameter") or "").strip(),
                value=parse_float(get_cell("value")),
                unit=str(get_cell("unit") or "").strip(),
                category=str(get_cell("category") or "").strip(),
                notes=str(get_cell("notes") or "").strip(),
            )
        )

    if not values:
        raise ValueError("Workbook does not contain any product data rows.")

    return values


def detect_rule_key(row: ProductValue) -> str | None:
    """Map an input row to one supported rule key."""
    parameter = normalize_text(row.parameter)
    category = normalize_text(row.category)
    combined = f"{parameter} {category}"

    if "energy" in parameter:
        return "energy"
    if "protein" in parameter:
        if "non milk" in combined or "non-milk" in combined or "soy" in combined:
            return "protein_non_milk_based"
        if "milk" in combined:
            return "protein_milk_based"
        return "protein_unspecified"
    if "docosahexaenoic" in combined or "dha" in combined:
        return "dha"
    if "trans" in combined and "fat" in combined:
        return "total_trans_fatty_acids"

    return None


def find_energy_kj_per_l(rows: list[ProductValue]) -> float | None:
    """Find energy content in kJ/L if provided."""
    for row in rows:
        if detect_rule_key(row) == "energy" and normalize_unit(row.unit) == "kJ/L":
            return row.value
    return None


def format_requirement(rule: Rule) -> str:
    """Return a readable requirement range."""
    if rule.min_value is not None and rule.max_value is not None:
        return f"{rule.min_value:g}-{rule.max_value:g} {rule.unit}"
    if rule.min_value is not None:
        return f">= {rule.min_value:g} {rule.unit}"
    if rule.max_value is not None:
        return f"<= {rule.max_value:g} {rule.unit}"
    return rule.unit


def compare_value(value: float, rule: Rule) -> str:
    """Return PASS or FAIL for a converted value."""
    if rule.min_value is not None and value < rule.min_value:
        return "FAIL"
    if rule.max_value is not None and value > rule.max_value:
        return "FAIL"
    return "PASS"


def convert_value(row: ProductValue, rule: Rule, energy_kj_per_l: float | None) -> tuple[float | None, str]:
    """Convert an input value to the rule unit when supported."""
    if row.value is None:
        return None, "Value is missing or not numeric."

    input_unit = normalize_unit(row.unit)
    if input_unit == rule.unit:
        return row.value, ""

    if rule.unit == "g/100 kJ" and input_unit == "g/L":
        if not energy_kj_per_l:
            return None, "Protein in g/L requires energy content in kJ/L for conversion."
        return row.value / energy_kj_per_l * 100, "Converted from g/L using energy content."

    return None, f"Unsupported unit conversion from {row.unit!r} to {rule.unit}."


def result_row(
    row: ProductValue,
    status: str,
    rule: Rule | None = None,
    converted_value: float | None = None,
    converted_unit: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build a serializable compliance result row."""
    return {
        "parameter": row.parameter,
        "input_value": row.value,
        "input_unit": row.unit,
        "category": row.category,
        "converted_value": round(converted_value, 6) if converted_value is not None else None,
        "converted_unit": converted_unit,
        "requirement": format_requirement(rule) if rule else "",
        "status": status,
        "source": rule.source if rule else "",
        "notes": notes,
    }


def check_product_values(rows: list[ProductValue]) -> dict[str, Any]:
    """Check uploaded product values against the supported rule set."""
    energy_kj_per_l = find_energy_kj_per_l(rows)
    results: list[dict[str, Any]] = []

    for row in rows:
        rule_key = detect_rule_key(row)
        if rule_key is None:
            results.append(
                result_row(
                    row,
                    status="NEEDS_REVIEW",
                    notes="No supported deterministic rule is implemented for this parameter.",
                )
            )
            continue

        if rule_key == "protein_unspecified":
            results.append(
                result_row(
                    row,
                    status="NEEDS_REVIEW",
                    notes="Protein checks require category to state milk-based or non-milk-based.",
                )
            )
            continue

        rule = RULES[rule_key]
        converted_value, note = convert_value(row=row, rule=rule, energy_kj_per_l=energy_kj_per_l)
        if converted_value is None:
            results.append(result_row(row, status="NEEDS_REVIEW", rule=rule, notes=note))
            continue

        status = compare_value(converted_value, rule)
        results.append(
            result_row(
                row,
                status=status,
                rule=rule,
                converted_value=converted_value,
                converted_unit=rule.unit,
                notes=note,
            )
        )

    summary = {
        "total": len(results),
        "passed": sum(1 for row in results if row["status"] == "PASS"),
        "failed": sum(1 for row in results if row["status"] == "FAIL"),
        "needs_review": sum(1 for row in results if row["status"] == "NEEDS_REVIEW"),
    }

    return {"summary": summary, "results": results}


def check_excel_workbook(workbook_bytes: bytes) -> dict[str, Any]:
    """Parse and check an uploaded Excel workbook."""
    rows = load_product_values(workbook_bytes)
    return check_product_values(rows)
