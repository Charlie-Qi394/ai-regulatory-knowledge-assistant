"""Tests for simple evaluation helpers."""

from backend.app.evaluation.evaluator import (
    answer_contains_expected_info,
    parse_expected_keywords,
)
from backend.app.evaluation.ragas_evaluator import parse_expected_contexts


def test_parse_expected_keywords() -> None:
    keywords = parse_expected_keywords(" label | allergen | nutrition ")

    assert keywords == ["label", "allergen", "nutrition"]


def test_answer_contains_expected_info_passes_when_all_keywords_present() -> None:
    answer = "The label should include allergen and nutrition information."

    assert answer_contains_expected_info(answer, ["label", "allergen", "nutrition"])


def test_answer_contains_expected_info_fails_when_keyword_missing() -> None:
    answer = "The label should include allergen information."

    assert not answer_contains_expected_info(answer, ["label", "nutrition"])


def test_parse_expected_contexts() -> None:
    contexts = parse_expected_contexts("first context || second context ")

    assert contexts == ["first context", "second context"]


def test_parse_expected_contexts_returns_empty_list_for_blank_input() -> None:
    assert parse_expected_contexts("") == []
    assert parse_expected_contexts(None) == []
