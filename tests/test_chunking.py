"""Unit tests for simple text chunking."""

import pytest

from backend.app.ingestion.chunking import chunk_text


def test_chunk_text_returns_single_chunk_for_short_text() -> None:
    chunks = chunk_text("Short regulatory note.", chunk_size=800, overlap=100)

    assert chunks == ["Short regulatory note."]


def test_chunk_text_splits_with_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_text(text, chunk_size=10, overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_chunk_text_normalizes_whitespace() -> None:
    chunks = chunk_text("Line one.\n\nLine   two.", chunk_size=100, overlap=10)

    assert chunks == ["Line one. Line two."]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_text_returns_empty_list_for_blank_text() -> None:
    assert chunk_text("   \n\t  ") == []
