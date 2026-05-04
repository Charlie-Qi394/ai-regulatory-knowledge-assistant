"""Tests for document loading across supported formats."""

from pathlib import Path

from docx import Document

from backend.app.ingestion.loader import get_source_type, load_document, load_documents


def test_load_txt_document(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Plain text policy note.", encoding="utf-8")

    document = load_document(path)

    assert document.filename == "sample.txt"
    assert document.source_type == "sample_txt"
    assert document.text == "Plain text policy note."
    assert document.pages[0].page_number is None


def test_load_docx_document(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("Word document policy note.")
    doc.save(path)

    document = load_document(path)

    assert document.filename == "sample.docx"
    assert document.source_type == "sample_docx"
    assert "Word document policy note." in document.text
    assert document.pages[0].page_number is None


def test_load_documents_ignores_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("Supported text.", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("Unsupported", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.filename for document in documents] == ["sample.txt"]


def test_get_source_type_for_supported_extensions() -> None:
    assert get_source_type(Path("a.txt")) == "sample_txt"
    assert get_source_type(Path("a.pdf")) == "sample_pdf"
    assert get_source_type(Path("a.docx")) == "sample_docx"
