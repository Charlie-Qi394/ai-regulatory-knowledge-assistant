"""Document ingestion service for saving document metadata."""

from __future__ import annotations

from pathlib import Path

from backend.app.database.connection import get_connection
from backend.app.ingestion.chunking import chunk_text
from backend.app.ingestion.loader import (
    DEFAULT_SAMPLE_DOCS_DIR,
    LoadedDocument,
    load_document,
    load_documents,
)


def document_exists(filename: str, source_type: str) -> bool:
    """Return True if a document metadata row already exists."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM documents
                WHERE filename = %s AND source_type = %s
                LIMIT 1;
                """,
                (filename, source_type),
            )
            return cursor.fetchone() is not None


def save_document_metadata(document: LoadedDocument) -> int | None:
    """Insert document metadata into the documents table.

    Args:
        document: Loaded text document.

    Returns:
        The inserted document ID, or None if the document already exists.
    """
    if document_exists(document.filename, document.source_type):
        print(f"Skipping existing document: {document.filename}")
        return None

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (filename, source_type)
                VALUES (%s, %s)
                RETURNING id;
                """,
                (document.filename, document.source_type),
            )
            inserted_id = cursor.fetchone()[0]
        conn.commit()

    print(f"Saved document metadata: {document.filename} (id={inserted_id})")
    return int(inserted_id)


def ingest_documents(directory: Path | None = None) -> dict[str, int]:
    """Load supported files and save their metadata to PostgreSQL."""
    documents = load_documents() if directory is None else load_documents(directory)

    print(f"Found {len(documents)} supported document(s) for ingestion.")

    inserted = 0
    skipped = 0

    for document in documents:
        inserted_id = save_document_metadata(document)
        if inserted_id is None:
            skipped += 1
        else:
            inserted += 1

    print(f"Ingestion complete. Inserted: {inserted}. Skipped: {skipped}.")
    return {"found": len(documents), "inserted": inserted, "skipped": skipped}


def ingest_txt_documents(directory: Path | None = None) -> dict[str, int]:
    """Backward-compatible wrapper for older scripts."""
    return ingest_documents(directory)


def get_ingested_documents() -> list[tuple[int, str, str]]:
    """Return document IDs, filenames, and source types currently saved."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, filename, source_type
                FROM documents
                WHERE source_type IN (%s, %s, %s)
                ORDER BY id;
                """,
                ("sample_txt", "sample_pdf", "sample_docx"),
            )
            return [(int(row[0]), str(row[1]), str(row[2])) for row in cursor.fetchall()]


def save_document_chunks(
    document_id: int,
    chunks: list[str],
    page_number: int | None = None,
) -> int:
    """Replace stored chunks for one document.

    Args:
        document_id: Database ID from the documents table.
        chunks: Ordered text chunks.
        page_number: Optional page number when available.

    Returns:
        Number of chunks inserted.
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM document_chunks WHERE document_id = %s;",
                (document_id,),
            )
            for index, chunk in enumerate(chunks):
                cursor.execute(
                    """
                    INSERT INTO document_chunks (
                        document_id,
                        chunk_text,
                        chunk_index,
                        page_number
                    )
                    VALUES (%s, %s, %s, %s);
                    """,
                    (document_id, chunk, index, page_number),
                )
        conn.commit()

    return len(chunks)


def chunk_ingested_documents(
    directory: Path = DEFAULT_SAMPLE_DOCS_DIR,
    chunk_size: int = 800,
    overlap: int = 100,
) -> dict[str, int]:
    """Chunk documents already registered in the documents table."""
    ingested_documents = get_ingested_documents()
    print(f"Found {len(ingested_documents)} ingested document(s) to chunk.")

    documents_processed = 0
    chunks_inserted = 0
    missing_files = 0

    for document_id, filename, _source_type in ingested_documents:
        source_path = directory / filename
        if not source_path.exists():
            print(f"Skipping missing file for document id={document_id}: {filename}")
            missing_files += 1
            continue

        document = load_document(source_path)
        inserted = 0
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s;",
                    (document_id,),
                )

                chunk_index = 0
                for page in document.pages:
                    chunks = chunk_text(page.text, chunk_size=chunk_size, overlap=overlap)
                    for chunk in chunks:
                        cursor.execute(
                            """
                            INSERT INTO document_chunks (
                                document_id,
                                chunk_text,
                                chunk_index,
                                page_number
                            )
                            VALUES (%s, %s, %s, %s);
                            """,
                            (document_id, chunk, chunk_index, page.page_number),
                        )
                        chunk_index += 1
                        inserted += 1
            conn.commit()

        print(f"Stored {inserted} chunk(s) for {filename}.")
        documents_processed += 1
        chunks_inserted += inserted

    print(
        "Chunking complete. "
        f"Documents processed: {documents_processed}. "
        f"Chunks inserted: {chunks_inserted}. "
        f"Missing files: {missing_files}."
    )

    return {
        "documents": len(ingested_documents),
        "processed": documents_processed,
        "chunks": chunks_inserted,
        "missing_files": missing_files,
    }
