"""Simple text chunking for local regulatory documents."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping character chunks.

    Args:
        text: Source text to split.
        chunk_size: Maximum characters in each chunk.
        overlap: Characters repeated from the previous chunk.

    Returns:
        Ordered list of non-empty chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(cleaned_text):
            break

        start = end - overlap

    return chunks
