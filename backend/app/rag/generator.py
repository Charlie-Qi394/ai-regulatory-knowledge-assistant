"""Basic RAG answer generation using retrieved document chunks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from backend.app.rag.embeddings import get_openai_api_key
from backend.app.rag.retriever import RetrievedChunk, retrieve_relevant_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

INSUFFICIENT_CONTEXT_RESPONSE = (
    "I do not have enough information in the provided documents to answer this confidently."
)

SYSTEM_PROMPT = """You are an AI regulatory knowledge assistant.
Answer only from the provided context.
Do not make unsupported claims.
If the context does not contain enough information, say:
"I do not have enough information in the provided documents to answer this confidently."
Keep the answer professional and concise.
Use citations like [Source 1] and [Source 2] when referencing context.
Use plain text only.
Do not use LaTeX, markdown tables, code blocks, or raw formulas with backslashes.
Do not assume missing values or perform calculations unless all required values are explicitly present in the context.
If a calculation needs a value that is not in the context, explain which value is missing and use the fallback sentence."""


class Source(TypedDict):
    """Citation source returned with an answer."""

    source_id: int
    filename: str
    chunk_index: int
    page_number: int | None
    distance: float
    similarity: float
    excerpt: str


class RagAnswer(TypedDict):
    """RAG answer with citation sources."""

    answer: str
    sources: list[Source]


def get_chat_model() -> str:
    """Return the configured OpenAI chat model."""
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def build_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    """Create source metadata for retrieved chunks."""
    sources: list[Source] = []
    for index, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "source_id": index,
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "distance": chunk["distance"],
                "similarity": chunk["similarity"],
                "excerpt": chunk["chunk_text"][:300],
            }
        )
    return sources


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into numbered context blocks."""
    context_blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        location = f"{chunk['filename']} | chunk {chunk['chunk_index']}"
        if chunk["page_number"] is not None:
            location += f" | page {chunk['page_number']}"

        context_blocks.append(
            f"[Source {index}]\n"
            f"Location: {location}\n"
            f"Text: {chunk['chunk_text']}"
        )

    return "\n\n".join(context_blocks)


def generate_answer_from_context(question: str, chunks: list[RetrievedChunk]) -> RagAnswer:
    """Generate a grounded answer using retrieved chunks as context."""
    if not chunks:
        return {"answer": INSUFFICIENT_CONTEXT_RESPONSE, "sources": []}

    context = format_context(chunks)
    client = OpenAI(api_key=get_openai_api_key())

    response = client.chat.completions.create(
        model=get_chat_model(),
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Question:\n"
                    f"{question}\n\n"
                    "Context:\n"
                    f"{context}\n\n"
                    "Answer using only the context above."
                ),
            },
        ],
    )

    answer = response.choices[0].message.content
    if not answer:
        answer = INSUFFICIENT_CONTEXT_RESPONSE

    return {"answer": answer.strip(), "sources": build_sources(chunks)}


def answer_question(question: str, top_k: int = 5) -> RagAnswer:
    """Retrieve relevant chunks and generate a grounded RAG answer."""
    chunks = retrieve_relevant_chunks(question=question, top_k=top_k)
    return generate_answer_from_context(question=question, chunks=chunks)
