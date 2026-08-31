"""Deterministic fixed-size chunking that never crosses PDF pages."""

from __future__ import annotations

from collections.abc import Iterable

from src.rag.models import PageText, TextChunk


def chunk_pages(
    pages: Iterable[PageText], chunk_size: int = 1200, overlap: int = 200
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks: list[TextChunk] = []
    for page in pages:
        start = 0
        chunk_index = 0
        while start < len(page.text):
            target_end = min(start + chunk_size, len(page.text))
            end = target_end
            if target_end < len(page.text):
                boundary_floor = start + int(chunk_size * 0.85)
                whitespace = page.text.rfind(" ", boundary_floor, target_end)
                if whitespace > start:
                    end = whitespace

            text = page.text[start:end].strip()
            if text:
                chunks.append(
                    TextChunk(
                        document_id=page.document_id,
                        filename=page.filename,
                        page_number=page.page_number,
                        chunk_id=(
                            f"{page.document_id}-p{page.page_number}-c{chunk_index}"
                        ),
                        text=text,
                    )
                )
                chunk_index += 1

            if end >= len(page.text):
                break
            start = max(end - overlap, start + 1)

    return chunks
