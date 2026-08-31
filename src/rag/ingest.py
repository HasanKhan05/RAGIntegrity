"""Extract readable page text from local PDF corpora."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pymupdf

from src.rag.models import PageText


class PdfExtractionError(RuntimeError):
    """Raised when a named PDF cannot be read safely."""


def _document_id(filename: str) -> str:
    return sha256(filename.encode("utf-8")).hexdigest()[:16]


def extract_pdf(path: Path) -> list[PageText]:
    """Extract non-empty pages while preserving source metadata."""

    source = Path(path)
    pages: list[PageText] = []
    try:
        with pymupdf.open(source) as document:
            for page_index, page in enumerate(document):
                text = " ".join(page.get_text().split())
                if not text:
                    continue
                pages.append(
                    PageText(
                        document_id=_document_id(source.name),
                        filename=source.name,
                        page_number=page_index + 1,
                        text=text,
                    )
                )
    except Exception as error:
        raise PdfExtractionError(f"Could not read PDF: {source.name}") from error
    return pages


def load_pdfs(directory: Path) -> list[PageText]:
    """Load all PDFs from a directory in stable filename order."""

    pdf_directory = Path(directory)
    if not pdf_directory.exists():
        return []
    paths = sorted(
        (
            path
            for path in pdf_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.lower(),
    )
    return [page for path in paths for page in extract_pdf(path)]


def load_clean_pdfs(directory: Path) -> list[PageText]:
    """Load clean PDFs through the generic local-corpus loader."""

    return load_pdfs(directory)
