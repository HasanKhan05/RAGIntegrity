from pathlib import Path

import pymupdf
import pytest

from src.rag.ingest import PdfExtractionError, extract_pdf, load_pdfs


def _write_two_page_pdf(path: Path) -> None:
    document = pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Toyota RAV4 luggage capacity is 580 litres.")
    document.new_page()
    document.save(path)
    document.close()


def _write_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_load_pdfs_reads_an_arbitrary_pdf_directory(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "poisoned"
    _write_pdf(pdf_dir / "update.pdf", "RAV4 fuel tank capacity is 72 litres")

    pages = load_pdfs(pdf_dir)

    assert [(page.filename, page.page_number) for page in pages] == [
        ("update.pdf", 1)
    ]


def test_extract_pdf_preserves_source_and_skips_blank_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "rav4.pdf"
    _write_two_page_pdf(pdf_path)

    first_result = extract_pdf(pdf_path)
    second_result = extract_pdf(pdf_path)

    assert len(first_result) == 1
    assert first_result[0].filename == "rav4.pdf"
    assert first_result[0].page_number == 1
    assert first_result[0].text == "Toyota RAV4 luggage capacity is 580 litres."
    assert first_result[0].document_id == second_result[0].document_id


def test_extract_pdf_names_unreadable_source(tmp_path: Path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(PdfExtractionError, match="broken.pdf"):
        extract_pdf(pdf_path)
