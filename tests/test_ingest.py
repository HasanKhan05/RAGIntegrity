from pathlib import Path

import pymupdf
import pytest

from src.rag.ingest import PdfExtractionError, extract_pdf


def _write_two_page_pdf(path: Path) -> None:
    document = pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Toyota RAV4 luggage capacity is 580 litres.")
    document.new_page()
    document.save(path)
    document.close()


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
