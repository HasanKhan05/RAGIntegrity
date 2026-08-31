import pytest

from src.rag.chunk import chunk_pages
from src.rag.models import PageText


def test_chunk_pages_is_deterministic_and_preserves_page_metadata() -> None:
    first_page_text = " ".join(f"word{number:03d}" for number in range(400))
    pages = [
        PageText("doc-1", "brochure.pdf", 1, first_page_text),
        PageText("doc-1", "brochure.pdf", 2, "SECOND_PAGE_ONLY"),
    ]

    first_result = chunk_pages(pages, chunk_size=120, overlap=24)
    second_result = chunk_pages(pages, chunk_size=120, overlap=24)

    page_one_chunks = [chunk for chunk in first_result if chunk.page_number == 1]
    assert len(page_one_chunks) > 1
    assert first_result == second_result
    assert page_one_chunks[0].chunk_id == "doc-1-p1-c0"
    assert all(chunk.filename == "brochure.pdf" for chunk in first_result)
    assert all("SECOND_PAGE_ONLY" not in chunk.text for chunk in page_one_chunks)
    assert set(page_one_chunks[0].text.split()) & set(page_one_chunks[1].text.split())


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 101)],
)
def test_chunk_pages_rejects_invalid_sizes(chunk_size: int, overlap: int) -> None:
    page = PageText("doc-1", "brochure.pdf", 1, "text")

    with pytest.raises(ValueError):
        chunk_pages([page], chunk_size=chunk_size, overlap=overlap)
