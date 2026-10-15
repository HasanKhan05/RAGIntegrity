"""Verify serving an existing corpus does not require PDF extraction libraries."""

import subprocess
import sys
from pathlib import Path


def test_api_starts_when_pdf_extraction_dependency_is_unavailable() -> None:
    script = """
import importlib.abc
import sys
from fastapi.testclient import TestClient

class BlockPdfExtraction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pymupdf" or fullname.startswith("pymupdf."):
            raise ImportError("PDF extraction unavailable in this startup test")

sys.meta_path.insert(0, BlockPdfExtraction())
from src.api.main import app

with TestClient(app) as client:
    response = client.get("/documents/catalog")
    assert response.status_code == 200
    assert len(response.json()["official_clean"]) == 7
    assert len(response.json()["synthetic_test"]) == 6
    assert client.get("/results/summary").status_code == 200
assert "pymupdf" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
