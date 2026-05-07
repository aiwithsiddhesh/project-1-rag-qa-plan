import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_txt_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text(
        "First paragraph with enough text to form a meaningful unit of content.\n\n"
        "Second paragraph with additional content for testing purposes here.\n\n"
        "Third paragraph providing yet more content for test completeness overall."
    )
    return path


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz", reason="PyMuPDF not available on this system", exc_type=ImportError)
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "PDF first paragraph with enough content for loading tests.\n\n"
        "PDF second paragraph with additional content for verification.\n\n"
        "PDF third paragraph rounding out the test document content.",
    )
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_docx_path(tmp_path: Path) -> Path:
    from docx import Document as DocxDocument

    path = tmp_path / "sample.docx"
    doc = DocxDocument()
    doc.add_paragraph("DOCX first paragraph with sufficient content for loading tests.")
    doc.add_paragraph("DOCX second paragraph providing additional content for tests.")
    doc.save(str(path))
    return path


@pytest.fixture
def sample_docs_dir(
    tmp_path: Path,
    sample_txt_path: Path,  # noqa: ARG001
    sample_pdf_path: Path,  # noqa: ARG001
) -> Path:
    return tmp_path


@pytest.fixture
def empty_docs_dir(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "empty"
    docs_dir.mkdir()
    return docs_dir
