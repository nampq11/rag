from pathlib import Path
from uuid import uuid4

import pytest

from app.models import DocumentMetadata
from app.utils import document_loader
from app.utils.document_loader import infer_content_type


def test_infer_content_type_returns_pdf_for_pdf_extension() -> None:
    assert infer_content_type(Path("report.pdf")) == "application/pdf"
    assert infer_content_type(Path("REPORT.PDF")) == "application/pdf"
    assert (
        infer_content_type(Path("report.docx"))
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert infer_content_type(Path("notes.txt")) == "text/plain"
    assert infer_content_type(Path("data")) == "text/plain"


def test_load_document_converts_office_document_to_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored_path = tmp_path / str(uuid4())
    stored_path.write_bytes(b"spreadsheet bytes")
    document_metadata = DocumentMetadata(
        id=uuid4(),
        filename="quarterly-report.xlsx",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        size=17,
        chunk_ids=[],
    )
    converted_formats: list[str | None] = []

    def convert_to_markdown(_: bytes, file_format: str | None = None) -> str:
        converted_formats.append(file_format)
        return "# Quarterly report\n\nRevenue: $100"

    monkeypatch.setattr(
        document_loader.anydoc, "to_markdown_bytes", convert_to_markdown
    )

    documents = document_loader.load_document(document_metadata, stored_path)

    assert converted_formats == ["xlsx"]
    assert [document.page_content for document in documents] == [
        "# Quarterly report\n\nRevenue: $100"
    ]
    assert documents[0].metadata == {"source": str(stored_path)}
