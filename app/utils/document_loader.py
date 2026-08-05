"""Document loading by supported content type."""

from pathlib import Path
from typing import Literal, TypeAlias

import anydoc
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.constants import DocumentContentType
from app.models import DocumentMetadata

AnydocFormat: TypeAlias = Literal[
    "csv",
    "doc",
    "docx",
    "epub",
    "odt",
    "ods",
    "odp",
    "pdf",
    "ppt",
    "pptx",
    "rtf",
    "xlsx",
]

DOCUMENT_FORMATS: dict[str, tuple[AnydocFormat, str]] = {
    ".pdf": ("pdf", DocumentContentType.PDF),
    ".doc": ("doc", "application/msword"),
    ".docx": (
        "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ".docm": (
        "docx",
        "application/vnd.ms-word.document.macroenabled.12",
    ),
    ".ppt": ("ppt", "application/vnd.ms-powerpoint"),
    ".pps": ("ppt", "application/vnd.ms-powerpoint"),
    ".pot": ("ppt", "application/vnd.ms-powerpoint"),
    ".pptx": (
        "pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    ".pptm": (
        "pptx",
        "application/vnd.ms-powerpoint.presentation.macroenabled.12",
    ),
    ".ppsx": (
        "pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    ),
    ".ppsm": (
        "pptx",
        "application/vnd.ms-powerpoint.slideshow.macroenabled.12",
    ),
    ".xls": ("xlsx", "application/vnd.ms-excel"),
    ".xlsx": (
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    ".xlsm": ("xlsx", "application/vnd.ms-excel.sheet.macroenabled.12"),
    ".xlsb": ("xlsx", "application/vnd.ms-excel.sheet.binary.macroenabled.12"),
    ".odt": ("odt", "application/vnd.oasis.opendocument.text"),
    ".ods": ("ods", "application/vnd.oasis.opendocument.spreadsheet"),
    ".odp": ("odp", "application/vnd.oasis.opendocument.presentation"),
    ".rtf": ("rtf", "application/rtf"),
    ".epub": ("epub", "application/epub+zip"),
    ".csv": ("csv", "text/csv"),
}


def load_document(metadata: DocumentMetadata, path: Path) -> list[Document]:
    """Loads a document into LangChain documents suitable for chunking."""
    document_format = DOCUMENT_FORMATS.get(Path(metadata.filename).suffix.lower())
    if document_format is None:
        return TextLoader(str(path), autodetect_encoding=True).load()

    markdown = anydoc.to_markdown_bytes(path.read_bytes(), document_format[0])
    return [Document(page_content=markdown, metadata={"source": str(path)})]


def infer_content_type(path: Path) -> str:
    """Guesses the content type of a local file from its filename extension."""
    document_format = DOCUMENT_FORMATS.get(path.suffix.lower())
    if document_format is None:
        return DocumentContentType.TEXT
    return document_format[1]
