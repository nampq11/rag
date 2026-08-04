"""Application-wide constant values."""

from enum import StrEnum


class DocumentContentType(StrEnum):
    """Content types supported by document ingestion."""

    TEXT = "text/plain"
    PDF = "application/pdf"
    BINARY = "application/octet-stream"
