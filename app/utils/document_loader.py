"""Document loading by supported content type."""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document

from app.constants import DocumentContentType
from app.models import DocumentMetadata


def load_document(metadata: DocumentMetadata, path: Path) -> list[Document]:
    """Loads a document using the loader for its content type."""
    if metadata.content_type == DocumentContentType.PDF:
        return PyPDFLoader(str(path)).load()
    return TextLoader(str(path), autodetect_encoding=True).load()
