"""Document chunking and vector-store ingestion functions."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import DocumentMetadata
from app.utils.document_loader import load_document

AddDocuments = Callable[[list[Document], list[str]], Awaitable[None]]
DeleteDocuments = Callable[[list[str]], Awaitable[None]]


class DocumentIngestionError(RuntimeError):
    """Indicates document ingestion or vector cleanup failed."""


class DocumentIngestionLimitError(DocumentIngestionError):
    """Indicates document ingestion exceeded a configured limit."""


async def ingest_document(
    add_documents: AddDocuments,
    delete_documents: DeleteDocuments,
    metadata: DocumentMetadata,
    path: Path,
    chunk_size: int,
    chunk_overlap: int,
    maximum_chunks: int,
    batch_size: int,
) -> list[str]:
    """Chunks and indexes a document, rolling back failed writes."""
    try:
        documents = await asyncio.to_thread(load_document, metadata, path)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_documents(documents)
        if len(chunks) > maximum_chunks:
            raise DocumentIngestionLimitError(
                "Document exceeds maximum allowed chunk count"
            )
        chunk_ids = [f"{metadata.id}:{index}" for index in range(len(chunks))]
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "document_id": str(metadata.id),
                    "filename": metadata.filename,
                    "content_type": metadata.content_type,
                }
            )
    except DocumentIngestionLimitError:
        raise
    except Exception as error:
        raise DocumentIngestionError(
            f"Unable to ingest document {metadata.id}"
        ) from error

    inserted_chunk_ids: list[str] = []
    try:
        for start in range(0, len(chunks), batch_size):
            batch_documents = chunks[start : start + batch_size]
            batch_chunk_ids = chunk_ids[start : start + batch_size]
            inserted_chunk_ids.extend(batch_chunk_ids)
            await add_documents(batch_documents, ids=batch_chunk_ids)
        return chunk_ids
    except Exception as error:
        if inserted_chunk_ids:
            try:
                await delete_documents(inserted_chunk_ids)
            except Exception as cleanup_error:
                raise DocumentIngestionError(
                    f"Unable to roll back document ingestion for {metadata.id}"
                ) from cleanup_error
        raise DocumentIngestionError(
            f"Unable to ingest document {metadata.id}"
        ) from error


async def delete_document_vectors(
    delete_documents: DeleteDocuments, chunk_ids: list[str]
) -> None:
    """Deletes a document's chunks from vector storage."""
    if not chunk_ids:
        return
    try:
        await delete_documents(chunk_ids)
    except Exception as error:
        raise DocumentIngestionError(
            "Unable to remove document from vector storage"
        ) from error
