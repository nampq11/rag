"""Document chunking and vector-store ingestion service."""

import asyncio
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import DocumentMetadata
from app.utils.document_loader import load_document


class VectorDocumentStore(Protocol):
    """Defines the vector-store operations used during ingestion."""

    async def add_documents(
        self, documents: list[Document], ids: list[str]
    ) -> None:
        """Adds document chunks to the vector store."""
        ...

    async def delete(self, ids: list[str]) -> None:
        """Deletes document chunks from the vector store."""
        ...


class DocumentIngestionError(RuntimeError):
    """Indicates document ingestion or vector cleanup failed."""


class DocumentIngestionLimitError(DocumentIngestionError):
    """Indicates document ingestion exceeded a configured limit."""


class DocumentIngestionService:
    """Loads, chunks, and indexes documents in a vector store."""

    def __init__(
        self,
        vector_store: VectorDocumentStore,
        chunk_size: int,
        chunk_overlap: int,
        maximum_chunks: int,
        batch_size: int,
    ) -> None:
        self.vector_store = vector_store
        self.maximum_chunks = maximum_chunks
        self.batch_size = batch_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def ingest(self, metadata: DocumentMetadata, path: Path) -> list[str]:
        """Chunks and indexes a document, rolling back failed writes."""
        try:
            documents = await asyncio.to_thread(
                load_document, metadata, path
            )
            chunks = self.text_splitter.split_documents(documents)
            if len(chunks) > self.maximum_chunks:
                raise DocumentIngestionLimitError(
                    "Document exceeds maximum allowed chunk count"
                )
            chunk_ids = [
                f"{metadata.id}:{index}" for index in range(len(chunks))
            ]
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
            for start in range(0, len(chunks), self.batch_size):
                batch_documents = chunks[start : start + self.batch_size]
                batch_chunk_ids = chunk_ids[start : start + self.batch_size]
                inserted_chunk_ids.extend(batch_chunk_ids)
                await self.vector_store.add_documents(
                    batch_documents, batch_chunk_ids
                )
            return chunk_ids
        except Exception as error:
            if inserted_chunk_ids:
                try:
                    await self.vector_store.delete(inserted_chunk_ids)
                except Exception as cleanup_error:
                    raise DocumentIngestionError(
                        "Unable to roll back document ingestion for "
                        f"{metadata.id}"
                    ) from cleanup_error
            raise DocumentIngestionError(
                f"Unable to ingest document {metadata.id}"
            ) from error

    async def delete(self, chunk_ids: list[str]) -> None:
        """Deletes a stored document."""
        if not chunk_ids:
            return
        try:
            await self.vector_store.delete(chunk_ids)
        except Exception as error:
            raise DocumentIngestionError(
                "Unable to remove document from vector storage"
            ) from error
