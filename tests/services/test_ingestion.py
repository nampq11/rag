import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from app.constants import DocumentContentType
from app.models import DocumentMetadata
from app.services.ingestion import (
    DocumentIngestionError,
    DocumentIngestionLimitError,
    delete_document_vectors,
    ingest_document,
)


class RecordingVectorStore:
    def __init__(self) -> None:
        self.added_batches: list[tuple[list[Document], list[str]]] = []
        self.deleted_ids: list[str] = []

    async def aadd_documents(
        self, documents: list[Document], ids: list[str]
    ) -> None:
        self.added_batches.append((documents, ids))

    async def adelete(self, ids: list[str]) -> None:
        self.deleted_ids = ids


def metadata() -> DocumentMetadata:
    return DocumentMetadata(
        id=uuid4(),
        filename="notes.txt",
        content_type=DocumentContentType.TEXT,
        size=30,
        chunk_ids=[],
    )


def ingest(
    vector_store: RecordingVectorStore,
    metadata: DocumentMetadata,
    path: Path,
    **settings: int,
) -> list[str]:
    return asyncio.run(
        ingest_document(
            vector_store.aadd_documents,
            vector_store.adelete,
            metadata,
            path,
            chunk_size=settings.get("chunk_size", 10),
            chunk_overlap=settings.get("chunk_overlap", 2),
            maximum_chunks=settings.get("maximum_chunks", 10),
            batch_size=settings.get("batch_size", 2),
        )
    )


def test_ingest_chunks_text_preserves_metadata_and_batches_vectors(
    tmp_path: Path,
) -> None:
    document_metadata = metadata()
    content_path = tmp_path / str(document_metadata.id)
    content_path.write_text("one two three four five six", encoding="utf-8")
    vector_store = RecordingVectorStore()

    chunk_ids = ingest(vector_store, document_metadata, content_path)

    assert chunk_ids == [
        f"{document_metadata.id}:{index}" for index in range(4)
    ]
    assert [ids for _, ids in vector_store.added_batches] == [
        chunk_ids[:2],
        chunk_ids[2:],
    ]
    assert [chunk.metadata for chunk in vector_store.added_batches[0][0]] == [
        {
            "source": str(content_path),
            "document_id": str(document_metadata.id),
            "filename": "notes.txt",
            "content_type": "text/plain",
        },
    ] * 2


def test_delete_uses_persisted_chunk_ids_without_reloading_document(
    tmp_path: Path,
) -> None:
    document_metadata = metadata()
    content_path = tmp_path / str(document_metadata.id)
    content_path.write_text(
        "content can change after ingestion", encoding="utf-8"
    )
    vector_store = RecordingVectorStore()
    persisted_chunk_ids = [
        f"{document_metadata.id}:{index}" for index in range(4)
    ]

    content_path.unlink()
    asyncio.run(
        delete_document_vectors(vector_store.adelete, persisted_chunk_ids)
    )

    assert vector_store.deleted_ids == persisted_chunk_ids


def test_ingest_rejects_documents_with_too_many_chunks(tmp_path: Path) -> None:
    document_metadata = metadata()
    content_path = tmp_path / str(document_metadata.id)
    content_path.write_text("one two three four five six", encoding="utf-8")

    with pytest.raises(DocumentIngestionLimitError, match="chunk count"):
        ingest(
            RecordingVectorStore(),
            document_metadata,
            content_path,
            maximum_chunks=3,
        )


def test_ingest_rolls_back_all_batches_when_insertion_fails(
    tmp_path: Path,
) -> None:
    class FailingVectorStore(RecordingVectorStore):
        async def aadd_documents(
            self, documents: list[Document], ids: list[str]
        ) -> None:
            await super().aadd_documents(documents, ids)
            if len(self.added_batches) == 2:
                raise RuntimeError("Database unavailable")

    document_metadata = metadata()
    content_path = tmp_path / str(document_metadata.id)
    content_path.write_text("one two three four five six", encoding="utf-8")
    vector_store = FailingVectorStore()

    with pytest.raises(DocumentIngestionError, match="Unable to ingest"):
        ingest(vector_store, document_metadata, content_path)

    assert vector_store.deleted_ids == [
        f"{document_metadata.id}:{index}" for index in range(4)
    ]


def test_ingest_wraps_loader_errors(tmp_path: Path) -> None:
    with pytest.raises(
        DocumentIngestionError, match="Unable to ingest document"
    ):
        ingest(RecordingVectorStore(), metadata(), tmp_path / "missing")
