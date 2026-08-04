"""HTTP endpoints for document storage and ingestion."""

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.config import logger
from app.models import (
    DocumentIdsResponse,
    DocumentMetadata,
    DocumentResponse,
    DocumentsResponse,
)
from app.services.documents import (
    DocumentStorageError,
    DocumentTooLargeError,
    content_path,
    delete_document,
    get_all_ids,
    get_metadata,
    save_document,
    set_chunk_ids,
)
from app.services.ingestion import (
    DocumentIngestionError,
    DocumentIngestionLimitError,
    delete_document_vectors,
    ingest_document,
)


def document_not_found() -> HTTPException:
    """Builds the standard document-not-found response."""
    return HTTPException(status_code=404, detail="Document not found")


def document_storage_unavailable(error: DocumentStorageError) -> HTTPException:
    """Builds the response for a document storage failure."""
    logger.exception("Document storage operation failed")
    return HTTPException(
        status_code=500, detail="Document storage is unavailable"
    )


def document_ingestion_unavailable(
    error: DocumentIngestionError,
) -> HTTPException:
    """Builds the response for a document ingestion failure."""
    logger.exception("Document ingestion operation failed")
    return HTTPException(
        status_code=500, detail="Document ingestion is unavailable"
    )


def document_limit_exceeded(detail: str) -> HTTPException:
    """Builds the response for a configured document limit."""
    return HTTPException(status_code=413, detail=detail)


async def clean_up_uploaded_documents(
    request: Request,
    documents: list[DocumentMetadata],
    chunk_ids_by_document_id: dict[UUID, list[str]],
) -> None:
    """Removes all state created by a failed multi-document upload."""
    for document in reversed(documents):
        try:
            await delete_document_vectors(
                request.app.state.vector_store.adelete,
                chunk_ids_by_document_id.get(document.id, []),
            )
        except DocumentIngestionError:
            logger.exception("Unable to remove uploaded document vectors")
        try:
            delete_document(request.app.state.upload_directory, document.id)
        except DocumentStorageError:
            logger.exception("Unable to remove uploaded document content")


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/embed", response_model=DocumentsResponse, status_code=201)
async def upload_documents(
    request: Request, files: Annotated[list[UploadFile], File()]
) -> DocumentsResponse:
    """Stores and indexes uploaded documents atomically."""
    if len(files) > request.app.state.maximum_document_count:
        raise document_limit_exceeded("Too many documents in one upload")
    file_sizes = [file.size for file in files]
    if any(size is None for size in file_sizes):
        raise HTTPException(
            status_code=500, detail="Unable to determine document size"
        )
    total_size = sum(size for size in file_sizes if size is not None)
    if total_size > request.app.state.maximum_total_document_size_bytes:
        raise document_limit_exceeded(
            "Documents exceed maximum total allowed size"
        )

    uploaded_documents: list[DocumentMetadata] = []
    chunk_ids_by_document_id: dict[UUID, list[str]] = {}
    try:
        for file in files:
            document = await save_document(
                request.app.state.upload_directory,
                request.app.state.maximum_document_size_bytes,
                file,
            )
            uploaded_documents.append(document)
            chunk_ids = await ingest_document(
                request.app.state.vector_store.aadd_documents,
                request.app.state.vector_store.adelete,
                document,
                content_path(request.app.state.upload_directory, document.id),
                request.app.state.document_chunk_size,
                request.app.state.document_chunk_overlap,
                request.app.state.document_maximum_chunks,
                request.app.state.embedding_batch_size,
            )
            chunk_ids_by_document_id[document.id] = chunk_ids
            uploaded_documents[-1] = set_chunk_ids(
                request.app.state.upload_directory, document, chunk_ids
            )
    except DocumentTooLargeError as error:
        await clean_up_uploaded_documents(
            request, uploaded_documents, chunk_ids_by_document_id
        )
        raise HTTPException(
            status_code=413, detail="Document exceeds maximum allowed size"
        ) from error
    except DocumentIngestionLimitError as error:
        await clean_up_uploaded_documents(
            request, uploaded_documents, chunk_ids_by_document_id
        )
        raise document_limit_exceeded(
            "Document exceeds maximum allowed chunk count"
        ) from error
    except DocumentIngestionError as error:
        await clean_up_uploaded_documents(
            request, uploaded_documents, chunk_ids_by_document_id
        )
        raise document_ingestion_unavailable(error) from error
    except DocumentStorageError as error:
        await clean_up_uploaded_documents(
            request, uploaded_documents, chunk_ids_by_document_id
        )
        raise document_storage_unavailable(error) from error
    return DocumentsResponse(
        documents=[
            DocumentResponse.model_validate(document)
            for document in uploaded_documents
        ]
    )


@router.get("/", response_model=DocumentsResponse)
def get_documents(
    request: Request, ids: Annotated[list[UUID], Query(min_length=1)]
) -> DocumentsResponse:
    """Returns metadata for the requested documents."""
    try:
        documents = []
        for document_id in ids:
            document = get_metadata(
                request.app.state.upload_directory, document_id
            )
            if document is None:
                raise document_not_found()
            documents.append(DocumentResponse.model_validate(document))
        return DocumentsResponse(documents=documents)
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error


@router.get("/ids", response_model=DocumentIdsResponse)
async def get_all_document_ids(request: Request) -> DocumentIdsResponse:
    """Returns identifiers for all valid stored documents."""
    try:
        return DocumentIdsResponse(
            ids=get_all_ids(request.app.state.upload_directory)
        )
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error


@router.get("/{document_id}")
def download_document(document_id: UUID, request: Request) -> FileResponse:
    """Returns the content of a stored document."""
    try:
        document = get_metadata(request.app.state.upload_directory, document_id)
        if document is None:
            raise document_not_found()
        return FileResponse(
            content_path(request.app.state.upload_directory, document_id),
            media_type=document.content_type,
            filename=document.filename,
        )
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error


@router.delete("/{document_id}", status_code=204)
async def delete_stored_document(
    document_id: UUID, request: Request
) -> Response:
    """Removes a document and its indexed chunks."""
    try:
        document = get_metadata(request.app.state.upload_directory, document_id)
        if document is None:
            raise document_not_found()
        await delete_document_vectors(
            request.app.state.vector_store.adelete, document.chunk_ids
        )
        delete_document(request.app.state.upload_directory, document_id)
        return Response(status_code=204)
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error
    except DocumentIngestionError as error:
        raise document_ingestion_unavailable(error) from error
