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
    DocumentService,
    DocumentStorageError,
    DocumentTooLargeError,
)
from app.services.ingestion import (
    DocumentIngestionError,
    DocumentIngestionLimitError,
    DocumentIngestionService,
)


def get_document_service(request: Request) -> DocumentService:
    """Returns the document storage service for this request."""
    return request.app.state.document_service


def get_document_ingestion_service(
    request: Request,
) -> DocumentIngestionService:
    """Returns the document ingestion service for this request."""
    return request.app.state.document_ingestion_service


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
    document_service: DocumentService,
    ingestion_service: DocumentIngestionService,
    documents: list[DocumentMetadata],
    chunk_ids_by_document_id: dict[UUID, list[str]],
) -> None:
    """Removes all state created by a failed multi-document upload."""
    for document in reversed(documents):
        try:
            await ingestion_service.delete(
                chunk_ids_by_document_id.get(document.id, [])
            )
        except DocumentIngestionError:
            logger.exception("Unable to remove uploaded document vectors")
        try:
            document_service.delete(document.id)
        except DocumentStorageError:
            logger.exception("Unable to remove uploaded document content")


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/embed", response_model=DocumentsResponse, status_code=201)
async def upload_documents(
    request: Request,
    files: Annotated[list[UploadFile], File()],
) -> DocumentsResponse:
    """Stores and indexes uploaded documents atomically."""
    document_service = get_document_service(request)
    ingestion_service = get_document_ingestion_service(request)
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
            document = await document_service.save(file)
            uploaded_documents.append(document)
            chunk_ids = await ingestion_service.ingest(
                document, document_service.content_path(document.id)
            )
            chunk_ids_by_document_id[document.id] = chunk_ids
            document = document_service.set_chunk_ids(document, chunk_ids)
            uploaded_documents[-1] = document
    except DocumentTooLargeError as error:
        await clean_up_uploaded_documents(
            document_service,
            ingestion_service,
            uploaded_documents,
            chunk_ids_by_document_id,
        )
        raise HTTPException(
            status_code=413, detail="Document exceeds maximum allowed size"
        ) from error
    except DocumentIngestionLimitError as error:
        await clean_up_uploaded_documents(
            document_service,
            ingestion_service,
            uploaded_documents,
            chunk_ids_by_document_id,
        )
        raise document_limit_exceeded(
            "Document exceeds maximum allowed chunk count"
        ) from error
    except DocumentIngestionError as error:
        await clean_up_uploaded_documents(
            document_service,
            ingestion_service,
            uploaded_documents,
            chunk_ids_by_document_id,
        )
        raise document_ingestion_unavailable(error) from error
    except DocumentStorageError as error:
        await clean_up_uploaded_documents(
            document_service,
            ingestion_service,
            uploaded_documents,
            chunk_ids_by_document_id,
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
    request: Request,
    ids: Annotated[list[UUID], Query(min_length=1)],
) -> DocumentsResponse:
    """Returns metadata for the requested documents."""
    try:
        document_service = get_document_service(request)
        documents = []
        for document_id in ids:
            document = document_service.get_metadata(document_id)
            if document is None:
                raise document_not_found()
            documents.append(DocumentResponse.model_validate(document))
        return DocumentsResponse(documents=documents)
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error


@router.get("/ids", response_model=DocumentIdsResponse)
async def get_all_ids(request: Request) -> DocumentIdsResponse:
    """Returns identifiers for all valid stored documents."""
    try:
        document_service = get_document_service(request)
        return DocumentIdsResponse(ids=document_service.get_all_ids())
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error


@router.get("/{document_id}")
def download_document(document_id: UUID, request: Request) -> FileResponse:
    """Returns the content of a stored document."""
    try:
        document_service = get_document_service(request)
        document = document_service.get_metadata(document_id)
        if document is None:
            raise document_not_found()
        return FileResponse(
            document_service.content_path(document_id),
            media_type=document.content_type,
            filename=document.filename,
        )
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: UUID, request: Request) -> Response:
    """Removes a document and its indexed chunks."""
    try:
        document_service = get_document_service(request)
        document = document_service.get_metadata(document_id)
        if document is None:
            raise document_not_found()
        await get_document_ingestion_service(request).delete(document.chunk_ids)
        document_service.delete(document_id)
        return Response(status_code=204)
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error
    except DocumentIngestionError as error:
        raise document_ingestion_unavailable(error) from error
