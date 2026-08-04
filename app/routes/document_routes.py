from dataclasses import asdict
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import logger
from app.services.documents import (
    DocumentService,
    DocumentStorageError,
    DocumentTooLargeError,
)


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    content_type: str
    size: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentResponse]


class DocumentIdsResponse(BaseModel):
    ids: list[UUID]


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


def document_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Document not found")


def document_storage_unavailable(error: DocumentStorageError) -> HTTPException:
    logger.exception("Document storage operation failed")
    return HTTPException(status_code=500, detail="Document storage is unavailable")


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=DocumentsResponse, status_code=201)
async def upload_documents(
    request: Request,
    files: Annotated[list[UploadFile], File()],
) -> DocumentsResponse:
    document_service = get_document_service(request)
    documents = []
    for file in files:
        try:
            document = await document_service.save(file)
        except DocumentTooLargeError as error:
            raise HTTPException(
                status_code=413, detail="Document exceeds maximum allowed size"
            ) from error
        except DocumentStorageError as error:
            raise document_storage_unavailable(error) from error
        documents.append(DocumentResponse(**asdict(document)))
    return DocumentsResponse(documents=documents)


@router.get("/", response_model=DocumentsResponse)
def get_documents(
    request: Request,
    ids: Annotated[list[UUID], Query(min_length=1)],
) -> DocumentsResponse:
    try:
        document_service = get_document_service(request)
        documents = []
        for document_id in ids:
            document = document_service.get_metadata(document_id)
            if document is None:
                raise document_not_found()
            documents.append(DocumentResponse(**asdict(document)))
        return DocumentsResponse(documents=documents)
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error


@router.get("/ids", response_model=DocumentIdsResponse)
async def get_all_ids(request: Request) -> DocumentIdsResponse:
    try:
        document_service = get_document_service(request)
        return DocumentIdsResponse(ids=document_service.get_all_ids())
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error


@router.get("/{document_id}")
def download_document(document_id: UUID, request: Request) -> FileResponse:
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
def delete_document(document_id: UUID, request: Request) -> Response:
    try:
        document_service = get_document_service(request)
        if not document_service.delete(document_id):
            raise document_not_found()
        return Response(status_code=204)
    except DocumentStorageError as error:
        raise document_storage_unavailable(error) from error
