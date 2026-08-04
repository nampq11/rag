"""Filesystem-backed document storage functions."""

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.constants import DocumentContentType
from app.models import DocumentMetadata


class DocumentStorageError(RuntimeError):
    """Indicates a document storage operation failed."""


class DocumentTooLargeError(DocumentStorageError):
    """Indicates an upload exceeded the configured size limit."""


def initialize_storage(directory: Path) -> None:
    """Creates the document storage directory when needed."""
    directory.mkdir(parents=True, exist_ok=True)


def content_path(directory: Path, document_id: UUID) -> Path:
    """Returns the filesystem path for document content."""
    return directory / str(document_id)


def metadata_path(directory: Path, document_id: UUID) -> Path:
    """Returns the filesystem path for document metadata."""
    return directory / f"{document_id}.json"


def write_metadata(directory: Path, metadata: DocumentMetadata) -> None:
    """Atomically writes document metadata."""
    destination = metadata_path(directory, metadata.id)
    temporary_path = destination.with_suffix(".json.uploading")
    try:
        temporary_path.write_text(metadata.model_dump_json(), encoding="utf-8")
        temporary_path.replace(destination)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise DocumentStorageError(
            "Unable to store document metadata"
        ) from error


def get_metadata(directory: Path, document_id: UUID) -> DocumentMetadata | None:
    """Loads metadata for a stored document when it exists."""
    stored_metadata_path = metadata_path(directory, document_id)
    if not stored_metadata_path.is_file():
        return None
    if not content_path(directory, document_id).is_file():
        raise DocumentStorageError("Document content is missing")

    try:
        return DocumentMetadata.model_validate_json(
            stored_metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise DocumentStorageError(
            "Unable to read document metadata"
        ) from error


def get_all_ids(directory: Path) -> list[UUID]:
    """Returns identifiers for all valid stored documents."""
    try:
        stored_metadata_paths = list(directory.glob("*.json"))
    except OSError as error:
        raise DocumentStorageError(
            "Unable to list document metadata"
        ) from error

    document_ids = []
    for stored_metadata_path in stored_metadata_paths:
        try:
            document_id = UUID(stored_metadata_path.stem)
        except ValueError as error:
            raise DocumentStorageError(
                "Document metadata has an invalid ID"
            ) from error
        if get_metadata(directory, document_id) is not None:
            document_ids.append(document_id)
    return sorted(document_ids, key=str)


async def save_document(
    directory: Path, maximum_size_bytes: int, upload: UploadFile
) -> DocumentMetadata:
    """Stores an uploaded file and its initial metadata."""
    document_id = uuid4()
    stored_content_path = content_path(directory, document_id)
    temporary_content_path = stored_content_path.with_suffix(".uploading")
    size = 0

    try:
        with temporary_content_path.open("wb") as file:
            while chunk := await upload.read(1024 * 1024):
                if size + len(chunk) > maximum_size_bytes:
                    raise DocumentTooLargeError(
                        "Document exceeds maximum allowed size"
                    )
                size += len(chunk)
                file.write(chunk)

        metadata = DocumentMetadata(
            id=document_id,
            filename=upload.filename or "unnamed",
            content_type=upload.content_type or DocumentContentType.BINARY,
            size=size,
            chunk_ids=[],
        )
        temporary_content_path.replace(stored_content_path)
        write_metadata(directory, metadata)
        return metadata
    except DocumentStorageError:
        _remove_document_files(directory, document_id)
        raise
    except OSError as error:
        _remove_document_files(directory, document_id)
        raise DocumentStorageError(
            "Unable to store uploaded document"
        ) from error
    finally:
        await upload.close()


def set_chunk_ids(
    directory: Path, metadata: DocumentMetadata, chunk_ids: list[str]
) -> DocumentMetadata:
    """Persists vector chunk identifiers for a document."""
    indexed_metadata = metadata.model_copy(update={"chunk_ids": chunk_ids})
    write_metadata(directory, indexed_metadata)
    return indexed_metadata


def delete_document(directory: Path, document_id: UUID) -> bool:
    """Deletes a stored document."""
    if get_metadata(directory, document_id) is None:
        return False

    content_path(directory, document_id).unlink()
    metadata_path(directory, document_id).unlink()
    return True


def _remove_document_files(directory: Path, document_id: UUID) -> None:
    content_path(directory, document_id).with_suffix(".uploading").unlink(
        missing_ok=True
    )
    content_path(directory, document_id).unlink(missing_ok=True)
    metadata_path(directory, document_id).with_suffix(".json.uploading").unlink(
        missing_ok=True
    )
    metadata_path(directory, document_id).unlink(missing_ok=True)
