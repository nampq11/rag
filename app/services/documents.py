"""Filesystem-backed document storage service."""

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.constants import DocumentContentType
from app.models import DocumentMetadata


class DocumentStorageError(RuntimeError):
    """Indicates a document storage operation failed."""


class DocumentTooLargeError(DocumentStorageError):
    """Indicates an upload exceeded the configured size limit."""


class DocumentService:
    """Stores document content and metadata on the filesystem."""

    def __init__(self, directory: Path, maximum_size_bytes: int) -> None:
        self.directory = directory
        self.maximum_size_bytes = maximum_size_bytes
        self.directory.mkdir(parents=True, exist_ok=True)

    def content_path(self, document_id: UUID) -> Path:
        """Returns the filesystem path for document content."""
        return self.directory / str(document_id)

    def metadata_path(self, document_id: UUID) -> Path:
        """Returns the filesystem path for document metadata."""
        return self.directory / f"{document_id}.json"

    def write_metadata(self, metadata: DocumentMetadata) -> None:
        """Atomically writes document metadata."""
        temporary_metadata_path = self.metadata_path(metadata.id).with_suffix(
            ".json.uploading"
        )
        try:
            temporary_metadata_path.write_text(
                metadata.model_dump_json(), encoding="utf-8"
            )
            temporary_metadata_path.replace(self.metadata_path(metadata.id))
        except OSError as error:
            temporary_metadata_path.unlink(missing_ok=True)
            raise DocumentStorageError(
                "Unable to store document metadata"
            ) from error

    def get_metadata(self, document_id: UUID) -> DocumentMetadata | None:
        """Loads metadata for a stored document when it exists."""
        metadata_path = self.metadata_path(document_id)
        content_path = self.content_path(document_id)
        if not metadata_path.is_file():
            return None
        if not content_path.is_file():
            raise DocumentStorageError("Document content is missing")

        try:
            return DocumentMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise DocumentStorageError(
                "Unable to read document metadata"
            ) from error

    def get_all_ids(self) -> list[UUID]:
        """Returns identifiers for all valid stored documents."""
        try:
            metadata_paths = list(self.directory.glob("*.json"))
        except OSError as error:
            raise DocumentStorageError(
                "Unable to list document metadata"
            ) from error

        document_ids = []
        for metadata_path in metadata_paths:
            try:
                document_id = UUID(metadata_path.stem)
            except ValueError as error:
                raise DocumentStorageError(
                    "Document metadata has an invalid ID"
                ) from error
            if self.get_metadata(document_id) is not None:
                document_ids.append(document_id)
        return sorted(document_ids, key=str)

    async def save(self, upload: UploadFile) -> DocumentMetadata:
        """Stores an uploaded file and its initial metadata."""
        document_id = uuid4()
        content_path = self.content_path(document_id)
        temporary_content_path = content_path.with_suffix(".uploading")
        size = 0

        try:
            with temporary_content_path.open("wb") as file:
                while chunk := await upload.read(1024 * 1024):
                    if size + len(chunk) > self.maximum_size_bytes:
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
            temporary_content_path.replace(content_path)
            self.write_metadata(metadata)
            return metadata
        except DocumentStorageError:
            temporary_content_path.unlink(missing_ok=True)
            content_path.unlink(missing_ok=True)
            self.metadata_path(document_id).with_suffix(
                ".json.uploading"
            ).unlink(missing_ok=True)
            self.metadata_path(document_id).unlink(missing_ok=True)
            raise
        except OSError as error:
            temporary_content_path.unlink(missing_ok=True)
            content_path.unlink(missing_ok=True)
            self.metadata_path(document_id).with_suffix(
                ".json.uploading"
            ).unlink(missing_ok=True)
            self.metadata_path(document_id).unlink(missing_ok=True)
            raise DocumentStorageError(
                "Unable to store uploaded document"
            ) from error
        finally:
            await upload.close()

    def set_chunk_ids(
        self, metadata: DocumentMetadata, chunk_ids: list[str]
    ) -> DocumentMetadata:
        """Persists vector chunk identifiers for a document."""
        indexed_metadata = metadata.model_copy(update={"chunk_ids": chunk_ids})
        self.write_metadata(indexed_metadata)
        return indexed_metadata

    def delete(self, document_id: UUID) -> bool:
        """Deletes a stored document."""
        if self.get_metadata(document_id) is None:
            return False

        self.content_path(document_id).unlink()
        self.metadata_path(document_id).unlink()
        return True
