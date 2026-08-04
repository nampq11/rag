import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile


@dataclass(frozen=True)
class DocumentMetadata:
    id: UUID
    filename: str
    content_type: str
    size: int


class DocumentStorageError(RuntimeError):
    pass


class DocumentTooLargeError(DocumentStorageError):
    pass


class DocumentService:
    def __init__(self, directory: Path, maximum_size_bytes: int) -> None:
        self.directory = directory
        self.maximum_size_bytes = maximum_size_bytes
        self.directory.mkdir(parents=True, exist_ok=True)

    def content_path(self, document_id: UUID) -> Path:
        return self.directory / str(document_id)

    def metadata_path(self, document_id: UUID) -> Path:
        return self.directory / f"{document_id}.json"

    def get_metadata(self, document_id: UUID) -> DocumentMetadata | None:
        metadata_path = self.metadata_path(document_id)
        content_path = self.content_path(document_id)
        if not metadata_path.is_file():
            return None
        if not content_path.is_file():
            raise DocumentStorageError("Document content is missing")

        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            return DocumentMetadata(
                id=UUID(data["id"]),
                filename=data["filename"],
                content_type=data["content_type"],
                size=data["size"],
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise DocumentStorageError("Unable to read document metadata") from error

    def get_all_ids(self) -> list[UUID]:
        try:
            metadata_paths = list(self.directory.glob("*.json"))
        except OSError as error:
            raise DocumentStorageError("Unable to list document metadata") from error

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
                content_type=upload.content_type or "application/octet-stream",
                size=size,
            )
            temporary_metadata_path = self.metadata_path(document_id).with_suffix(
                ".json.uploading"
            )
            temporary_metadata_path.write_text(
                json.dumps(asdict(metadata), default=str), encoding="utf-8"
            )
            temporary_content_path.replace(content_path)
            temporary_metadata_path.replace(self.metadata_path(document_id))
            return metadata
        except DocumentStorageError:
            temporary_content_path.unlink(missing_ok=True)
            content_path.unlink(missing_ok=True)
            self.metadata_path(document_id).with_suffix(".json.uploading").unlink(
                missing_ok=True
            )
            self.metadata_path(document_id).unlink(missing_ok=True)
            raise
        except OSError as error:
            temporary_content_path.unlink(missing_ok=True)
            content_path.unlink(missing_ok=True)
            self.metadata_path(document_id).with_suffix(".json.uploading").unlink(
                missing_ok=True
            )
            self.metadata_path(document_id).unlink(missing_ok=True)
            raise DocumentStorageError("Unable to store uploaded document") from error
        finally:
            await upload.close()

    def delete(self, document_id: UUID) -> bool:
        if self.get_metadata(document_id) is None:
            return False

        self.content_path(document_id).unlink()
        self.metadata_path(document_id).unlink()
        return True
