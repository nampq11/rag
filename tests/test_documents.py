import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import httpx
import jwt
import pytest

os.environ.setdefault("CORS_ALLOWED_ORIGINS", "https://frontend.example.com")
os.environ.setdefault(
    "JWT_SECRET_KEY", "test-secret-key-must-be-at-least-32-characters"
)

from app.services.documents import DocumentStorageError, get_metadata


class InMemoryVectorStore:
    async def aadd_documents(self, *_: object, **__: object) -> None:
        pass

    async def adelete(self, *_: object, **__: object) -> None:
        pass


class FailingVectorStore(InMemoryVectorStore):
    async def aadd_documents(self, *_: object, **__: object) -> None:
        raise RuntimeError("Database unavailable")


class PersistingVectorStore:
    def __init__(self) -> None:
        self.deleted_ids: list[str] = []

    async def aadd_documents(self, *_: object, **__: object) -> None:
        pass

    async def adelete(self, ids: list[str]) -> None:
        self.deleted_ids = ids


class SecondInsertionFailsVectorStore:
    def __init__(self) -> None:
        self.insertion_count = 0
        self.deleted_chunk_ids: list[list[str]] = []

    async def aadd_documents(self, *_: object, **__: object) -> None:
        self.insertion_count += 1
        if self.insertion_count == 2:
            raise RuntimeError("Database unavailable")

    async def adelete(self, ids: list[str]) -> None:
        self.deleted_chunk_ids.append(ids)


class FailingDeleteVectorStore(PersistingVectorStore):
    async def adelete(self, *_: object, **__: object) -> None:
        raise RuntimeError("Database unavailable")


from main import app


def create_access_token() -> str:
    return jwt.encode(
        {"sub": "test-user", "exp": datetime.now(UTC) + timedelta(minutes=5)},
        os.environ["JWT_SECRET_KEY"],
        algorithm="HS256",
    )


def send_request(
    method: str,
    path: str,
    **kwargs: object,
) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(request())


@pytest.fixture(autouse=True)
def document_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_files_directory = tmp_path.parent / f"{tmp_path.name}-local-files"
    local_files_directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app.state, "upload_directory", tmp_path)
    monkeypatch.setattr(
        app.state, "local_files_directory", local_files_directory
    )
    monkeypatch.setattr(app.state, "maximum_document_size_bytes", 1024)
    monkeypatch.setattr(app.state, "vector_store", InMemoryVectorStore())


def upload_document(filename: str, content: bytes) -> dict[str, object]:
    response = send_request(
        "POST",
        "/documents/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        files={"files": (filename, content, "text/plain")},
    )

    assert response.status_code == 201
    return response.json()["documents"][0]


def embed_local_file(
    path: str, entity_id: str | None = None
) -> dict[str, object]:
    params = {"entity_id": entity_id} if entity_id is not None else None
    response = send_request(
        "POST",
        "/documents/local/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        json={"path": path},
        params=params,
    )

    assert response.status_code == 201
    return response.json()["documents"][0]


def test_pgvector_diagnostic_routes_require_a_jwt() -> None:
    response = send_request("GET", "/db/tables")

    assert response.status_code == 401


def test_pgvector_records_reject_unknown_tables() -> None:
    response = send_request(
        "GET",
        "/records/all?table_name=unknown_table",
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported pgvector table"}


def test_check_is_public(monkeypatch: pytest.MonkeyPatch) -> None:
    async def successful_health_check(_: object) -> None:
        pass

    monkeypatch.setattr(
        "main.check_vector_store_health", successful_health_check
    )

    response = send_request("GET", "/check/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_get_and_delete_document(tmp_path: Path) -> None:
    uploaded_document = upload_document("notes.txt", b"important notes")
    document_id = uploaded_document["id"]

    assert UUID(document_id)
    assert uploaded_document == {
        "id": document_id,
        "filename": "notes.txt",
        "content_type": "text/plain",
        "size": 15,
    }

    get_response = send_request(
        "GET",
        f"/documents/{document_id}",
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )
    assert get_response.status_code == 200
    assert get_response.content == b"important notes"
    assert get_response.headers["content-type"].startswith("text/plain")
    assert 'filename="notes.txt"' in get_response.headers["content-disposition"]

    delete_response = send_request(
        "DELETE",
        f"/documents/{document_id}",
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )
    assert delete_response.status_code == 204
    assert list(tmp_path.iterdir()) == []

    missing_response = send_request(
        "GET",
        f"/documents/{document_id}",
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )
    assert missing_response.status_code == 404


def test_upload_rejects_documents_larger_than_configured_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state, "maximum_document_size_bytes", 3)

    response = send_request(
        "POST",
        "/documents/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        files={"files": ("large.txt", b"four", "text/plain")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Document exceeds maximum allowed size"
    }
    assert list(tmp_path.iterdir()) == []


def test_upload_removes_file_when_document_ingestion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state, "vector_store", FailingVectorStore())

    response = send_request(
        "POST",
        "/documents/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        files={"files": ("notes.txt", b"important notes", "text/plain")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Document ingestion is unavailable"}
    assert list(tmp_path.iterdir()) == []


def test_upload_rolls_back_preceding_documents_when_later_ingestion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vector_store = SecondInsertionFailsVectorStore()
    monkeypatch.setattr(app.state, "vector_store", vector_store)

    response = send_request(
        "POST",
        "/documents/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        files=[
            ("files", ("first.txt", b"first", "text/plain")),
            ("files", ("second.txt", b"second", "text/plain")),
        ],
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Document ingestion is unavailable"}
    assert [len(chunk_ids) for chunk_ids in vector_store.deleted_chunk_ids] == [
        1,
        1,
    ]
    assert list(tmp_path.iterdir()) == []


def test_upload_removes_local_file_when_vector_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state, "vector_store", FailingDeleteVectorStore())

    def fail_to_persist_chunk_ids(*_: object) -> None:
        raise DocumentStorageError("Unable to store document metadata")

    monkeypatch.setattr(
        "app.routes.document_routes.set_chunk_ids", fail_to_persist_chunk_ids
    )

    response = send_request(
        "POST",
        "/documents/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        files={"files": ("notes.txt", b"important notes", "text/plain")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Document storage is unavailable"}
    assert list(tmp_path.iterdir()) == []


def test_upload_rejects_too_many_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app.state, "maximum_document_count", 1)

    response = send_request(
        "POST",
        "/documents/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        files=[
            ("files", ("first.txt", b"first", "text/plain")),
            ("files", ("second.txt", b"second", "text/plain")),
        ],
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Too many documents in one upload"}


def test_upload_rejects_documents_exceeding_total_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app.state, "maximum_total_document_size_bytes", 3)

    response = send_request(
        "POST",
        "/documents/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        files={"files": ("large.txt", b"four", "text/plain")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Documents exceed maximum total allowed size"
    }


def test_delete_uses_chunk_ids_persisted_during_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_store = PersistingVectorStore()
    monkeypatch.setattr(app.state, "vector_store", vector_store)

    uploaded_document = upload_document("notes.txt", b"important notes")
    document_id = UUID(uploaded_document["id"])
    metadata = get_metadata(app.state.upload_directory, document_id)

    assert metadata is not None
    assert metadata.chunk_ids == [f"{document_id}:0"]

    response = send_request(
        "DELETE",
        f"/documents/{document_id}",
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )

    assert response.status_code == 204
    assert vector_store.deleted_ids == [f"{document_id}:0"]


def test_corrupt_metadata_returns_a_storage_error(tmp_path: Path) -> None:
    uploaded_document = upload_document("notes.txt", b"important notes")
    document_id = UUID(uploaded_document["id"])
    (tmp_path / f"{document_id}.json").write_text(
        "{invalid json", encoding="utf-8"
    )

    response = send_request(
        "GET",
        f"/documents/{document_id}",
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Document storage is unavailable"}


def test_legacy_document_metadata_defaults_to_no_chunk_ids(
    tmp_path: Path,
) -> None:
    uploaded_document = upload_document("notes.txt", b"important notes")
    document_id = UUID(uploaded_document["id"])
    metadata_path = tmp_path / f"{document_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["chunk_ids"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded_metadata = get_metadata(app.state.upload_directory, document_id)

    assert loaded_metadata is not None
    assert loaded_metadata.chunk_ids == []


def test_get_all_ids_returns_all_uploaded_document_ids() -> None:
    first_document = upload_document("first.txt", b"first")
    second_document = upload_document("second.txt", b"second")

    response = send_request(
        "GET",
        "/documents/ids",
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ids": sorted([first_document["id"], second_document["id"]])
    }


def test_get_documents_by_ids_returns_metadata_in_requested_order() -> None:
    first_document = upload_document("first.txt", b"first")
    second_document = upload_document("second.txt", b"second")

    response = send_request(
        "GET",
        "/documents/",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        params=[("ids", second_document["id"]), ("ids", first_document["id"])],
    )

    assert response.status_code == 200
    assert response.json() == {
        "documents": [
            second_document,
            first_document,
        ]
    }


def test_get_documents_by_ids_returns_not_found_for_unknown_id() -> None:
    response = send_request(
        "GET",
        "/documents/",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        params={"ids": "00000000-0000-0000-0000-000000000000"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}


def test_embed_local_file_stores_and_indexes_a_server_file() -> None:
    local_files_directory = app.state.local_files_directory
    (local_files_directory / "notes.txt").write_bytes(b"important notes")

    embedded_document = embed_local_file("notes.txt")
    document_id = embedded_document["id"]

    assert UUID(document_id)
    assert embedded_document == {
        "id": document_id,
        "filename": "notes.txt",
        "content_type": "text/plain",
        "size": 15,
    }

    metadata = get_metadata(app.state.upload_directory, UUID(document_id))
    assert metadata is not None
    assert metadata.chunk_ids == [f"{document_id}:0"]

    get_response = send_request(
        "GET",
        f"/documents/{document_id}",
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )
    assert get_response.status_code == 200
    assert get_response.content == b"important notes"


def test_embed_local_file_returns_not_found_for_missing_path() -> None:
    response = send_request(
        "POST",
        "/documents/local/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        json={"path": "missing.txt"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Local document not found"}


@pytest.mark.parametrize("traversal_path", ["/etc/passwd", "../../etc/passwd"])
def test_embed_local_file_rejects_paths_outside_allowed_directory(
    traversal_path: str,
) -> None:
    response = send_request(
        "POST",
        "/documents/local/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        json={"path": traversal_path},
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Local document is outside the allowed directory"
    }


def test_embed_local_file_rejects_symlinks_outside_allowed_directory(
    tmp_path: Path,
) -> None:
    outside_file = tmp_path / "outside.txt"
    outside_file.write_bytes(b"private notes")
    (app.state.local_files_directory / "outside.txt").symlink_to(outside_file)

    response = send_request(
        "POST",
        "/documents/local/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        json={"path": "outside.txt"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Local document is outside the allowed directory"
    }


def test_embed_local_file_rejects_documents_larger_than_configured_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state, "maximum_document_size_bytes", 3)
    (app.state.local_files_directory / "large.txt").write_bytes(b"four")

    response = send_request(
        "POST",
        "/documents/local/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        json={"path": "large.txt"},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Document exceeds maximum allowed size"
    }
    assert list(tmp_path.iterdir()) == []


def test_embed_local_file_removes_stored_copy_when_ingestion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state, "vector_store", FailingVectorStore())
    (app.state.local_files_directory / "notes.txt").write_bytes(
        b"important notes"
    )

    response = send_request(
        "POST",
        "/documents/local/embed",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        json={"path": "notes.txt"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Document ingestion is unavailable"}
    assert list(tmp_path.iterdir()) == []


def test_embed_local_file_accepts_entity_id_query_parameter() -> None:
    (app.state.local_files_directory / "notes.txt").write_bytes(
        b"important notes"
    )

    embedded_document = embed_local_file("notes.txt", entity_id="tenant-42")

    assert embedded_document["filename"] == "notes.txt"


def test_infer_content_type_returns_pdf_for_pdf_extension() -> None:
    from app.utils.document_loader import infer_content_type

    assert infer_content_type(Path("report.pdf")) == "application/pdf"
    assert infer_content_type(Path("REPORT.PDF")) == "application/pdf"
    assert infer_content_type(Path("notes.txt")) == "text/plain"
    assert infer_content_type(Path("data")) == "text/plain"
