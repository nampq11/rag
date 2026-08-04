import asyncio
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

from app.services.documents import DocumentService
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
    monkeypatch.setattr(
        app.state,
        "document_service",
        DocumentService(tmp_path, maximum_size_bytes=1024),
    )


def upload_document(filename: str, content: bytes) -> dict[str, object]:
    response = send_request(
        "POST",
        "/documents/",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        files={"files": (filename, content, "text/plain")},
    )

    assert response.status_code == 201
    return response.json()["documents"][0]


def test_check_is_public() -> None:
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
    monkeypatch.setattr(
        app.state, "document_service", DocumentService(tmp_path, maximum_size_bytes=3)
    )

    response = send_request(
        "POST",
        "/documents/",
        headers={"Authorization": f"Bearer {create_access_token()}"},
        files={"files": ("large.txt", b"four", "text/plain")},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Document exceeds maximum allowed size"}
    assert list(tmp_path.iterdir()) == []


def test_corrupt_metadata_returns_a_storage_error(tmp_path: Path) -> None:
    uploaded_document = upload_document("notes.txt", b"important notes")
    document_id = UUID(uploaded_document["id"])
    (tmp_path / f"{document_id}.json").write_text("{invalid json", encoding="utf-8")

    response = send_request(
        "GET",
        f"/documents/{document_id}",
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Document storage is unavailable"}


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
