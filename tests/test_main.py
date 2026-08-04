import asyncio
import importlib
import os
from datetime import UTC, datetime, timedelta

import httpx
import jwt
import pytest

os.environ.setdefault("CORS_ALLOWED_ORIGINS", "https://frontend.example.com")
os.environ.setdefault(
    "JWT_SECRET_KEY", "test-secret-key-must-be-at-least-32-characters"
)

from main import app


def request(path: str, token: str | None = None) -> httpx.Response:
    async def send_request() -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(path, headers=headers)

    return asyncio.run(send_request())


def create_access_token() -> str:
    return jwt.encode(
        {"sub": "test-user", "exp": datetime.now(UTC) + timedelta(minutes=5)},
        os.environ["JWT_SECRET_KEY"],
        algorithm="HS256",
    )


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/health"])
def test_public_routes_do_not_require_a_jwt(path: str) -> None:
    response = request(path)

    assert response.status_code == 200


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_cors_allows_document_api_methods_from_configured_origin(method: str) -> None:
    async def send_preflight_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.options(
                "/",
                headers={
                    "Origin": "https://frontend.example.com",
                    "Access-Control-Request-Method": method,
                    "Access-Control-Request-Headers": "Authorization",
                },
            )

    response = asyncio.run(send_preflight_request())

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "https://frontend.example.com"
    )


def test_application_routes_require_a_jwt() -> None:
    response = request("/")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_application_routes_accept_a_valid_jwt() -> None:
    response = request("/", create_access_token())

    assert response.status_code == 200
    assert response.json() == {"message": "RAG API is running"}


def test_health_check_returns_ok_without_a_jwt() -> None:
    response = request("/health")

    assert response.json() == {"status": "ok"}


def test_server_configuration_uses_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import config

    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "9000")

    importlib.reload(config)

    assert config.api_host == "127.0.0.1"
    assert config.api_port == 9000
