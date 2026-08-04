import asyncio

import httpx

from main import app


def test_health_check() -> None:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get("/")

    response = asyncio.run(send_request())

    assert response.status_code == 200
    assert response.json() == {"message": "RAG API is running"}
