import os

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.config import api_host, api_port
from app.middleware import security_middleware


def get_allowed_origins() -> list[str]:
    allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    return [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]


app = FastAPI(title="RAG API")
app.middleware("http")(security_middleware)

allowed_origins = get_allowed_origins()
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["Authorization", "Content-Type"],
    )


@app.get("/")
async def health_check() -> dict[str, str]:
    return {"message": "RAG API is running"}


@app.get("/health")
async def health_check_status() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=api_host, port=api_port)
