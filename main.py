"""Application entry point and dependency wiring."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.config import (
    api_host,
    api_port,
    database_url,
    document_chunk_overlap,
    document_chunk_size,
    document_maximum_chunks,
    embedding_batch_size,
    embedding_model,
    maximum_document_count,
    maximum_document_size_bytes,
    maximum_total_document_size_bytes,
    upload_directory,
    vector_collection_name,
)
from app.middleware import security_middleware
from app.routes.document_routes import router as documents_router
from app.services.documents import DocumentService
from app.services.ingestion import DocumentIngestionService
from app.services.vector_store import PgVectorDocumentStore

vector_store = PgVectorDocumentStore(
    database_url, vector_collection_name, embedding_model
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initializes application dependencies for the process lifetime."""
    await asyncio.to_thread(vector_store.provision)
    yield


def get_allowed_origins() -> list[str]:
    """Returns configured allowed CORS origins."""
    allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    return [
        origin.strip()
        for origin in allowed_origins.split(",")
        if origin.strip()
    ]


app = FastAPI(title="RAG API", lifespan=lifespan)
app.state.document_service = DocumentService(
    upload_directory, maximum_document_size_bytes
)
app.state.document_ingestion_service = DocumentIngestionService(
    vector_store,
    document_chunk_size,
    document_chunk_overlap,
    document_maximum_chunks,
    embedding_batch_size,
)
app.state.maximum_document_count = maximum_document_count
app.state.maximum_total_document_size_bytes = maximum_total_document_size_bytes
app.middleware("http")(security_middleware)
app.include_router(documents_router)

allowed_origins = get_allowed_origins()
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )


@app.get("/")
async def health_check() -> dict[str, str]:
    """Returns the API health message."""
    return {"message": "RAG API is running"}


@app.get("/health")
async def health_check_status() -> dict[str, str]:
    """Returns the API health status."""
    return {"status": "ok"}


@app.get("/check/")
async def check() -> dict[str, str]:
    """Returns the public readiness status."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=api_host, port=api_port)
