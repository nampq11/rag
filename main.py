"""Application entry point and dependency wiring."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
    logger,
    maximum_document_count,
    maximum_document_size_bytes,
    maximum_total_document_size_bytes,
    upload_directory,
    vector_collection_name,
)
from app.middleware import security_middleware
from app.routes.document_routes import router as documents_router
from app.routes.pgvector_routes import router as pgvector_router
from app.services.documents import DocumentService
from app.services.ingestion import DocumentIngestionService
from app.services.vector_store import PgVectorDocumentStore

vector_store = PgVectorDocumentStore(
    database_url, vector_collection_name, embedding_model
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initializes application dependencies for the process lifetime."""
    await vector_store.provision()
    try:
        yield
    finally:
        await vector_store.close()


def get_allowed_origins() -> list[str]:
    """Returns configured allowed CORS origins."""
    allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    return [
        origin.strip()
        for origin in allowed_origins.split(",")
        if origin.strip()
    ]


app = FastAPI(title="RAG API", lifespan=lifespan)
app.state.vector_store = vector_store
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
app.include_router(pgvector_router)

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
    """Returns readiness based on PostgreSQL availability."""
    try:
        await vector_store.check_health()
    except Exception as error:
        logger.exception("PostgreSQL readiness check failed")
        raise HTTPException(
            status_code=503, detail="Database is unavailable"
        ) from error
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=api_host, port=api_port)
