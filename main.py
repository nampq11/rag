"""Application entry point and dependency wiring."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
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
    ollama_base_url,
    upload_directory,
    vector_collection_name,
)
from app.middleware import PUBLIC_PATHS, security_middleware
from app.routes.document_routes import router as documents_router
from app.routes.pgvector_routes import router as pgvector_router
from app.services.documents import initialize_storage
from app.services.vector_store import (
    check_vector_store_health,
    close_vector_store,
    create_vector_store,
    provision_vector_store,
)

engine, vector_store = create_vector_store(
    database_url,
    vector_collection_name,
    embedding_model,
    ollama_base_url,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initializes application dependencies for the process lifetime."""
    await provision_vector_store(engine, vector_store)
    try:
        yield
    finally:
        await close_vector_store(engine)


def get_allowed_origins() -> list[str]:
    """Returns configured allowed CORS origins."""
    allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    return [
        origin.strip()
        for origin in allowed_origins.split(",")
        if origin.strip()
    ]


def configure_application(application: FastAPI) -> None:
    """Stores application-wide dependencies and settings."""
    initialize_storage(upload_directory)
    application.state.engine = engine
    application.state.vector_store = vector_store
    application.state.upload_directory = upload_directory
    application.state.maximum_document_size_bytes = maximum_document_size_bytes
    application.state.document_chunk_size = document_chunk_size
    application.state.document_chunk_overlap = document_chunk_overlap
    application.state.document_maximum_chunks = document_maximum_chunks
    application.state.embedding_batch_size = embedding_batch_size
    application.state.maximum_document_count = maximum_document_count
    application.state.maximum_total_document_size_bytes = (
        maximum_total_document_size_bytes
    )


app = FastAPI(title="RAG API", lifespan=lifespan)
configure_application(app)
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
        await check_vector_store_health(engine)
    except Exception as error:
        logger.exception("PostgreSQL readiness check failed")
        raise HTTPException(
            status_code=503, detail="Database is unavailable"
        ) from error
    return {"status": "ok"}


def add_binary_format_for_file_uploads(schema: object) -> None:
    """Makes FastAPI file schemas render as file pickers in Swagger UI."""
    if isinstance(schema, dict):
        if schema.get("contentMediaType") == "application/octet-stream":
            schema["format"] = "binary"
            del schema["contentMediaType"]
        for value in schema.values():
            add_binary_format_for_file_uploads(value)
    elif isinstance(schema, list):
        for value in schema:
            add_binary_format_for_file_uploads(value)


def custom_openapi() -> dict[str, object]:
    """Describes JWT authentication for protected API operations."""
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title, version=app.version, routes=app.routes
    )
    add_binary_format_for_file_uploads(schema)
    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }
    for path, path_item in schema["paths"].items():
        if path.rstrip("/") in PUBLIC_PATHS:
            continue
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation["security"] = [{"BearerAuth": []}]

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=api_host, port=api_port)
