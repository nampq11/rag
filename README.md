# RAG API

FastAPI service that stores uploaded documents locally and indexes their LangChain chunks in PostgreSQL with pgvector.

## Run

1. Copy `.env.example` to `.env` and set `JWT_SECRET_KEY`.
2. Set `OLLAMA_BASE_URL` in `.env` to the URL of an Ollama endpoint that serves the model:

```bash
OLLAMA_BASE_URL=http://localhost:11434
```

The default `EMBEDDING_MODEL` is `hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF`.

3. Start the application and database:

```bash
docker compose up --build
```

PostgreSQL is available on the host port configured by `POSTGRES_PORT` (default `5434`). API startup enables the pgvector extension, provisions the LangChain collection, and fails readiness if the database is unavailable or the application database role cannot create the extension.

## Document ingestion

`POST /documents/embed` saves each upload, then uses LangChain loaders and `RecursiveCharacterTextSplitter` to split it into chunks. Chunks are embedded in bounded `EMBEDDING_BATCH_SIZE` batches and stored in the pgvector collection configured by `VECTOR_COLLECTION_NAME`. Their IDs are persisted with the document metadata, so deletion remains correct if chunking settings change later.

Supported ingestion formats are UTF-8 text and PDF. The original document can still be downloaded with `GET /documents/{document_id}`. Deleting a document removes its matching vector chunks before removing the stored file.

`POST /documents/local/embed` ingests a file already present below `LOCAL_FILES_DIRECTORY`. Send `{"path": "relative/path/to/file.txt"}`. The service rejects absolute paths, traversal outside that directory, and symlinks that resolve outside it; it copies the file into `UPLOAD_DIRECTORY` before indexing so download and deletion behave exactly like uploaded documents.

Configuration and limits are documented in `.env.example`. The API creates a URI-safe `postgresql+asyncpg` URL from separate `POSTGRES_*` settings; production may instead provide a fully URI-encoded `DATABASE_URL` using the same driver scheme.

## Verify

```bash
uv run ruff check .
uv run pytest
```
