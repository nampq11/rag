"""PGVector setup and lifecycle functions."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# Assigned once by create_vector_store at startup. The process owns a single
# embeddings model, so the cache below can key on the query text alone.
_query_embeddings: Embeddings | None = None


@lru_cache(maxsize=128)
def get_cached_query_embedding(query: str) -> list[float]:
    """Returns the embedding for a query, reusing the 128 most recent results.

    Callers must run this in a worker thread (``embed_query`` blocks on the
    Ollama HTTP call) and must not mutate the returned list, since every cache
    hit returns the same shared object.
    """
    embeddings = _query_embeddings
    if embeddings is None:
        raise RuntimeError(
            "Query embeddings are not initialized; "
            "create_vector_store must run before any query"
        )
    return embeddings.embed_query(query)


def create_vector_store(
    database_url: str,
    collection_name: str,
    embedding_model: str,
    ollama_base_url: str,
) -> tuple[AsyncEngine, PGVector]:
    """Creates the PostgreSQL engine and its PGVector client."""
    global _query_embeddings
    engine = create_async_engine(database_url)
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_base_url,
    )
    _query_embeddings = embeddings
    vector_store = PGVector(
        embeddings=embeddings,
        connection=engine,
        collection_name=collection_name,
        use_jsonb=True,
        create_extension=False,
        async_mode=True,
    )
    return engine, vector_store


async def provision_vector_store(
    engine: AsyncEngine, vector_store: PGVector
) -> None:
    """Initializes the pgvector extension and collection."""
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "SELECT pg_advisory_xact_lock("
                "hashtext('rag-pgvector-extension'))"
            )
        )
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await vector_store.__apost_init__()


async def check_vector_store_health(engine: AsyncEngine) -> None:
    """Raises when PostgreSQL is unavailable."""
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def close_vector_store(engine: AsyncEngine) -> None:
    """Releases PostgreSQL connections."""
    await engine.dispose()
