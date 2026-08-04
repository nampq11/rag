"""PGVector setup and lifecycle functions."""

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_vector_store(
    database_url: str,
    collection_name: str,
    embedding_model: str,
    ollama_base_url: str,
) -> tuple[AsyncEngine, PGVector]:
    """Creates the PostgreSQL engine and its PGVector client."""
    engine = create_async_engine(database_url)
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_base_url,
    )
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
