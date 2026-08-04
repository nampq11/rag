"""PGVector adapter for document chunks."""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class PgVectorDocumentStore:
    """Adapts PGVector to the ingestion vector-store interface."""

    def __init__(
        self,
        database_url: str,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.engine: AsyncEngine = create_async_engine(database_url)
        self._vector_store: PGVector | None = None

    @property
    def vector_store(self) -> PGVector:
        """Returns the initialized PGVector client."""
        if self._vector_store is None:
            embeddings = OpenAIEmbeddings(model=self.embedding_model)
            self._vector_store = PGVector(
                embeddings=embeddings,
                connection=self.engine,
                collection_name=self.collection_name,
                use_jsonb=True,
                create_extension=False,
                async_mode=True,
            )
        return self._vector_store

    async def provision(self) -> None:
        """Initializes the PGVector collection."""
        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    "SELECT pg_advisory_xact_lock("
                    "hashtext('rag-pgvector-extension'))"
                )
            )
            await connection.execute(
                text("CREATE EXTENSION IF NOT EXISTS vector")
            )
        await self.vector_store.__apost_init__()

    async def check_health(self) -> None:
        """Raises when PostgreSQL is unavailable."""
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def close(self) -> None:
        """Releases PostgreSQL connections."""
        await self.engine.dispose()

    async def add_documents(
        self, documents: list[Document], ids: list[str]
    ) -> None:
        """Adds document chunks to the vector store."""
        await self.vector_store.aadd_documents(documents, ids=ids)

    async def delete(self, ids: list[str]) -> None:
        """Deletes a stored document."""
        await self.vector_store.adelete(ids=ids)
