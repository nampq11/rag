"""PGVector adapter for document chunks."""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector


class PgVectorDocumentStore:
    """Adapts PGVector to the ingestion vector-store interface."""

    def __init__(
        self,
        database_url: str,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        self.database_url = database_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._vector_store: PGVector | None = None

    @property
    def vector_store(self) -> PGVector:
        """Returns the initialized PGVector client."""
        if self._vector_store is None:
            embeddings = OpenAIEmbeddings(model=self.embedding_model)
            self._vector_store = PGVector(
                embeddings=embeddings,
                connection=self.database_url,
                collection_name=self.collection_name,
                use_jsonb=True,
                create_extension=True,
            )
        return self._vector_store

    def provision(self) -> None:
        """Initializes the PGVector collection."""
        _ = self.vector_store

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        """Adds document chunks to the vector store."""
        self.vector_store.add_documents(documents, ids=ids)

    def delete(self, ids: list[str]) -> None:
        """Deletes a stored document."""
        self.vector_store.delete(ids=ids)
