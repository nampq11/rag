"""Document-scoped vector retrieval with explicit timing."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol
from uuid import UUID


class VectorStore(Protocol):
    """Defines the vector-store operation used by document retrieval."""

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        *,
        k: int,
        filter: dict[str, str],
    ) -> list[Any]: ...


@dataclass(frozen=True)
class VectorSearchRequest:
    """Defines a document-scoped vector search."""

    embedding: list[float]
    document_id: UUID
    limit: int


@dataclass(frozen=True)
class VectorSearchResult:
    """Contains retrieved documents and vector-search duration."""

    documents: list[Any]
    duration_seconds: float


async def search_document_vectors(
    *,
    vector_store: VectorStore,
    embedding: list[float],
    document_id: UUID,
    limit: int,
) -> VectorSearchResult:
    """Searches one document and measures only the vector-store operation."""
    if not embedding:
        raise ValueError("Embedding must not be empty")
    if limit < 1 or limit > 20:
        raise ValueError("Limit must be between 1 and 20")

    request = VectorSearchRequest(
        embedding=embedding, document_id=document_id, limit=limit
    )
    started_at = perf_counter()
    documents = await vector_store.asimilarity_search_by_vector(
        request.embedding,
        k=request.limit,
        filter={"document_id": str(request.document_id)},
    )
    return VectorSearchResult(
        documents=documents,
        duration_seconds=perf_counter() - started_at,
    )
