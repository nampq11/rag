import asyncio
from uuid import UUID

import pytest

from app.models import QueryRequestBody
from app.services.vector_store import search_document_vectors


class RecordingVectorStore:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        *,
        k: int,
        filter: dict[str, str],
    ) -> list[str]:
        self.arguments = {"embedding": embedding, "k": k, "filter": filter}
        return ["matching chunk"]


def test_search_document_vectors_scopes_and_limits_search() -> None:
    vector_store = RecordingVectorStore()
    document_id = UUID("e6c00139-4dfa-4ac9-9d5c-2b399eb13291")

    result = asyncio.run(
        search_document_vectors(
            vector_store=vector_store,
            embedding=[0.1, 0.2],
            document_id=document_id,
            limit=5,
        )
    )

    assert result.documents == ["matching chunk"]
    assert result.duration_seconds >= 0
    assert vector_store.arguments == {
        "embedding": [0.1, 0.2],
        "k": 5,
        "filter": {"document_id": str(document_id)},
    }


@pytest.mark.parametrize("embedding,limit", [([], 4), ([0.1], 0), ([0.1], 21)])
def test_search_document_vectors_rejects_invalid_requests(
    embedding: list[float], limit: int
) -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            search_document_vectors(
                vector_store=RecordingVectorStore(),
                embedding=embedding,
                document_id=UUID("e6c00139-4dfa-4ac9-9d5c-2b399eb13291"),
                limit=limit,
            )
        )


def test_query_request_defaults_to_four_results() -> None:
    query = QueryRequestBody(
        query="Where is the answer?",
        file_id=UUID("e6c00139-4dfa-4ac9-9d5c-2b399eb13291"),
    )

    assert query.limit == 4
