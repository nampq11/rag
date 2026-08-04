"""Pydantic models for persisted documents and API responses."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Describes a stored document and its vector chunk identifiers."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    filename: str
    content_type: str
    size: int = Field(ge=0)
    chunk_ids: list[str] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    """Represents document metadata returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    content_type: str
    size: int


class DocumentsResponse(BaseModel):
    """Represents a collection of document metadata."""

    documents: list[DocumentResponse]


class DocumentIdsResponse(BaseModel):
    """Represents a collection of document identifiers."""

    ids: list[UUID]


class QueryRequestBody(BaseModel):
    """Carries a natural-language query scoped to a single document."""

    query: str
    file_id: UUID


class QueryResultItem(BaseModel):
    """Represents a retrieved document chunk returned by a query."""

    content: str
    metadata: dict[str, str]


class QueryResponse(BaseModel):
    """Represents the chunks retrieved for a document-scoped query."""

    query: str
    file_id: UUID
    results: list[QueryResultItem]
