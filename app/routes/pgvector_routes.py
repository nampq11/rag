"""Authenticated PostgreSQL diagnostics for pgvector development."""

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

router = APIRouter(tags=["pgvector diagnostics"])
SUPPORTED_TABLES = frozenset(
    {"langchain_pg_collection", "langchain_pg_embedding"}
)


def get_engine(request: Request) -> AsyncEngine:
    """Returns the application's pgvector database engine."""
    return request.app.state.vector_store.engine


def validate_table_name(table_name: str) -> str:
    """Returns a pgvector table name approved for diagnostic reads."""
    if table_name not in SUPPORTED_TABLES:
        raise HTTPException(
            status_code=400, detail="Unsupported pgvector table"
        )
    return table_name


async def check_index_exists(
    engine: AsyncEngine,
    table_name: str,
    column_name: str,
    schema: str,
) -> bool:
    """Checks whether an index definition references the requested column."""
    statement = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = :schema
              AND tablename = :table_name
              AND indexdef LIKE '%' || :column_name || '%'
        )
        """
    )
    async with engine.connect() as connection:
        result = await connection.execute(
            statement,
            {
                "schema": schema,
                "table_name": table_name,
                "column_name": column_name,
            },
        )
    return bool(result.scalar_one())


@router.get("/test/check_index")
async def check_index(
    request: Request,
    table_name: str,
    column_name: str,
    schema: str = "public",
) -> dict[str, str]:
    """Reports whether an index references the requested column expression."""
    exists = await check_index_exists(
        get_engine(request), table_name, column_name, schema
    )
    if not exists:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No index on {column_name} found in table "
                f"{schema}.{table_name}."
            ),
        )
    return {
        "message": f"Index on {column_name} exists in table {schema}.{table_name}."
    }


@router.get("/db/tables")
async def get_table_names(
    request: Request,
    schema: str = "public",
) -> dict[str, list[str] | str]:
    """Lists tables in a PostgreSQL schema."""
    statement = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
        ORDER BY table_name
        """
    )
    async with get_engine(request).connect() as connection:
        result = await connection.execute(statement, {"schema": schema})
    return {"schema": schema, "tables": list(result.scalars())}


@router.get("/db/tables/columns")
async def get_table_columns(
    request: Request,
    table_name: str,
    schema: str = "public",
) -> dict[str, list[str] | str]:
    """Lists columns of a PostgreSQL table."""
    statement = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table_name
        ORDER BY ordinal_position
        """
    )
    async with get_engine(request).connect() as connection:
        result = await connection.execute(
            statement,
            {"schema": schema, "table_name": table_name},
        )
    return {
        "schema": schema,
        "table_name": table_name,
        "columns": list(result.scalars()),
    }


@router.get("/records/all")
async def get_all_records(
    request: Request,
    table_name: str = "langchain_pg_embedding",
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    """Returns a bounded page of records from a pgvector table."""
    table_name = validate_table_name(table_name)
    statement = text(
        f"SELECT * FROM {table_name} ORDER BY 1 LIMIT :limit OFFSET :offset"
    )
    async with get_engine(request).connect() as connection:
        result = await connection.execute(
            statement, {"limit": limit, "offset": offset}
        )
    return {
        "table_name": table_name,
        "limit": limit,
        "offset": offset,
        "records": [dict(record) for record in result.mappings()],
    }


@router.get("/records")
async def get_records_by_id(
    request: Request,
    record_id: str,
) -> dict[str, object]:
    """Returns embedding records matching their LangChain PGVector ID."""
    statement = text(
        "SELECT * FROM langchain_pg_embedding WHERE id = :record_id"
    )
    async with get_engine(request).connect() as connection:
        result = await connection.execute(statement, {"record_id": record_id})
    return {
        "record_id": record_id,
        "records": [dict(record) for record in result.mappings()],
    }
