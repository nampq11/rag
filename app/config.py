"""Application configuration and logging setup."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL

load_dotenv()


def positive_integer_setting(name: str, default: int) -> int:
    """Reads and validates a positive integer environment setting."""
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


api_host = os.environ.get("API_HOST", "0.0.0.0")
api_port = int(os.environ.get("API_PORT", "8000"))
upload_directory = Path(os.environ.get("UPLOAD_DIRECTORY", "uploads"))
maximum_document_size_bytes = positive_integer_setting(
    "MAXIMUM_DOCUMENT_SIZE_BYTES", 10 * 1024 * 1024
)
maximum_document_count = positive_integer_setting("MAXIMUM_DOCUMENT_COUNT", 10)
maximum_total_document_size_bytes = positive_integer_setting(
    "MAXIMUM_TOTAL_DOCUMENT_SIZE_BYTES", 50 * 1024 * 1024
)

default_database_url = URL.create(
    "postgresql+psycopg",
    username=os.environ.get("POSTGRES_USER", "myuser"),
    password=os.environ.get("POSTGRES_PASSWORD", "mypassword"),
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=positive_integer_setting("POSTGRES_PORT", 5433),
    database=os.environ.get("POSTGRES_DB", "mydatabase"),
).render_as_string(hide_password=False)
database_url = os.environ.get("DATABASE_URL", default_database_url)
vector_collection_name = os.environ.get("VECTOR_COLLECTION_NAME", "documents")
embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
document_chunk_size = positive_integer_setting("DOCUMENT_CHUNK_SIZE", 1000)
document_chunk_overlap = int(os.environ.get("DOCUMENT_CHUNK_OVERLAP", "200"))
document_maximum_chunks = positive_integer_setting(
    "DOCUMENT_MAXIMUM_CHUNKS", 1000
)
embedding_batch_size = positive_integer_setting("EMBEDDING_BATCH_SIZE", 100)
if document_chunk_overlap < 0 or document_chunk_overlap >= document_chunk_size:
    raise ValueError(
        "DOCUMENT_CHUNK_OVERLAP must be non-negative and smaller than "
        "DOCUMENT_CHUNK_SIZE"
    )

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger("rag")
logger.setLevel(log_level)
