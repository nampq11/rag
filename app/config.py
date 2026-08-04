import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

api_host = os.environ.get("API_HOST", "0.0.0.0")
api_port = int(os.environ.get("API_PORT", "8000"))
upload_directory = Path(os.environ.get("UPLOAD_DIRECTORY", "uploads"))
maximum_document_size_bytes = int(
    os.environ.get("MAXIMUM_DOCUMENT_SIZE_BYTES", str(10 * 1024 * 1024))
)

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger("rag")
logger.setLevel(log_level)
