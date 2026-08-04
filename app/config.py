import logging
import os

from dotenv import load_dotenv

load_dotenv()

api_host = os.environ.get("API_HOST", "0.0.0.0")
api_port = int(os.environ.get("API_PORT", "8000"))

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger("rag")
logger.setLevel(log_level)
