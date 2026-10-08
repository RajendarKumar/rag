import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PACKAGE_DIR.parent

DB_DIR = Path(os.getenv("RAG_DB_DIR", str(WORKSPACE_DIR / "chroma_db"))).expanduser()
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(WORKSPACE_DIR / "data"))).expanduser()

PARENT_CHUNK_SIZE = 800
PARENT_CHUNK_OVERLAP = 50

CHILD_CHUNK_SIZE = 150
CHILD_CHUNK_OVERLAP = 20

EMBEDDING_MODEL_NAME = "gemini-embedding-001"
FILE_EXTENSION_GLOB = "*.pdf"
TEXT_FILE_EXTENSION_GLOB = "*.txt"


def validate_environment() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("CRITICAL: GOOGLE_API_KEY environment variable is missing.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)