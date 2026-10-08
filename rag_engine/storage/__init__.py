"""Vector database providers used by the RAG engine."""

from importlib import import_module
from typing import Any

__all__ = ("LocalVectorStoreManager", "MilvusStorageProvider")


def __getattr__(name: str) -> Any:
    if name == "LocalVectorStoreManager":
        return import_module(".chroma", __name__).LocalVectorStoreManager
    if name == "MilvusStorageProvider":
        return import_module(".milvus", __name__).MilvusStorageProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
