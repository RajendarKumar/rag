"""Composable retrieval-augmented generation components."""

from importlib import import_module
from typing import Any

__all__ = (
    "validate_environment",
    "LocalVectorStoreManager",
    "LocalIngestWorker",
    "LocalQueryEngine",
    "MilvusStorageProvider",
    "HybridSearchOrchestrator",
)

_EXPORTS = {
    "validate_environment": (".config", "validate_environment"),
    "LocalVectorStoreManager": (".storage.chroma", "LocalVectorStoreManager"),
    "LocalIngestWorker": (".ingestion", "LocalIngestWorker"),
    "LocalQueryEngine": (".retrieval.local", "LocalQueryEngine"),
    "MilvusStorageProvider": (".storage.milvus", "MilvusStorageProvider"),
    "HybridSearchOrchestrator": (".retrieval.hybrid", "HybridSearchOrchestrator"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name, __name__), attribute_name)