"""Local and hybrid retrieval implementations."""

from importlib import import_module
from typing import Any

__all__ = ("HybridSearchOrchestrator", "LocalQueryEngine")


def __getattr__(name: str) -> Any:
    if name == "HybridSearchOrchestrator":
        return import_module(".hybrid", __name__).HybridSearchOrchestrator
    if name == "LocalQueryEngine":
        return import_module(".local", __name__).LocalQueryEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
