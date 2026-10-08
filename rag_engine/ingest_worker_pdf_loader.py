"""Compatibility exports for the local PDF ingestion worker."""

from .ingestion.pdf_loader import LocalIngestWorker

__all__ = ["LocalIngestWorker"]
