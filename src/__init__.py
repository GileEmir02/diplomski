"""Shared ingestion API, loaded lazily to support ``python -m src.ingestion``."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ingestion import (
        BatchImportError,
        BatchResult,
        DocumentInput,
        DocumentReadError,
        ImportIssue,
        IngestedDocument,
        PageRecord,
        clean_text,
        default_manifest_path,
        ingest_batch,
        load_manifest_inputs,
        read_document,
        write_pages_jsonl,
    )

__all__ = [
    "DocumentInput",
    "PageRecord",
    "IngestedDocument",
    "ImportIssue",
    "DocumentReadError",
    "BatchImportError",
    "BatchResult",
    "clean_text",
    "default_manifest_path",
    "read_document",
    "ingest_batch",
    "load_manifest_inputs",
    "write_pages_jsonl",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        return getattr(import_module(".ingestion", __name__), name)
    raise AttributeError(f"Modul {__name__!r} nema atribut {name!r}.")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
