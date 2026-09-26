"""Base types for the asynchronous job worker."""

from dataclasses import dataclass
from typing import Any, Callable

from footage_engine.config import Settings
from footage_engine.embeddings.base import EmbeddingBackend
from footage_engine.retrieval.api import RetrievalAPI
from footage_engine.storage.base import StorageBackend
from footage_engine.vector.base import VectorStore


@dataclass
class TaskContext:
    """Engine handles shared by every task handler for the worker's lifetime."""

    settings: Settings
    storage: StorageBackend
    embedder: EmbeddingBackend
    vector_store: VectorStore
    retrieval_api: RetrievalAPI
    database_url: str
    backend: str = "xclip"
    worker_id: str = ""


# A handler receives the shared context plus the job's JSON payload and returns
# a JSON-serializable dict that is stored verbatim as the job result.
TaskHandler = Callable[[TaskContext, dict[str, Any]], dict[str, Any]]


__all__ = ["TaskContext", "TaskHandler"]
