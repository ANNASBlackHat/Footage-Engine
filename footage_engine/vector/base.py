"""Vector store abstractions and data models."""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class VectorRecord:
    """Represents a vector payload to index into the vector database."""

    id: str  # Vector ID (e.g. chunk.id or custom vector UUID)
    vector: list[float]
    chunk_id: str
    media_item_id: str
    provider: str
    media_type: str  # "video" or "image"
    duration_sec: Optional[float] = None
    embedding_model: str = "xclip-base-patch32"
    embedding_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """Represents a single match returned from a vector similarity search."""

    id: str
    score: float  # Cosine similarity score
    chunk_id: str
    media_item_id: str
    provider: str
    media_type: str
    duration_sec: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """Protocol for vector indexing backends (Zilliz Cloud, Milvus, In-Memory)."""

    def init_collection(self, collection_name: str, dimension: int = 512) -> None:
        """Ensures the collection and vector index exist."""
        ...

    def upsert(self, collection_name: str, records: list[VectorRecord]) -> list[str]:
        """Inserts or updates vector records. Returns list of vector IDs."""
        ...

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        """Performs approximate nearest neighbor search."""
        ...

    def delete(self, collection_name: str, ids: list[str]) -> bool:
        """Deletes vector records by ID."""
        ...
