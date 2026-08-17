"""Data models for retrieval API results and filters."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class SearchFilters:
    """Filter options for footage chunk retrieval."""

    media_type: Optional[str] = None  # "video" or "image"
    provider: Optional[str] = None  # "pexels", "pixabay", etc.
    license_type: Optional[str] = None
    min_duration_sec: Optional[float] = None
    max_duration_sec: Optional[float] = None
    tags: Optional[list[str]] = None

    def to_milvus_expr(self) -> str:
        """Converts filters into a Milvus/Zilliz scalar boolean expression."""
        clauses = []
        if self.media_type:
            clauses.append(f'media_type == "{self.media_type}"')
        if self.provider:
            clauses.append(f'provider == "{self.provider}"')
        if self.min_duration_sec is not None:
            clauses.append(f"duration_sec >= {float(self.min_duration_sec)}")
        if self.max_duration_sec is not None:
            clauses.append(f"duration_sec <= {float(self.max_duration_sec)}")
        return " and ".join(clauses) if clauses else ""


@dataclass
class ChunkResult:
    """Hydrated, ready-to-use search result representing a retrievable footage chunk."""

    chunk_id: str
    media_item_id: str
    score: float
    start_ts: float
    end_ts: Optional[float]
    duration_sec: Optional[float]
    media_type: str
    provider: str
    source_url: str
    storage_path: str
    storage_url: str
    resolution: Optional[str] = None
    license_type: str = "unknown"
    caption: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    item_metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<ChunkResult id={self.chunk_id} score={self.score:.3f} "
            f"[{self.start_ts:.1f}s-{self.end_ts if self.end_ts is not None else 'end'}s] "
            f"provider={self.provider}>"
        )
