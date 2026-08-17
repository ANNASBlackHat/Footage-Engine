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

    @property
    def width(self) -> Optional[int]:
        if self.resolution and "x" in self.resolution:
            try:
                return int(self.resolution.split("x")[0])
            except ValueError:
                pass
        return None

    @property
    def height(self) -> Optional[int]:
        if self.resolution and "x" in self.resolution:
            try:
                return int(self.resolution.split("x")[1])
            except ValueError:
                pass
        return None

    @property
    def orientation(self) -> str:
        """Returns 'horizontal' (landscape, 16:9), 'vertical' (portrait, 9:16/shorts/reels), or 'square' (1:1)."""
        w, h = self.width, self.height
        if not w or not h:
            return "unknown"
        if w > h:
            return "horizontal"
        elif h > w:
            return "vertical"
        return "square"

    @property
    def aspect_ratio(self) -> Optional[str]:
        """Returns standard aspect ratio such as '16:9', '9:16', '1:1', '4:3'."""
        w, h = self.width, self.height
        if not w or not h:
            return None
        import math
        ratio = round(w / h, 2)
        if ratio in (1.77, 1.78):
            return "16:9"
        elif ratio in (0.56, 0.57):
            return "9:16"
        elif ratio == 1.0:
            return "1:1"
        elif ratio in (1.33, 1.34):
            return "4:3"
        gcd = math.gcd(w, h)
        return f"{w // gcd}:{h // gcd}"

    def __repr__(self) -> str:
        return (
            f"<ChunkResult id={self.chunk_id} score={self.score:.3f} "
            f"[{self.start_ts:.1f}s-{self.end_ts if self.end_ts is not None else 'end'}s] "
            f"res={self.resolution or 'unknown'} ({self.orientation}) "
            f"provider={self.provider}>"
        )
