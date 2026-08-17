"""Base classes and dataclasses for Ingestion Sources."""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class Candidate:
    """Standardized representation of a media asset candidate returned from any provider."""

    provider: str
    source_url: str
    source_id: Optional[str] = None
    license_type: str = "unknown"
    media_type: str = "video"  # "video" or "image"
    duration_sec: Optional[float] = None
    resolution: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceAdapter(Protocol):
    """Protocol for provider search adapters (Pixabay, Pexels, Coverr, etc.)."""

    name: str

    def search(self, keyword: str, max_results: int = 20) -> list[Candidate]:
        """Search provider for media matching keyword and return list of standardized candidates."""
        ...
