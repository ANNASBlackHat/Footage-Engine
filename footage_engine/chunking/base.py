"""Data models and definitions for media chunking."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkCandidate:
    """Represents a proposed segment/chunk of a media item before embedding."""

    start_ts: float
    end_ts: Optional[float] = None
    media_type: str = "video"

    @property
    def duration(self) -> Optional[float]:
        if self.end_ts is None:
            return None
        return max(0.0, self.end_ts - self.start_ts)

    def __repr__(self) -> str:
        return f"<ChunkCandidate [{self.start_ts:.2f}s - {self.end_ts if self.end_ts is not None else 'end'}s]>"
