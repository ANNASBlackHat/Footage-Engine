"""Base definitions and protocols for embedding models."""

from typing import Optional, Protocol
import numpy as np


class EmbeddingBackend(Protocol):
    """Protocol for multimodal video/image/text embedding backends."""

    model_name: str
    version: str
    dimension: int

    def embed_video(
        self,
        video_path: str,
        start_ts: float = 0.0,
        end_ts: Optional[float] = None,
        num_frames: int = 8,
    ) -> list[float]:
        """Extracts and embeds frames from a video segment into a unit-normalized vector."""
        ...

    def embed_image(self, image_path: str) -> list[float]:
        """Embeds a still image into the shared video-text embedding space."""
        ...

    def embed_text(self, text: str) -> list[float]:
        """Embeds a natural language query into the shared embedding space."""
        ...
