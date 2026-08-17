"""Mock embedding backend for ultra-fast, zero-download unit testing."""

import hashlib
import math
from typing import Optional


class MockEmbedder:
    """Generates deterministic unit-normalized 512-dimensional vectors."""

    def __init__(
        self,
        model_name: str = "mock-xclip-base",
        version: str = "1.0",
        dimension: int = 512,
    ):
        self.model_name = model_name
        self.version = version
        self.dimension = dimension

    def _generate_vector(self, seed_str: str) -> list[float]:
        # Hash seed string to generate pseudo-random deterministic floats
        hasher = hashlib.sha256(seed_str.encode("utf-8"))
        seed_bytes = hasher.digest()

        vec = []
        for i in range(self.dimension):
            byte_val = seed_bytes[i % len(seed_bytes)]
            val = math.sin((i + 1) * byte_val)
            vec.append(val)

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_video(
        self,
        video_path: str,
        start_ts: float = 0.0,
        end_ts: Optional[float] = None,
        num_frames: int = 8,
    ) -> list[float]:
        seed = f"video:{video_path}:{start_ts}:{end_ts}"
        return self._generate_vector(seed)

    def embed_image(self, image_path: str) -> list[float]:
        return self._generate_vector(f"image:{image_path}")

    def embed_text(self, text: str) -> list[float]:
        return self._generate_vector(f"text:{text.strip().lower()}")
