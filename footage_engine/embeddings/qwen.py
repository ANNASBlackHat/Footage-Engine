"""Qwen3-VL embedding backend (Qwen/Qwen3-VL-Embedding-2B, 2048 dimensions).

Additive second backend: implements the existing EmbeddingBackend protocol so it
is interchangeable with XCLIPEmbedder. Validated on Colab T4 (see
experiments/embed_compare.py and experiments/embed_compare_video.py).
"""

import logging
from typing import Optional
from PIL import Image

from footage_engine.embeddings.frames import sample_frames_from_video

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class QwenEmbedder:
    """Multimodal video, image, and text embedder using Qwen3-VL-Embedding-2B."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        version: str = "1.0",
        device: str = "auto",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required for QwenEmbedder. "
                'Install with: pip install -e ".[qwen]"'
            ) from err

        if torch is None:
            raise ImportError("torch is required for QwenEmbedder.")

        self.model_name = model_name
        self.version = version
        self.dimension = 2048

        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = device
        self.device = resolved

        logger.info(f"Loading Qwen embedding model '{model_name}' on device '{resolved}'...")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=resolved,
            model_kwargs={"dtype": "float16" if resolved == "cuda" else "float32"},
        )

    def embed_image(self, image: str | Image.Image) -> list[float]:
        """Generate a 2048-dim embedding for a single image / video frame."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        out = self.model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
        return [float(x) for x in out[0]]

    def _embed_frames_mean(self, frames: list[Image.Image]) -> list[float]:
        import numpy as np

        vecs = self.model.encode(
            [f.convert("RGB") for f in frames],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=4,
        )
        mean = np.mean(np.asarray(vecs, dtype=float), axis=0)
        norm = float(np.linalg.norm(mean)) or 1.0
        return [float(x / norm) for x in mean]

    def embed_video(
        self,
        video_path: str,
        start_ts: float = 0.0,
        end_ts: Optional[float] = None,
        num_frames: int = 8,
    ) -> list[float]:
        """Samples frames and computes a normalized 2048-d clip embedding (mean-pool)."""
        frames = sample_frames_from_video(
            video_path=video_path,
            start_ts=start_ts,
            end_ts=end_ts,
            num_frames=num_frames,
        )
        return self._embed_frames_mean(frames)

    def embed_video_batch(
        self,
        video_path: str,
        chunk_ranges: list[tuple[float, Optional[float]]],
        batch_size: int = 8,
        num_frames: int = 8,
    ) -> list[list[float]]:
        """Embeds multiple video segments (sequential range loop, GPU-batched frames)."""
        return [
            self.embed_video(video_path, s_ts, e_ts, num_frames)
            for s_ts, e_ts in chunk_ranges
        ]

    def embed_text(self, text: str) -> list[float]:
        """Embeds natural language text query into the 2048-d shared space."""
        prompts = getattr(self.model, "prompts", None) or {}
        out = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name="query" if "query" in prompts else None,
        )
        return [float(x) for x in out[0]]
