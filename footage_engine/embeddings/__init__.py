"""Embeddings module factory and exports."""

from footage_engine.config import Settings, get_settings
from footage_engine.embeddings.base import EmbeddingBackend
from footage_engine.embeddings.frames import sample_frames_from_video
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.embeddings.xclip import XCLIPEmbedder

_embedder_instance: EmbeddingBackend | None = None


def get_embedder(
    settings: Settings | None = None,
    use_mock: bool = False,
) -> EmbeddingBackend:
    global _embedder_instance
    if _embedder_instance is not None and not use_mock:
        return _embedder_instance

    if use_mock:
        return MockEmbedder()

    cfg = settings or get_settings()
    _embedder_instance = XCLIPEmbedder(
        model_name=cfg.DEFAULT_EMBEDDING_MODEL,
        version=cfg.DEFAULT_EMBEDDING_VERSION,
        device=cfg.EMBEDDING_DEVICE,
    )
    return _embedder_instance


__all__ = [
    "EmbeddingBackend",
    "XCLIPEmbedder",
    "MockEmbedder",
    "sample_frames_from_video",
    "get_embedder",
]
