"""Embeddings module factory and exports."""

from footage_engine.config import Settings, get_settings
from footage_engine.embeddings.base import EmbeddingBackend
from footage_engine.embeddings.frames import sample_frames_from_video
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.embeddings.xclip import XCLIPEmbedder

_embedder_instances: dict[str, EmbeddingBackend] = {}


def backend_for(settings: Settings, embedder: EmbeddingBackend) -> str:
    """Detect which backend an embedder instance belongs to ('qwen' or 'xclip')."""
    if embedder.model_name == settings.QWEN_MODEL_NAME:
        return "qwen"
    return "xclip"


def collection_name_for(settings: Settings, backend: str | None = None) -> str:
    """Resolve the vector collection for a backend.

    Qwen vectors (2048d) cannot share the X-CLIP (512d) Milvus collection since
    the dimension is fixed at creation time. On a dedicated Qwen cluster the
    collection name may equal the base name; otherwise set
    QWEN_ZILLIZ_COLLECTION_NAME explicitly.
    """
    be = (backend or settings.EMBEDDING_BACKEND or "xclip").lower()
    if be == "qwen":
        return settings.QWEN_ZILLIZ_COLLECTION_NAME or settings.ZILLIZ_COLLECTION_NAME
    return settings.ZILLIZ_COLLECTION_NAME


def get_embedder(
    settings: Settings | None = None,
    use_mock: bool = False,
    backend: str | None = None,
) -> EmbeddingBackend:
    global _embedder_instances
    if use_mock:
        return MockEmbedder()

    cfg = settings or get_settings()
    be = (backend or cfg.EMBEDDING_BACKEND or "xclip").lower()
    if be in _embedder_instances:
        return _embedder_instances[be]

    if be == "qwen":
        from footage_engine.embeddings.qwen import QwenEmbedder

        instance = QwenEmbedder(
            model_name=cfg.QWEN_MODEL_NAME,
            version=cfg.QWEN_MODEL_VERSION,
            device=cfg.EMBEDDING_DEVICE,
        )
    else:
        instance = XCLIPEmbedder(
            model_name=cfg.DEFAULT_EMBEDDING_MODEL,
            version=cfg.DEFAULT_EMBEDDING_VERSION,
            device=cfg.EMBEDDING_DEVICE,
        )
    _embedder_instances[be] = instance
    return instance


__all__ = [
    "EmbeddingBackend",
    "XCLIPEmbedder",
    "MockEmbedder",
    "sample_frames_from_video",
    "get_embedder",
    "backend_for",
    "collection_name_for",
]
