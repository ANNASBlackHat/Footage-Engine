import logging
from footage_engine.config import Settings, get_settings
from footage_engine.vector.base import VectorRecord, VectorSearchResult, VectorStore
from footage_engine.vector.in_memory import InMemoryVectorStore
from footage_engine.vector.zilliz import ZillizVectorStore

logger = logging.getLogger(__name__)

_vector_store_instances: dict[str, VectorStore] = {}


def zilliz_creds_for(
    settings: Settings, backend: str | None = None
) -> tuple[str | None, str | None]:
    """Resolve (uri, token) for a backend, falling back to base ZILLIZ_* values."""
    be = (backend or settings.EMBEDDING_BACKEND or "xclip").lower()
    if be == "qwen":
        return (
            settings.QWEN_ZILLIZ_URI or settings.ZILLIZ_URI,
            settings.QWEN_ZILLIZ_TOKEN or settings.ZILLIZ_TOKEN,
        )
    return (settings.ZILLIZ_URI, settings.ZILLIZ_TOKEN)


def get_vector_store(
    settings: Settings | None = None,
    force_in_memory: bool = False,
    backend: str | None = None,
) -> VectorStore:
    global _vector_store_instances
    settings = settings or get_settings()
    be = (backend or settings.EMBEDDING_BACKEND or "xclip").lower()

    if force_in_memory:
        return InMemoryVectorStore()

    if be in _vector_store_instances:
        return _vector_store_instances[be]

    if settings.VECTOR_STORE == "zilliz":
        uri, token = zilliz_creds_for(settings, be)
        try:
            _vector_store_instances[be] = ZillizVectorStore(uri=uri, token=token)
            return _vector_store_instances[be]
        except Exception as e:
            logger.warning(
                f"Could not connect to Zilliz Vector Store ({e}). Falling back to InMemoryVectorStore."
            )
            print(f"⚠️  Zilliz unavailable ({e}). Using InMemoryVectorStore for search.", flush=True)

    _vector_store_instances[be] = InMemoryVectorStore()
    return _vector_store_instances[be]


__all__ = [
    "VectorRecord",
    "VectorSearchResult",
    "VectorStore",
    "InMemoryVectorStore",
    "ZillizVectorStore",
    "get_vector_store",
    "zilliz_creds_for",
]
