import logging
from footage_engine.config import Settings, get_settings
from footage_engine.vector.base import VectorRecord, VectorSearchResult, VectorStore
from footage_engine.vector.in_memory import InMemoryVectorStore
from footage_engine.vector.zilliz import ZillizVectorStore

logger = logging.getLogger(__name__)

_vector_store_instance: VectorStore | None = None


def get_vector_store(
    settings: Settings | None = None,
    force_in_memory: bool = False,
) -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is not None and not force_in_memory:
        return _vector_store_instance

    if force_in_memory:
        return InMemoryVectorStore()

    settings = settings or get_settings()

    if settings.VECTOR_STORE == "zilliz":
        try:
            _vector_store_instance = ZillizVectorStore(
                uri=settings.ZILLIZ_URI,
                token=settings.ZILLIZ_TOKEN,
            )
            return _vector_store_instance
        except Exception as e:
            logger.warning(
                f"Could not connect to Zilliz Vector Store ({e}). Falling back to InMemoryVectorStore."
            )
            print(f"⚠️  Zilliz unavailable ({e}). Using InMemoryVectorStore for search.", flush=True)

    _vector_store_instance = InMemoryVectorStore()
    return _vector_store_instance


__all__ = [
    "VectorRecord",
    "VectorSearchResult",
    "VectorStore",
    "InMemoryVectorStore",
    "ZillizVectorStore",
    "get_vector_store",
]
