"""Tests for dual-model (X-CLIP + Qwen) support.

All tests use MockEmbedder + InMemoryVectorStore: no model downloads, no GPU.
"""

from footage_engine.embeddings import backend_for, collection_name_for
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaStatus, MediaType
from footage_engine.pipeline.processor import BatchProcessor
from footage_engine.retrieval.api import RetrievalAPI
from footage_engine.vector import zilliz_creds_for
from footage_engine.vector.in_memory import InMemoryVectorStore


def _make_image_item(test_settings, test_storage, source_url="https://example.com/a.jpg"):
    test_storage.save_file(b"img data", "a.jpg")
    with get_db_session(test_settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="manual",
            source_url=source_url,
            storage_path="a.jpg",
            media_type=MediaType.IMAGE,
            status=MediaStatus.PENDING,
        )
        session.add(item)
        session.flush()
        return item.id


def test_collection_name_for(test_settings):
    base = test_settings.ZILLIZ_COLLECTION_NAME
    assert collection_name_for(test_settings, "xclip") == base
    assert collection_name_for(test_settings, None) == base  # default backend is xclip
    # No explicit Qwen collection -> falls back to base name (dedicated cluster case)
    assert collection_name_for(test_settings, "qwen") == base
    test_settings.QWEN_ZILLIZ_COLLECTION_NAME = "footage_chunks_qwen"
    assert collection_name_for(test_settings, "qwen") == "footage_chunks_qwen"


def test_backend_for_and_zilliz_creds(test_settings):
    xclip_like = MockEmbedder(model_name="mock-xclip", dimension=512)
    qwen_like = MockEmbedder(model_name=test_settings.QWEN_MODEL_NAME, dimension=2048)
    assert backend_for(test_settings, xclip_like) == "xclip"
    assert backend_for(test_settings, qwen_like) == "qwen"

    test_settings.ZILLIZ_URI = "https://base.example.com"
    test_settings.ZILLIZ_TOKEN = "base-token"
    test_settings.QWEN_ZILLIZ_URI = None
    test_settings.QWEN_ZILLIZ_TOKEN = None
    assert zilliz_creds_for(test_settings, "xclip") == ("https://base.example.com", "base-token")
    # Qwen falls back to base when QWEN_* unset ...
    assert zilliz_creds_for(test_settings, "qwen") == ("https://base.example.com", "base-token")
    # ... and uses dedicated values when set
    test_settings.QWEN_ZILLIZ_URI = "https://qwen.example.com"
    test_settings.QWEN_ZILLIZ_TOKEN = "qwen-token"
    assert zilliz_creds_for(test_settings, "qwen") == ("https://qwen.example.com", "qwen-token")


def test_processor_and_retrieval_collection_resolution(test_settings, test_storage):
    xclip_like = MockEmbedder(model_name="mock-xclip", dimension=512)
    qwen_like = MockEmbedder(model_name=test_settings.QWEN_MODEL_NAME, dimension=2048)

    proc_x = BatchProcessor(
        settings=test_settings, storage=test_storage, embedder=xclip_like,
        vector_store=InMemoryVectorStore(), database_url=test_settings.DATABASE_URL,
    )
    proc_q = BatchProcessor(
        settings=test_settings, storage=test_storage, embedder=qwen_like,
        vector_store=InMemoryVectorStore(), database_url=test_settings.DATABASE_URL,
    )
    assert proc_x.collection_name == test_settings.ZILLIZ_COLLECTION_NAME
    assert proc_q.collection_name == collection_name_for(test_settings, "qwen")

    api_q = RetrievalAPI(
        settings=test_settings, storage=test_storage, embedder=qwen_like,
        vector_store=InMemoryVectorStore(), database_url=test_settings.DATABASE_URL,
    )
    assert api_q.collection_name == proc_q.collection_name


def test_second_backend_creates_own_chunks(test_settings, test_storage):
    init_db(test_settings.DATABASE_URL)
    item_id = _make_image_item(test_settings, test_storage)

    store = InMemoryVectorStore()
    embedder_a = MockEmbedder(model_name="model-a", dimension=512)
    embedder_b = MockEmbedder(model_name="model-b", dimension=512)

    proc_a = BatchProcessor(
        settings=test_settings, storage=test_storage, embedder=embedder_a,
        vector_store=store, database_url=test_settings.DATABASE_URL,
        collection_name="col_a",
    )
    assert proc_a.process_item(item_id) is True

    # Simulate a second backend run: reopen item as pending (backfill does this per-range)
    with get_db_session(test_settings.DATABASE_URL) as session:
        item = session.get(MediaItem, item_id)
        assert item.status == MediaStatus.DONE
        item.status = MediaStatus.PENDING

    proc_b = BatchProcessor(
        settings=test_settings, storage=test_storage, embedder=embedder_b,
        vector_store=store, database_url=test_settings.DATABASE_URL,
        collection_name="col_b",
    )
    assert proc_b.process_item(item_id) is True

    with get_db_session(test_settings.DATABASE_URL) as session:
        chunks = session.query(Chunk).filter(Chunk.media_item_id == item_id).all()
        models = sorted(c.embedding_model for c in chunks)
        assert models == ["model-a", "model-b"]
        assert len({c.id for c in chunks}) == 2  # distinct rows, distinct vector ids

    assert len(store.collections["col_a"]) == 1
    assert len(store.collections["col_b"]) == 1
