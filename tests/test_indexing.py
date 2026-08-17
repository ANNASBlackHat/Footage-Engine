"""Tests for Vector Indexing and Resumable Batch Processing Pipeline."""

import os
import pytest
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import MediaItem, MediaStatus, MediaType
from footage_engine.pipeline.processor import BatchProcessor
from footage_engine.vector.base import VectorRecord
from footage_engine.vector.in_memory import InMemoryVectorStore


def test_in_memory_vector_store_crud():
    store = InMemoryVectorStore()
    col_name = "test_chunks"

    rec1 = VectorRecord(
        id="c1",
        vector=[1.0, 0.0, 0.0],
        chunk_id="c1",
        media_item_id="m1",
        provider="pexels",
        media_type="video",
        duration_sec=10.0,
    )
    rec2 = VectorRecord(
        id="c2",
        vector=[0.0, 1.0, 0.0],
        chunk_id="c2",
        media_item_id="m2",
        provider="pixabay",
        media_type="image",
        duration_sec=None,
    )

    ids = store.upsert(col_name, [rec1, rec2])
    assert ids == ["c1", "c2"]

    # Query with vector identical to rec1
    results = store.search(col_name, query_vector=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "c1"
    assert abs(results[0].score - 1.0) < 1e-5
    assert results[1].id == "c2"
    assert abs(results[1].score - 0.0) < 1e-5

    # Filter test
    filtered = store.search(col_name, query_vector=[1.0, 0.0, 0.0], top_k=2, filter_expr='provider == "pixabay"')
    assert len(filtered) == 1
    assert filtered[0].id == "c2"

    # Delete
    assert store.delete(col_name, ["c1"]) is True
    post_delete = store.search(col_name, query_vector=[1.0, 0.0, 0.0], top_k=2)
    assert len(post_delete) == 1
    assert post_delete[0].id == "c2"


def test_batch_processor_end_to_end(test_settings, test_storage, temp_dir):
    init_db(test_settings.DATABASE_URL)
    vector_store = InMemoryVectorStore()
    embedder = MockEmbedder(dimension=512)

    # 1. Create a dummy stored media file
    dummy_file = "sample_test_clip.mp4"
    test_storage.save_file(b"dummy video content", dummy_file)

    # 2. Add MediaItem with status=PENDING
    with get_db_session(test_settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="pexels",
            source_url="https://example.com/sample_test_clip.mp4",
            storage_path=dummy_file,
            duration_sec=20.0,  # <= 45s so 1 atomic chunk
            status=MediaStatus.PENDING,
        )
        session.add(item)
        session.flush()
        item_id = item.id

    # 3. Run BatchProcessor
    processor = BatchProcessor(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )

    stats = processor.process_all_pending()
    assert stats["total"] == 1
    assert stats["succeeded"] == 1
    assert stats["failed"] == 0

    # 4. Verify item and chunk states
    with get_db_session(test_settings.DATABASE_URL) as session:
        updated_item = session.get(MediaItem, item_id)
        assert updated_item.status == MediaStatus.DONE
        assert len(updated_item.chunks) == 1
        chunk = updated_item.chunks[0]
        assert chunk.vector_id == chunk.id
        assert chunk.start_ts == 0.0
        assert chunk.end_ts == 20.0

    # 5. Verify vectors exist in vector store
    results = vector_store.search(
        test_settings.ZILLIZ_COLLECTION_NAME,
        query_vector=embedder.embed_text("test query"),
        top_k=1,
    )
    assert len(results) == 1
    assert results[0].media_item_id == item_id

    # 6. Re-run batch processor (must skip done items)
    stats2 = processor.process_all_pending()
    assert stats2["total"] == 0
