"""Tests for Retrieval API, search filtering, and fine localization."""

import os
import pytest
from sqlalchemy import select
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaStatus, MediaType
from footage_engine.pipeline.processor import BatchProcessor
from footage_engine.retrieval.api import RetrievalAPI
from footage_engine.retrieval.models import SearchFilters
from footage_engine.vector.in_memory import InMemoryVectorStore


def test_retrieval_api_search_and_hydration(test_settings, test_storage):
    init_db(test_settings.DATABASE_URL)
    vector_store = InMemoryVectorStore()
    embedder = MockEmbedder(dimension=512)

    # 1. Create and ingest dummy items
    test_storage.save_file(b"clip 1", "clip1.mp4")
    test_storage.save_file(b"clip 2", "clip2.mp4")

    with get_db_session(test_settings.DATABASE_URL) as session:
        item1 = MediaItem(
            provider="pexels",
            source_url="https://example.com/clip1.mp4",
            storage_path="clip1.mp4",
            duration_sec=15.0,
            resolution="1920x1080",
            license_type="pexels_free",
            status=MediaStatus.PENDING,
        )
        item2 = MediaItem(
            provider="pixabay",
            source_url="https://example.com/clip2.mp4",
            storage_path="clip2.mp4",
            duration_sec=30.0,
            resolution="1280x720",
            license_type="pixabay_free",
            status=MediaStatus.PENDING,
        )
        session.add_all([item1, item2])

    # 2. Process items
    processor = BatchProcessor(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )
    processor.process_all_pending()

    # 3. Test Retrieval API
    api = RetrievalAPI(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )

    results = api.search(query="nature landscape", top_k=5)
    assert len(results) == 2
    assert results[0].storage_url.startswith("file://")
    assert results[0].resolution in ["1920x1080", "1280x720"]

    # Check usage count increment
    assert results[0].usage_count >= 1
    assert results[0].last_used_at is not None

    # Test filtering by provider
    filtered_results = api.search(
        query="nature landscape",
        top_k=5,
        filters=SearchFilters(provider="pixabay"),
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].provider == "pixabay"


def test_get_chunk_and_get_media_item(test_settings, test_storage):
    init_db(test_settings.DATABASE_URL)
    vector_store = InMemoryVectorStore()
    embedder = MockEmbedder(dimension=512)

    test_storage.save_file(b"img data", "photo.jpg")

    with get_db_session(test_settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="manual",
            source_url="https://example.com/photo.jpg",
            storage_path="photo.jpg",
            media_type=MediaType.IMAGE,
            status=MediaStatus.PENDING,
        )
        session.add(item)
        session.flush()
        item_id = item.id

    processor = BatchProcessor(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )
    processor.process_all_pending()

    api = RetrievalAPI(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )

    # get_media_item
    fetched_item = api.get_media_item(item_id)
    assert fetched_item is not None
    assert fetched_item.id == item_id
    assert fetched_item.status == MediaStatus.DONE

    # get_chunk
    with get_db_session(test_settings.DATABASE_URL) as session:
        chunk = session.execute(select(Chunk).where(Chunk.media_item_id == item_id)).scalars().first()
        chunk_id = chunk.id

    chunk_res = api.get_chunk(chunk_id)
    assert chunk_res is not None
    assert chunk_res.chunk_id == chunk_id
    assert chunk_res.media_type == "image"


def test_fine_localize_synthetic_video(test_settings, test_storage, temp_dir):
    import cv2
    import numpy as np

    init_db(test_settings.DATABASE_URL)
    vector_store = InMemoryVectorStore()
    embedder = MockEmbedder(dimension=512)

    # Create synthetic video: 5 seconds (120 frames at 24fps)
    video_filename = "synth_localize.mp4"
    local_path = os.path.join(test_storage.base_dir, video_filename)
    fps = 24
    width, height = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(local_path, fourcc, fps, (width, height))

    for i in range(120):
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        out.write(frame)
    out.release()

    with get_db_session(test_settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="manual",
            source_url="https://example.com/synth_localize.mp4",
            storage_path=video_filename,
            duration_sec=5.0,
            status=MediaStatus.PENDING,
        )
        session.add(item)
        session.flush()
        item_id = item.id

    processor = BatchProcessor(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )
    processor.process_all_pending()

    api = RetrievalAPI(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )

    with get_db_session(test_settings.DATABASE_URL) as session:
        chunk = session.execute(select(Chunk).where(Chunk.media_item_id == item_id)).scalars().first()
        chunk_id = chunk.id

    start_ts, end_ts = api.fine_localize(chunk_id, query="bright daylight scene")
    assert start_ts >= 0.0
    assert end_ts <= 5.0
    assert start_ts < end_ts
