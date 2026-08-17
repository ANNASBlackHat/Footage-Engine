"""Tests for database models and schema constraints."""

import pytest
from sqlalchemy.exc import IntegrityError
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaStatus, MediaType


def test_media_item_creation_and_defaults(test_settings):
    init_db(test_settings.DATABASE_URL)
    with get_db_session(test_settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="pexels",
            source_id="12345",
            source_url="https://images.pexels.com/videos/12345/video.mp4",
            storage_path="pexels_12345.mp4",
            duration_sec=15.5,
            resolution="1920x1080",
        )
        session.add(item)
        session.flush()

        assert item.id is not None
        assert len(item.id) == 36  # UUID string
        assert item.status == MediaStatus.PENDING
        assert item.media_type == MediaType.VIDEO
        assert item.ingested_at is not None
        assert item.updated_at is not None


def test_unique_constraint_on_source_url(test_settings):
    init_db(test_settings.DATABASE_URL)
    with get_db_session(test_settings.DATABASE_URL) as session:
        item1 = MediaItem(
            provider="pixabay",
            source_id="1",
            source_url="https://cdn.pixabay.com/video1.mp4",
            storage_path="path1.mp4",
        )
        session.add(item1)
        session.flush()

    # Attempt inserting duplicate source_url
    with pytest.raises(IntegrityError):
        with get_db_session(test_settings.DATABASE_URL) as session:
            item2 = MediaItem(
                provider="different_provider",
                source_id="2",
                source_url="https://cdn.pixabay.com/video1.mp4",
                storage_path="path2.mp4",
            )
            session.add(item2)
            session.flush()


def test_unique_constraint_on_provider_and_source_id(test_settings):
    init_db(test_settings.DATABASE_URL)
    with get_db_session(test_settings.DATABASE_URL) as session:
        item1 = MediaItem(
            provider="pixabay",
            source_id="abc999",
            source_url="https://cdn.pixabay.com/video_a.mp4",
            storage_path="path_a.mp4",
        )
        session.add(item1)
        session.flush()

    # Attempt inserting duplicate provider + source_id
    with pytest.raises(IntegrityError):
        with get_db_session(test_settings.DATABASE_URL) as session:
            item2 = MediaItem(
                provider="pixabay",
                source_id="abc999",
                source_url="https://cdn.pixabay.com/video_b.mp4",
                storage_path="path_b.mp4",
            )
            session.add(item2)
            session.flush()


def test_chunk_relationship_and_cascade(test_settings):
    init_db(test_settings.DATABASE_URL)
    with get_db_session(test_settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="manual",
            source_url="https://example.com/video.mp4",
            storage_path="video.mp4",
            duration_sec=60.0,
        )
        session.add(item)
        session.flush()

        chunk1 = Chunk(
            media_item_id=item.id,
            start_ts=0.0,
            end_ts=10.0,
            embedding_model="xclip-base-patch32",
            embedding_version="1.0",
        )
        chunk2 = Chunk(
            media_item_id=item.id,
            start_ts=10.0,
            end_ts=20.0,
            embedding_model="xclip-base-patch32",
            embedding_version="1.0",
        )
        session.add_all([chunk1, chunk2])
        session.flush()

        assert len(item.chunks) == 2
        chunk_ids = [chunk1.id, chunk2.id]

    # Deleting the parent MediaItem should cascade delete the Chunks
    with get_db_session(test_settings.DATABASE_URL) as session:
        item_to_delete = session.get(MediaItem, item.id)
        session.delete(item_to_delete)
        session.flush()

    with get_db_session(test_settings.DATABASE_URL) as session:
        for cid in chunk_ids:
            assert session.get(Chunk, cid) is None


def test_media_item_and_chunk_aspect_ratio():
    from footage_engine.retrieval.models import ChunkResult

    # Landscape / Horizontal 16:9
    item_16_9 = MediaItem(
        provider="pexels",
        source_url="https://example.com/h.mp4",
        storage_path="h.mp4",
        resolution="1920x1080",
    )
    assert item_16_9.width == 1920
    assert item_16_9.height == 1080
    assert item_16_9.orientation == "horizontal"
    assert item_16_9.aspect_ratio == "16:9"

    # Vertical 9:16 (Shorts / Reels / TikTok)
    item_9_16 = MediaItem(
        provider="pexels",
        source_url="https://example.com/v.mp4",
        storage_path="v.mp4",
        resolution="1080x1920",
    )
    assert item_9_16.orientation == "vertical"
    assert item_9_16.aspect_ratio == "9:16"

    # Square 1:1
    item_1_1 = MediaItem(
        provider="pixabay",
        source_url="https://example.com/s.mp4",
        storage_path="s.mp4",
        resolution="1080x1080",
    )
    assert item_1_1.orientation == "square"
    assert item_1_1.aspect_ratio == "1:1"

    # ChunkResult properties
    res = ChunkResult(
        chunk_id="chunk123",
        media_item_id="item123",
        score=0.95,
        start_ts=0.0,
        end_ts=10.0,
        duration_sec=10.0,
        media_type="video",
        provider="pexels",
        source_url="https://example.com/h.mp4",
        storage_path="h.mp4",
        storage_url="https://storage/h.mp4",
        resolution="3840x2160",
    )
    assert res.orientation == "horizontal"
    assert res.aspect_ratio == "16:9"
