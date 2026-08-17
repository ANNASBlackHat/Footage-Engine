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
