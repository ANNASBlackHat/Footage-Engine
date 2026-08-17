"""Tests for media preprocessing, scene detection, and chunking."""

import os
import pytest
from footage_engine.chunking.base import ChunkCandidate
from footage_engine.chunking.detector import (
    create_chunks_in_db,
    detect_scenes,
    preprocess_media,
    probe_video_metadata,
    sliding_window_split,
)
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import MediaItem, MediaStatus, MediaType
from footage_engine.storage.local import LocalStorageBackend


def test_chunk_candidate_duration():
    c1 = ChunkCandidate(start_ts=0.0, end_ts=12.5)
    assert c1.duration == 12.5
    assert c1.media_type == "video"

    c2 = ChunkCandidate(start_ts=0.0, end_ts=None, media_type="image")
    assert c2.duration is None


def test_image_preprocessing(test_storage):
    item = MediaItem(
        provider="manual",
        source_url="https://example.com/photo.jpg",
        storage_path="photo.jpg",
        media_type=MediaType.IMAGE,
    )
    chunks = preprocess_media(item, test_storage)
    assert len(chunks) == 1
    assert chunks[0].start_ts == 0.0
    assert chunks[0].end_ts is None
    assert chunks[0].media_type == "image"


def test_short_clip_preprocessing(test_storage):
    item = MediaItem(
        provider="pexels",
        source_url="https://example.com/short.mp4",
        storage_path="short.mp4",
        duration_sec=30.0,
        media_type=MediaType.VIDEO,
    )
    chunks = preprocess_media(item, test_storage, chunk_threshold_sec=45.0)
    assert len(chunks) == 1
    assert chunks[0].start_ts == 0.0
    assert chunks[0].end_ts == 30.0
    assert chunks[0].media_type == "video"


def test_sliding_window_split():
    # 25 seconds split with 10s window and 50% overlap (step 5s)
    chunks = sliding_window_split(start_ts=0.0, end_ts=25.0, window_sec=10.0, overlap_ratio=0.5)
    assert len(chunks) == 4
    assert chunks[0].start_ts == 0.0 and chunks[0].end_ts == 10.0
    assert chunks[1].start_ts == 5.0 and chunks[1].end_ts == 15.0
    assert chunks[2].start_ts == 10.0 and chunks[2].end_ts == 20.0
    assert chunks[3].start_ts == 15.0 and chunks[3].end_ts == 25.0


def test_synthetic_video_scene_detection(temp_dir):
    import cv2
    import numpy as np

    video_path = os.path.join(temp_dir, "synthetic_scenes.mp4")
    fps = 24
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Scene 1: 3 seconds of solid black (72 frames)
    for _ in range(72):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)

    # Scene 2: 3 seconds of solid white (72 frames)
    for _ in range(72):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        out.write(frame)

    out.release()

    # 1. Test probe metadata
    meta = probe_video_metadata(video_path)
    assert meta["resolution"] == "320x240"
    assert meta["frame_count"] == 144
    assert abs(meta["duration_sec"] - 6.0) < 0.1

    # 2. Test scene detection
    scenes = detect_scenes(video_path)
    assert len(scenes) >= 1


def test_create_chunks_in_db(test_settings):
    init_db(test_settings.DATABASE_URL)
    with get_db_session(test_settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="pexels",
            source_url="https://example.com/test_video.mp4",
            storage_path="test_video.mp4",
            duration_sec=60.0,
            status=MediaStatus.PENDING,
        )
        session.add(item)
        session.flush()

        candidates = [
            ChunkCandidate(0.0, 10.0),
            ChunkCandidate(10.0, 20.0),
        ]
        created = create_chunks_in_db(item, candidates, session)
        assert len(created) == 2
        assert item.status == MediaStatus.CHUNKING
        assert created[0].embedding_model == "xclip-base-patch32"
        assert created[0].embedding_version == "1.0"
