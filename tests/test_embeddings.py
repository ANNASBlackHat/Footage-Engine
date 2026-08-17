"""Tests for embedding generation and frame extraction."""

import math
import os
import pytest
from PIL import Image

from footage_engine.embeddings.frames import sample_frames_from_video
from footage_engine.embeddings.mock import MockEmbedder


def test_mock_embedder_dimension_and_norm():
    embedder = MockEmbedder(dimension=512)

    # Test text embedding
    t_vec = embedder.embed_text("running dog in park")
    assert len(t_vec) == 512
    l2_norm = math.sqrt(sum(x * x for x in t_vec))
    assert abs(l2_norm - 1.0) < 1e-5

    # Test image embedding
    i_vec = embedder.embed_image("test_image.jpg")
    assert len(i_vec) == 512
    l2_norm_i = math.sqrt(sum(x * x for x in i_vec))
    assert abs(l2_norm_i - 1.0) < 1e-5

    # Test video segment embedding
    v_vec = embedder.embed_video("test_video.mp4", start_ts=0.0, end_ts=10.0)
    assert len(v_vec) == 512
    l2_norm_v = math.sqrt(sum(x * x for x in v_vec))
    assert abs(l2_norm_v - 1.0) < 1e-5


def test_mock_embedder_determinism_and_uniqueness():
    embedder = MockEmbedder(dimension=512)
    v1 = embedder.embed_text("sunset beach")
    v2 = embedder.embed_text("sunset beach")
    v3 = embedder.embed_text("city skyline at night")

    assert v1 == v2
    assert v1 != v3


def test_sample_frames_from_synthetic_video(temp_dir):
    import cv2
    import numpy as np

    video_path = os.path.join(temp_dir, "frame_test.mp4")
    fps = 30
    width, height = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # 3 seconds = 90 frames with changing colors
    for i in range(90):
        color = (i * 2 % 256, 128, 255 - (i * 2 % 256))
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        out.write(frame)

    out.release()

    # Extract 8 frames
    frames = sample_frames_from_video(video_path, start_ts=0.0, end_ts=3.0, num_frames=8)
    assert len(frames) == 8
    for frame in frames:
        assert isinstance(frame, Image.Image)
        assert frame.size == (160, 120)
        assert frame.mode == "RGB"
