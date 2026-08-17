"""Chunking and preprocessing package."""

from footage_engine.chunking.base import ChunkCandidate
from footage_engine.chunking.detector import (
    detect_scenes,
    preprocess_media,
    probe_video_metadata,
    sliding_window_split,
)

__all__ = [
    "ChunkCandidate",
    "detect_scenes",
    "probe_video_metadata",
    "sliding_window_split",
    "preprocess_media",
]
