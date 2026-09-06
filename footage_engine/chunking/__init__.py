"""Chunking and preprocessing package."""

from footage_engine.chunking.base import ChunkCandidate
from footage_engine.chunking.detector import (
    detect_scenes,
    preprocess_media,
    probe_video_metadata,
    sliding_window_split,
)
from footage_engine.chunking.transnet_detector import (
    detect_scenes_transnet,
    extract_clip_fast,
    parallel_extract_clips,
    preprocess_media_transnet,
)

__all__ = [
    "ChunkCandidate",
    "detect_scenes",
    "probe_video_metadata",
    "sliding_window_split",
    "preprocess_media",
    "detect_scenes_transnet",
    "preprocess_media_transnet",
    "extract_clip_fast",
    "parallel_extract_clips",
]
