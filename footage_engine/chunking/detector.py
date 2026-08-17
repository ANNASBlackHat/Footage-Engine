"""Scene detection and chunking preprocessing engine."""

import logging
from typing import Any, Optional
from sqlalchemy.orm import Session

from footage_engine.chunking.base import ChunkCandidate
from footage_engine.models.media import Chunk, MediaItem, MediaStatus, MediaType
from footage_engine.storage.base import StorageBackend

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import scenedetect
    from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video
except ImportError:
    scenedetect = None  # type: ignore


def probe_video_metadata(video_path: str) -> dict[str, Any]:
    """Extracts duration, resolution, fps, and total frames using OpenCV."""
    if cv2 is None:
        raise ImportError("opencv-python is required for video metadata probing.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file at {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration_sec = 0.0
        if fps and fps > 0:
            duration_sec = float(frame_count / fps)

        resolution = f"{width}x{height}" if width and height else None

        return {
            "duration_sec": duration_sec,
            "resolution": resolution,
            "fps": fps,
            "frame_count": int(frame_count),
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def detect_scenes(
    video_path: str,
    adaptive_threshold: float = 3.0,
    min_scene_len_sec: float = 0.8,
) -> list[tuple[float, float]]:
    """Detects scene boundaries in a video using PySceneDetect AdaptiveDetector."""
    if scenedetect is None:
        raise ImportError("scenedetect is required for scene detection. Install scenedetect[opencv].")

    video = open_video(video_path)
    fps = video.frame_rate
    min_scene_len_frames = int(min_scene_len_sec * fps) if fps else 15

    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len_frames,
        )
    )

    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        meta = probe_video_metadata(video_path)
        return [(0.0, meta["duration_sec"])]

    results: list[tuple[float, float]] = []
    for start_time, end_time in scene_list:
        s_sec = getattr(start_time, "seconds", None)
        if s_sec is None:
            s_sec = start_time.get_seconds()
        e_sec = getattr(end_time, "seconds", None)
        if e_sec is None:
            e_sec = end_time.get_seconds()
        results.append((float(s_sec), float(e_sec)))

    return results


def sliding_window_split(
    start_ts: float,
    end_ts: float,
    window_sec: float = 10.0,
    overlap_ratio: float = 0.5,
) -> list[ChunkCandidate]:
    """Splits an uncut or long continuous scene into overlapping sliding window chunks."""
    total_dur = end_ts - start_ts
    if total_dur <= window_sec:
        return [ChunkCandidate(start_ts=start_ts, end_ts=end_ts, media_type="video")]

    step_sec = max(1.0, window_sec * (1.0 - overlap_ratio))
    chunks: list[ChunkCandidate] = []
    cur_start = start_ts

    while cur_start < end_ts:
        cur_end = min(cur_start + window_sec, end_ts)
        # Avoid creating tiny stub chunk at the tail
        if (cur_end - cur_start) < 2.0 and len(chunks) > 0:
            # Extend previous chunk to cover remaining tail
            chunks[-1].end_ts = end_ts
            break
        chunks.append(ChunkCandidate(start_ts=round(cur_start, 2), end_ts=round(cur_end, 2), media_type="video"))
        if cur_end >= end_ts:
            break
        cur_start += step_sec

    return chunks


def preprocess_media(
    media_item: MediaItem,
    storage: StorageBackend,
    chunk_threshold_sec: float = 45.0,
    window_sec: float = 10.0,
    overlap_ratio: float = 0.5,
) -> list[ChunkCandidate]:
    """Preprocesses a MediaItem into ChunkCandidate segments according to SPEC.md rules."""
    # 1. Image handling: single chunk [0, None]
    if media_item.media_type in (MediaType.IMAGE, "image"):
        return [ChunkCandidate(start_ts=0.0, end_ts=None, media_type="image")]

    # 2. Already-short clip (< CHUNK_THRESHOLD) with known duration? Treat whole clip as one chunk
    if media_item.duration_sec is not None and 0 < media_item.duration_sec <= chunk_threshold_sec:
        return [ChunkCandidate(start_ts=0.0, end_ts=round(media_item.duration_sec, 2), media_type="video")]

    # 3. Get local file path for video inspection
    try:
        local_path = storage.get_local_path(media_item.storage_path)
    except Exception:
        local_path = storage.get_local_path(media_item.source_url)

    # Probe duration if missing
    duration = media_item.duration_sec
    if duration is None or duration <= 0:
        meta = probe_video_metadata(local_path)
        duration = meta["duration_sec"]

    if duration <= chunk_threshold_sec:
        return [ChunkCandidate(start_ts=0.0, end_ts=round(duration, 2), media_type="video")]

    # 4. Longer source? Detect scenes via PySceneDetect AdaptiveDetector
    scenes = detect_scenes(local_path)
    chunks: list[ChunkCandidate] = []
    for scene_start, scene_end in scenes:
        scene_dur = scene_end - scene_start
        if scene_dur <= chunk_threshold_sec:
            chunks.append(
                ChunkCandidate(
                    start_ts=round(scene_start, 2),
                    end_ts=round(scene_end, 2),
                    media_type="video",
                )
            )
        else:
            # Long uncut scene split via fixed sliding window
            sub_chunks = sliding_window_split(
                start_ts=scene_start,
                end_ts=scene_end,
                window_sec=window_sec,
                overlap_ratio=overlap_ratio,
            )
            chunks.extend(sub_chunks)

    return chunks


def create_chunks_in_db(
    media_item: MediaItem,
    candidates: list[ChunkCandidate],
    session: Session,
    embedding_model: str = "xclip-base-patch32",
    embedding_version: str = "1.0",
) -> list[Chunk]:
    """Persists ChunkCandidate segments as database Chunk records linked to media_item."""
    created_chunks: list[Chunk] = []
    for cand in candidates:
        chunk = Chunk(
            media_item_id=media_item.id,
            start_ts=cand.start_ts,
            end_ts=cand.end_ts,
            media_type=MediaType.IMAGE if cand.media_type == "image" else MediaType.VIDEO,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
        )
        session.add(chunk)
        created_chunks.append(chunk)

    media_item.status = MediaStatus.CHUNKING
    session.flush()
    return created_chunks
