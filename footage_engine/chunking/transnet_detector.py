"""TransNetV2 scene detection and fast parallel clipping engine."""

import concurrent.futures
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Optional

from footage_engine.chunking.base import ChunkCandidate
from footage_engine.chunking.detector import probe_video_metadata, sliding_window_split
from footage_engine.models.media import MediaItem, MediaType
from footage_engine.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def _load_transnetv2_class():
    """Dynamically attempts to import TransNetV2 from installed packages or common clone directories."""
    # 1. Direct import if already in python path
    try:
        from transnetv2 import TransNetV2  # type: ignore
        return TransNetV2
    except ImportError:
        pass

    # 2. PyTorch port if available
    try:
        from transnetv2_pytorch import TransNetV2  # type: ignore
        return TransNetV2
    except ImportError:
        pass

    # 3. Check common clone paths (Colab / local repo)
    candidate_dirs = [
        os.environ.get("TRANSNET_DIR", ""),
        "/content/TransNetV2/inference",
        os.path.join(os.getcwd(), "TransNetV2", "inference"),
        os.path.join(os.getcwd(), "TransNetV2"),
        os.path.expanduser("~/TransNetV2/inference"),
    ]

    for cdir in candidate_dirs:
        if cdir and os.path.isdir(cdir):
            if cdir not in sys.path:
                sys.path.insert(0, cdir)
            try:
                from transnetv2 import TransNetV2  # type: ignore
                return TransNetV2
            except ImportError:
                continue

    raise ImportError(
        "TransNetV2 is not found in Python path. To use TransNetV2:\n"
        "1. Clone TransNetV2: git clone https://github.com/soCzech/TransNetV2.git\n"
        "2. Ensure tensorflow and ffmpeg-python are installed: pip install tensorflow ffmpeg-python\n"
        "3. Or set environment variable TRANSNET_DIR to the 'TransNetV2/inference' directory."
    )


def detect_scenes_transnet(
    video_path: str,
    threshold: float = 0.5,
    min_scene_len_sec: float = 0.5,
    model: Any = None,
) -> list[tuple[float, float]]:
    """Detects scene transitions using deep-learning based TransNetV2.

    Args:
        video_path: Path to the video file on disk.
        threshold: Prediction probability threshold (e.g., 0.5 or 0.7).
        min_scene_len_sec: Minimum scene duration in seconds to keep.
        model: Optional pre-instantiated TransNetV2 instance.

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    if model is None:
        transnet_cls = _load_transnetv2_class()
        logger.info("Initializing TransNetV2 model weights...")
        model = transnet_cls()

    logger.info(f"Running TransNetV2 inference on {video_path}...")
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)

    # Convert single frame predictions to scene tuples of (start_frame, end_frame)
    scenes_frames = model.predictions_to_scenes(single_frame_predictions, threshold=threshold)

    meta = probe_video_metadata(video_path)
    fps = meta.get("fps") or 25.0
    duration_sec = meta.get("duration_sec") or 0.0

    if not scenes_frames:
        return [(0.0, float(duration_sec))]

    results: list[tuple[float, float]] = []
    for s_frame, e_frame in scenes_frames:
        start_sec = round(float(s_frame) / fps, 3)
        end_sec = round(float(e_frame) / fps, 3)

        if duration_sec and end_sec > duration_sec:
            end_sec = round(duration_sec, 3)

        if (end_sec - start_sec) >= min_scene_len_sec:
            results.append((start_sec, end_sec))

    if not results and duration_sec > 0:
        results.append((0.0, float(duration_sec)))

    return results


def preprocess_media_transnet(
    media_item: MediaItem,
    storage: StorageBackend,
    threshold: float = 0.5,
    chunk_threshold_sec: float = 45.0,
    window_sec: float = 10.0,
    overlap_ratio: float = 0.5,
    transnet_model: Any = None,
) -> list[ChunkCandidate]:
    """Generates ChunkCandidates using TransNetV2 scene detection and sliding window split for long scenes."""
    if media_item.media_type in (MediaType.IMAGE, "image"):
        return [ChunkCandidate(start_ts=0.0, end_ts=None, media_type="image")]

    try:
        local_path = storage.get_local_path(media_item.storage_path)
    except Exception:
        local_path = storage.get_local_path(media_item.source_url)

    scenes = detect_scenes_transnet(local_path, threshold=threshold, model=transnet_model)

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
            sub_chunks = sliding_window_split(
                start_ts=scene_start,
                end_ts=scene_end,
                window_sec=window_sec,
                overlap_ratio=overlap_ratio,
            )
            chunks.extend(sub_chunks)

    return chunks


def extract_clip_fast(
    video_path: str,
    start_ts: float,
    end_ts: float,
    output_path: str,
    stream_copy: bool = False,
    use_nvenc: bool = False,
) -> str:
    """Extracts a video sub-clip using fast keyframe seek (-ss before -i).

    Why it is fast:
    - Placing '-ss' BEFORE '-i' enables demuxer-level fast seeking, skipping decoding of earlier frames.
    - If stream_copy=True, avoids re-encoding completely (-c copy).
    - If re-encoding is needed, uses '-preset veryfast' or NVIDIA NVENC for hardware acceleration.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    dur = max(0.1, end_ts - start_ts)
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", f"{start_ts:.3f}",  # Fast seek BEFORE -i
        "-i", video_path,
        "-t", f"{dur:.3f}",
    ]

    if stream_copy:
        cmd.extend(["-c", "copy"])
    elif use_nvenc:
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-c:a", "aac",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-c:a", "aac",
        ])

    cmd.extend(["-movflags", "+faststart", output_path])

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.warning(f"ffmpeg error for clip {start_ts:.2f}-{end_ts:.2f}: {res.stderr[:200]}")
        raise RuntimeError(f"FFmpeg error: {res.stderr}")

    return output_path


def parallel_extract_clips(
    video_path: str,
    clips: list[tuple[int, float, float, str]],
    max_workers: int = 4,
    stream_copy: bool = False,
    use_nvenc: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[tuple[int, str, bool]]:
    """Extracts multiple video clips concurrently using ThreadPoolExecutor.

    Args:
        video_path: Path to the original full video.
        clips: List of tuples (index, start_sec, end_sec, output_path).
        max_workers: Number of parallel FFmpeg processes (threads).
        stream_copy: If True, uses '-c copy' for instant splitting.
        use_nvenc: If True, uses GPU hardware encoder 'h264_nvenc'.
        progress_callback: Optional callback func(completed_count, total_count).

    Returns:
        List of (index, output_path, success_bool).
    """
    total = len(clips)
    results: list[tuple[int, str, bool]] = []
    completed = 0

    def _worker(item: tuple[int, float, float, str]):
        idx, s, e, out = item
        try:
            extract_clip_fast(
                video_path=video_path,
                start_ts=s,
                end_ts=e,
                output_path=out,
                stream_copy=stream_copy,
                use_nvenc=use_nvenc,
            )
            return idx, out, True
        except Exception as err:
            logger.error(f"Clip {idx} ({s:.2f}-{e:.2f}) failed: {err}")
            return idx, out, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_clip = {executor.submit(_worker, c): c for c in clips}
        for future in concurrent.futures.as_completed(future_to_clip):
            res = future.result()
            results.append(res)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    results.sort(key=lambda x: x[0])
    return results
