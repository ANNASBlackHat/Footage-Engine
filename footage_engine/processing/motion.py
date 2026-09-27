"""Motion scoring engine using normalized frame-difference variance.

Measures dynamic change across a video or chunk:
- motion_mean: average magnitude of frame-to-frame pixel differences.
  Higher = more overall movement in the scene.
- motion_std: standard deviation of frame-to-frame differences.
  Higher = varying dynamics across the clip (camera panning, subject entering).
  Near-zero = repetitive static loop (e.g. 1-second stock wave looping indefinitely).
"""

import logging
from typing import Optional, TypedDict
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionScoreResult(TypedDict):
    motion_mean: float
    motion_std: float


def compute_motion_score(
    video_source: str,
    start_ts: float = 0.0,
    end_ts: Optional[float] = None,
    sample_fps: int = 4,
    resize_width: int = 320,
    max_duration_sec: float = 30.0,
) -> Optional[MotionScoreResult]:
    """Calculates motion_mean and motion_std for a video or time window.

    Args:
        video_source: Local file path or direct HTTP stream URL.
        start_ts: Chunk start time in seconds (default: 0.0).
        end_ts: Chunk end time in seconds (default: None = read until end or max_duration_sec).
        sample_fps: Number of frames to sample per second (default: 4).
        resize_width: Frame downscaling width for fast processing (default: 320).
        max_duration_sec: Maximum seconds of video to process per chunk (default: 30.0).

    Returns:
        Dict with "motion_mean" and "motion_std" floats, or None if unreadable.
    """
    if not video_source or not str(video_source).strip():
        return None

    cap = cv2.VideoCapture(str(video_source))
    if not cap.isOpened():
        logger.warning(f"Could not open video source for motion scoring: {video_source[:80]}")
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        frame_interval = max(1, int(fps / sample_fps))

        # Seek to start timestamp
        if start_ts > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_ts * 1000.0)

        effective_end_ts = (
            min(end_ts, start_ts + max_duration_sec)
            if end_ts is not None and end_ts > start_ts
            else start_ts + max_duration_sec
        )

        prev_gray: Optional[np.ndarray] = None
        diffs: list[float] = []
        frame_idx = 0
        max_frames_to_read = int(max_duration_sec * fps) + 50

        while frame_idx < max_frames_to_read:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_sec = current_msec / 1000.0

            # Only check end boundary if timestamp is sane
            if 0 <= current_sec <= effective_end_ts + 5.0:
                if current_sec > effective_end_ts:
                    break

            if frame_idx % frame_interval == 0:
                h, w = frame.shape[:2]
                if w > 0 and h > 0:
                    scale = resize_width / float(w)
                    target_h = max(1, int(h * scale))
                    small = cv2.resize(frame, (resize_width, target_h), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                    if prev_gray is not None:
                        diff = float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))
                        diffs.append(diff)
                    prev_gray = gray

            frame_idx += 1

        if not diffs:
            return None

        diffs_arr = np.array(diffs, dtype=np.float32)
        return {
            "motion_mean": round(float(diffs_arr.mean()), 3),
            "motion_std": round(float(diffs_arr.std()), 3),
        }

    except Exception as exc:
        logger.warning(f"Error computing motion score for {video_source[:60]}: {exc}")
        return None
    finally:
        cap.release()
