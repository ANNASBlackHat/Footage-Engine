"""Video frame extraction utilities."""

import logging
from typing import Optional
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


def sample_frames_from_video(
    video_path: str,
    start_ts: float = 0.0,
    end_ts: Optional[float] = None,
    num_frames: int = 8,
) -> list[Image.Image]:
    """Uniformly extracts N frames from a video segment [start_ts, end_ts] as PIL Images."""
    if cv2 is None:
        raise ImportError("opencv-python is required for frame extraction.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video at: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = float(total_frames / fps) if fps > 0 else 0.0

        actual_start = max(0.0, start_ts)
        actual_end = min(duration, end_ts) if end_ts is not None and end_ts > 0 else duration

        if actual_end <= actual_start:
            actual_end = actual_start + 1.0

        segment_dur = actual_end - actual_start
        # Compute evenly spaced target timestamps
        target_timestamps = [
            actual_start + (i + 0.5) * (segment_dur / num_frames)
            for i in range(num_frames)
        ]

        frames: list[Image.Image] = []
        last_valid_frame = None

        for ts in target_timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                last_valid_frame = pil_img
                frames.append(pil_img)
            elif last_valid_frame is not None:
                # Fallback to duplicate last valid frame
                frames.append(last_valid_frame)

        # If no frames could be decoded, create placeholder black frames
        if not frames:
            logger.warning(f"No frames could be extracted from {video_path}. Using fallback.")
            placeholder = Image.new("RGB", (224, 224), color=(0, 0, 0))
            frames = [placeholder] * num_frames
        elif len(frames) < num_frames:
            # Pad with the last frame if short
            while len(frames) < num_frames:
                frames.append(frames[-1])

        return frames
    finally:
        cap.release()
