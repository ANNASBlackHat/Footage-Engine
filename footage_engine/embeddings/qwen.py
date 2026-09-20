"""Qwen3-VL embedding backend (Qwen/Qwen3-VL-Embedding-2B, 2048 dimensions).

Additive second backend: implements the existing EmbeddingBackend protocol so it
is interchangeable with XCLIPEmbedder. Validated on Colab T4 (see
experiments/embed_compare.py and experiments/embed_compare_video.py).
"""

import logging
from typing import Optional
from PIL import Image

from footage_engine.embeddings.frames import sample_frames_from_video

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class QwenEmbedder:
    """Multimodal video, image, and text embedder using Qwen3-VL-Embedding-2B."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        version: str = "1.0",
        device: str = "auto",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required for QwenEmbedder. "
                'Install with: pip install -e ".[qwen]"'
            ) from err

        if torch is None:
            raise ImportError("torch is required for QwenEmbedder.")

        self.model_name = model_name
        self.version = version
        self.dimension = 2048

        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = device
        self.device = resolved

        logger.info(f"Loading Qwen embedding model '{model_name}' on device '{resolved}'...")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=resolved,
            model_kwargs={"dtype": "float16" if resolved == "cuda" else "float32"},
        )

    def embed_image(self, image: str | Image.Image) -> list[float]:
        """Generate a 2048-dim embedding for a single image / video frame."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        out = self.model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
        return [float(x) for x in out[0]]

    def _embed_frames_mean(self, frames: list[Image.Image]) -> list[float]:
        import numpy as np

        vecs = self.model.encode(
            [f.convert("RGB") for f in frames],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=4,
        )
        mean = np.mean(np.asarray(vecs, dtype=float), axis=0)
        norm = float(np.linalg.norm(mean)) or 1.0
        return [float(x / norm) for x in mean]

    def embed_video(
        self,
        video_path: str,
        start_ts: float = 0.0,
        end_ts: Optional[float] = None,
        num_frames: int = 8,
    ) -> list[float]:
        """Samples frames and computes a normalized 2048-d clip embedding (mean-pool)."""
        frames = sample_frames_from_video(
            video_path=video_path,
            start_ts=start_ts,
            end_ts=end_ts,
            num_frames=num_frames,
        )
        return self._embed_frames_mean(frames)

    def embed_video_batch(
        self,
        video_path: str,
        chunk_ranges: list[tuple[float, Optional[float]]],
        batch_size: int = 8,
        num_frames: int = 8,
    ) -> list[list[float]]:
        """Embeds multiple video segments with shared VideoCapture and batched GPU inference.

        Instead of opening the video N times (once per chunk), this opens it once,
        sorts all target timestamps for forward-only seeking, extracts frames in one
        pass, then batch-encodes all frames through the model.  Results are split
        back per chunk and mean-pooled.
        """
        if not chunk_ranges:
            return []

        import numpy as np

        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required for batch video embedding.")

        num_chunks = len(chunk_ranges)

        # 1. Compute per-chunk target timestamps and a flat sorted schedule
        #    schedule_items: (abs_ts, chunk_idx, frame_idx_within_chunk)
        schedule_items: list[tuple[float, int, int]] = []
        chunk_frame_lists: list[list[float]] = [[] for _ in range(num_chunks)]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video at: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = float(total_frames / fps) if fps > 0 else 0.0

            for ci, (start_ts, end_ts) in enumerate(chunk_ranges):
                actual_start = max(0.0, start_ts)
                actual_end = min(duration, end_ts) if end_ts is not None and end_ts > 0 else duration
                if actual_end <= actual_start:
                    actual_end = actual_start + 1.0
                segment_dur = actual_end - actual_start
                for fi in range(num_frames):
                    ts = actual_start + (fi + 0.5) * (segment_dur / num_frames)
                    schedule_items.append((ts, ci, fi))

            # Sort by timestamp so we seek forward through the video
            schedule_items.sort(key=lambda x: x[0])

            # 2. Single-pass frame extraction — one VideoCapture, sorted seek
            all_frames: list[Optional[Image.Image]] = [None] * len(schedule_items)
            last_valid: Optional[Image.Image] = None

            for slot_idx, (abs_ts, ci, fi) in enumerate(schedule_items):
                cap.set(cv2.CAP_PROP_POS_MSEC, abs_ts * 1000.0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    last_valid = pil_img
                    all_frames[slot_idx] = pil_img
                elif last_valid is not None:
                    all_frames[slot_idx] = last_valid

        finally:
            cap.release()

        # 3. Fill any remaining None slots with black placeholder
        placeholder = Image.new("RGB", (224, 224), color=(0, 0, 0))
        for i in range(len(all_frames)):
            if all_frames[i] is None:
                all_frames[i] = placeholder

        # 4. Group frames back per chunk for model input
        #    Build a flat list of all frames and track per-chunk slices
        chunk_slices: list[tuple[int, int]] = []  # (start, end) indices into flat_frames
        flat_frames: list[Image.Image] = []
        frame_cursor = 0
        for ci in range(num_chunks):
            start = frame_cursor
            for slot_idx, (_, ci2, _) in enumerate(schedule_items):
                if ci2 == ci:
                    flat_frames.append(all_frames[slot_idx])
                    frame_cursor += 1
            chunk_slices.append((start, frame_cursor))

        # 5. Batch encode ALL frames through the model in one encode() call
        if not flat_frames:
            return [np.zeros(self.dimension).tolist() for _ in chunk_ranges]

        encoded = self.model.encode(
            [f.convert("RGB") for f in flat_frames],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
        )

        # 6. Split back per chunk and mean-pool
        results: list[list[float]] = []
        for ci in range(num_chunks):
            s, e = chunk_slices[ci]
            chunk_vecs = encoded[s:e]
            if len(chunk_vecs) == 0:
                results.append(np.zeros(self.dimension).tolist())
                continue
            mean = np.mean(np.asarray(chunk_vecs, dtype=float), axis=0)
            norm = float(np.linalg.norm(mean)) or 1.0
            results.append([float(x / norm) for x in mean])

        return results

    def embed_text(self, text: str) -> list[float]:
        """Embeds natural language text query into the 2048-d shared space."""
        prompts = getattr(self.model, "prompts", None) or {}
        out = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name="query" if "query" in prompts else None,
        )
        return [float(x) for x in out[0]]
