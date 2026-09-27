"""X-CLIP embedding implementation using microsoft/xclip-base-patch32."""

import logging
from typing import Any, Optional
from PIL import Image

from footage_engine.embeddings.frames import sample_frames_from_video

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModel, AutoProcessor
except ImportError:
    torch = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoProcessor = None  # type: ignore


class XCLIPEmbedder:
    """Multimodal video, image, and text embedder using X-CLIP Base (512 dimensions)."""

    def __init__(
        self,
        model_name: str = "microsoft/xclip-base-patch32",
        version: str = "1.0",
        device: str = "auto",
    ):
        if torch is None or AutoModel is None:
            raise ImportError(
                "torch and transformers packages are required for XCLIPEmbedder. "
                "Install with: pip install torch transformers"
            )

        self.model_name = model_name
        self.version = version
        self.dimension = 512

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading X-CLIP model '{model_name}' on device '{self.device}'...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _extract_tensor(self, features: Any) -> "torch.Tensor":
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            return features.last_hidden_state[:, 0, :]
        if isinstance(features, (tuple, list)):
            return features[0]
        return features

    def _preprocess_video_frames(self, frames_batch: list[list[Any]]) -> "torch.Tensor":
        """Preprocesses a batch of video frame lists into a pixel_values tensor on the target device.

        Directly uses self.processor.image_processor (VideoMAEImageProcessor) to bypass
        transformers ProcessorMixin / XCLIPProcessor argument routing issues across versions.
        """
        # 1. Direct call to underlying image_processor (most robust across all transformers versions)
        if hasattr(self.processor, "image_processor"):
            inputs = self.processor.image_processor(frames_batch, return_tensors="pt")
            if "pixel_values" in inputs:
                return inputs["pixel_values"].to(self.device)

        # 2. Try processor with videos=
        try:
            inputs = self.processor(videos=frames_batch, return_tensors="pt")
            if "pixel_values" in inputs:
                return inputs["pixel_values"].to(self.device)
        except Exception:
            pass

        # 3. Try processor with images= (newer ProcessorMixin mapping)
        try:
            inputs = self.processor(images=frames_batch, return_tensors="pt")
            if "pixel_values" in inputs:
                return inputs["pixel_values"].to(self.device)
        except Exception:
            pass

        raise RuntimeError("Failed to extract pixel_values tensor from XCLIP processor.")

    def embed_video(
        self,
        video_path: str,
        start_ts: float = 0.0,
        end_ts: Optional[float] = None,
        num_frames: int = 8,
    ) -> list[float]:
        """Samples frames and computes normalized 512-d video embedding."""
        frames = sample_frames_from_video(
            video_path=video_path,
            start_ts=start_ts,
            end_ts=end_ts,
            num_frames=num_frames,
        )

        pixel_values = self._preprocess_video_frames([frames])

        with torch.no_grad():
            video_features = self.model.get_video_features(pixel_values=pixel_values)
            video_features = self._extract_tensor(video_features)
            # L2 normalize
            normalized = video_features / video_features.norm(p=2, dim=-1, keepdim=True)

        return normalized.squeeze(0).cpu().tolist()

    def embed_video_batch(
        self,
        video_path: str,
        chunk_ranges: list[tuple[float, Optional[float]]],
        batch_size: int = 8,
        num_frames: int = 8,
    ) -> list[list[float]]:
        """Samples frames and computes normalized 512-d video embeddings in batched forward passes."""
        if not chunk_ranges:
            return []

        all_vectors: list[list[float]] = []

        for i in range(0, len(chunk_ranges), batch_size):
            batch_slice = chunk_ranges[i : i + batch_size]
            batch_frames = [
                sample_frames_from_video(
                    video_path=video_path,
                    start_ts=s_ts,
                    end_ts=e_ts,
                    num_frames=num_frames,
                )
                for s_ts, e_ts in batch_slice
            ]

            pixel_values = self._preprocess_video_frames(batch_frames)

            with torch.no_grad():
                video_features = self.model.get_video_features(pixel_values=pixel_values)
                video_features = self._extract_tensor(video_features)
                # L2 normalize
                normalized = video_features / video_features.norm(p=2, dim=-1, keepdim=True)

            all_vectors.extend(normalized.cpu().tolist())

        return all_vectors

    def embed_image(self, image: str | Image.Image) -> list[float]:
        """Generate a 512-dim embedding for a single image / video frame."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        frames = [img] * 8
        pixel_values = self._preprocess_video_frames([frames])

        with torch.no_grad():
            video_features = self.model.get_video_features(pixel_values=pixel_values)
            video_features = self._extract_tensor(video_features)
            normalized = video_features / video_features.norm(p=2, dim=-1, keepdim=True)

        return normalized.squeeze(0).cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embeds natural language text query into 512-d shared space."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if "input_ids" in inputs:
                text_features = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
            else:
                text_features = self.model.get_text_features(**inputs)
            text_features = self._extract_tensor(text_features)
            normalized = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return normalized.squeeze(0).cpu().tolist()
