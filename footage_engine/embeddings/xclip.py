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

        inputs = self.processor(videos=[frames], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            video_features = self.model.get_video_features(**inputs)
            video_features = self._extract_tensor(video_features)
            # L2 normalize
            normalized = video_features / video_features.norm(p=2, dim=-1, keepdim=True)

        return normalized.squeeze(0).cpu().tolist()

    def embed_image(self, image: str | Image.Image) -> list[float]:
        """Generate a 512-dim embedding for a single image / video frame."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        frames = [img] * 8
        inputs = self.processor(videos=[frames], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            video_features = self.model.get_video_features(**inputs)
            video_features = self._extract_tensor(video_features)
            normalized = video_features / video_features.norm(p=2, dim=-1, keepdim=True)

        return normalized.squeeze(0).cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embeds natural language text query into 512-d shared space."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = self._extract_tensor(text_features)
            normalized = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return normalized.squeeze(0).cpu().tolist()
