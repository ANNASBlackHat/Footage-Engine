"""Qwen-VL caption backend for footage chunks.

Generates a rich visual caption + semantic tags for each chunk and writes them
to ``Chunk.caption`` / ``Chunk.tags``. Distinct from the embedding backends in
``footage_engine.embeddings`` — those return 2048-dim vectors, this one returns
text.

Mirrors the loading path from ``video-generation-pipeline/scripts/caption_chunks.py``
but reuses ``sample_frames_from_video`` from ``footage_engine.embeddings.frames``
so frame extraction stays a single code path.
"""

import logging
from typing import Optional

from PIL import Image

from footage_engine.embeddings.frames import sample_frames_from_video

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert video analyst for a documentary footage catalog.\n"
    "Analyze the provided frames from this {duration:.1f}-second video clip.\n\n"
    "Generate an objective, high-precision visual caption and semantic tags in this EXACT format:\n"
    "CAPTION: <A detailed 1-2 sentence description specifying the subject/creature, physical "
    "action, environment/setting, camera perspective/movement, and lighting/visibility>\n"
    "TAGS: <3 to 6 comma-separated lowercase keywords, e.g. underwater, submersible, deep sea, "
    "murky, oceanic trench>"
)

MAX_IMAGE_SIDE = 448


def parse_model_response(raw_text: str) -> tuple[str, list[str]]:
    """Extracts (caption, tags) from a Qwen response.

    Parses ``CAPTION:`` / ``TAGS:`` header lines; falls back to the raw text
    plus a keyword slice of the caption when the model omitted the headers.
    Ported verbatim from ``video-generation-pipeline/scripts/caption_chunks.py``.
    """
    caption = ""
    tags: list[str] = []
    lines = raw_text.strip().splitlines()

    for line in lines:
        line_clean = line.strip()
        if line_clean.upper().startswith("CAPTION:"):
            caption = line_clean[len("CAPTION:") :].strip()
        elif line_clean.upper().startswith("TAGS:"):
            raw_tags = line_clean[len("TAGS:") :].strip()
            tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]

    if not caption:
        caption = raw_text.strip()
    if not tags:
        words = [w.strip(".,;:\"'").lower() for w in caption.split() if len(w) > 3]
        tags = list(dict.fromkeys(words))[:5]

    return caption, tags


class QwenCaptioner:
    """Multimodal video captioner using a Qwen3-VL Instruct VLM.

    Loads the model lazily so the optional ``qwen`` extra (transformers,
    torch, ...) is not required for the rest of the engine. ``model_name``
    defaults to the pipeline's caption model, ``Qwen/Qwen3-VL-2B-Instruct``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        version: str = "1.0",
        device: str = "auto",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        num_frames: int = 3,
        max_new_tokens: int = 256,
    ):
        if torch is None:
            raise ImportError(
                "torch is required for QwenCaptioner. "
                'Install with: pip install -e ".[qwen]"'
            )

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_name = model_name
        self.version = version
        self.prompt_template = prompt_template
        self.num_frames = num_frames
        self.max_new_tokens = max_new_tokens

        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = device
        self.device = resolved

        logger.info(f"Loading Qwen caption model '{model_name}' on device '{resolved}'...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    # ------------------------------------------------------------------ API

    def caption_chunk(
        self,
        video_path: str,
        start_ts: float = 0.0,
        end_ts: Optional[float] = None,
        num_frames: Optional[int] = None,
    ) -> tuple[str, list[str]]:
        """Samples frames and returns (caption, tags) for the segment."""
        frames = self._extract_frames(video_path, start_ts, end_ts, num_frames)
        if not frames:
            return "", []

        dur = max(0.1, (end_ts or start_ts + 1.0) - start_ts)
        prompt = self.prompt_template.format(duration=dur)

        content: list[dict] = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        import torch

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )

        generated_ids = [
            out[len(in_ids) :] for in_ids, out in zip(inputs.input_ids, output_ids)
        ]
        raw = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return parse_model_response(raw)

    # ----------------------------------------------------------- internals

    def _extract_frames(
        self,
        video_path: str,
        start_ts: float,
        end_ts: Optional[float],
        num_frames: Optional[int],
    ) -> list[Image.Image]:
        n = num_frames or self.num_frames
        frames = sample_frames_from_video(
            video_path=video_path,
            start_ts=start_ts,
            end_ts=end_ts,
            num_frames=n,
        )
        # Downscale oversized frames the way the pipeline does, so the VLM
        # input stays bounded regardless of source resolution.
        import numpy as np

        for i, img in enumerate(frames):
            w, h = img.size
            if max(w, h) > MAX_IMAGE_SIDE:
                scale = MAX_IMAGE_SIDE / max(w, h)
                frames[i] = img.resize(
                    (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
                )
        return frames
