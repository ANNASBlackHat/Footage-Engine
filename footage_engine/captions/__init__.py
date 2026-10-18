"""Caption generation for footage chunks (writes Chunk.caption / Chunk.tags)."""

from footage_engine.captions.qwen_captioner import QwenCaptioner

__all__ = ["QwenCaptioner"]
