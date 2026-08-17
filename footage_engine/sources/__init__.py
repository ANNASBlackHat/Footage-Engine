"""Sources package."""

from footage_engine.sources.base import Candidate, SourceAdapter
from footage_engine.sources.pixabay import PixabayAdapter
from footage_engine.sources.pexels import PexelsAdapter
from footage_engine.sources.coverr import CoverrAdapter
from footage_engine.sources.direct import DirectURLAdapter
from footage_engine.sources.protocol import IterableURLProvider, URLProvider
from footage_engine.sources.youtube import (
    YouTubeAdapter,
    extract_youtube_video_id,
    is_youtube_url,
    normalize_youtube_url,
)

__all__ = [
    "Candidate",
    "SourceAdapter",
    "PixabayAdapter",
    "PexelsAdapter",
    "CoverrAdapter",
    "DirectURLAdapter",
    "YouTubeAdapter",
    "URLProvider",
    "IterableURLProvider",
    "is_youtube_url",
    "extract_youtube_video_id",
    "normalize_youtube_url",
]
