"""Direct URL list adapter."""

import os
import urllib.parse
from typing import Any
from footage_engine.sources.base import Candidate


class DirectURLAdapter:
    """Handles direct URL lists for video or image media items."""

    name: str = "direct"

    @staticmethod
    def url_to_candidate(
        url: str,
        provider: str = "manual",
        source_id: str | None = None,
        media_type: str = "video",
        metadata: dict[str, Any] | None = None,
    ) -> Candidate:
        from footage_engine.sources.youtube import YouTubeAdapter, is_youtube_url

        if is_youtube_url(url):
            try:
                adapter = YouTubeAdapter()
                return adapter.url_to_candidate(url, metadata=metadata)
            except Exception:
                from footage_engine.sources.youtube import extract_youtube_video_id, normalize_youtube_url
                vid = extract_youtube_video_id(url)
                return Candidate(
                    provider="youtube",
                    source_id=vid,
                    source_url=normalize_youtube_url(url),
                    license_type="youtube_standard",
                    media_type="video",
                    metadata=metadata or {},
                )

        # Determine media type if not explicitly set
        clean_path = urllib.parse.urlparse(url).path.lower()
        if clean_path.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            media_type = "image"
        elif clean_path.endswith((".mp4", ".mov", ".mkv", ".webm", ".avi")):
            media_type = "video"

        return Candidate(
            provider=provider,
            source_id=source_id,
            source_url=url,
            license_type="unknown",
            media_type=media_type,
            metadata=metadata or {},
        )
