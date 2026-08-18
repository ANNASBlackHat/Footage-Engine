"""Pixabay provider adapter."""

import logging
from typing import Optional
import requests

from footage_engine.config import get_settings
from footage_engine.sources.base import Candidate

logger = logging.getLogger(__name__)


class PixabayAdapter:
    name: str = "pixabay"
    BASE_VIDEO_URL: str = "https://pixabay.com/api/videos/"
    BASE_IMAGE_URL: str = "https://pixabay.com/api/"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_settings().PIXABAY_API_KEY

    def _normalize_orientation(self, orientation: Optional[str]) -> str:
        if not orientation:
            return "horizontal"
        o = orientation.lower()
        if o in ("landscape", "horizontal"):
            return "horizontal"
        elif o in ("portrait", "vertical"):
            return "vertical"
        return "all"

    def search(
        self,
        keyword: str,
        max_results: int = 20,
        media_type: str = "video",
        orientation: Optional[str] = "horizontal",
    ) -> list[Candidate]:
        if not self.api_key:
            raise ValueError(
                "Pixabay API key not found. Set PIXABAY_API_KEY in environment or .env"
            )

        if media_type in ("image", "photo"):
            return self.search_photos(keyword=keyword, max_results=max_results, orientation=orientation)

        norm_orientation = self._normalize_orientation(orientation)
        params = {
            "key": self.api_key,
            "q": keyword,
            "per_page": min(max(max_results, 3), 200),
            "video_type": "all",
        }
        if norm_orientation != "all":
            params["orientation"] = norm_orientation

        try:
            resp = requests.get(self.BASE_VIDEO_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Pixabay video search failed for query '{keyword}': {e}")
            raise

        candidates: list[Candidate] = []
        for hit in data.get("hits", []):
            hit_id = str(hit.get("id"))
            videos = hit.get("videos", {})

            # Select best available video quality (prefer medium/small for fast transfer)
            selected_video = videos.get("medium") or videos.get("small") or videos.get("tiny") or videos.get("large")
            if not selected_video or not selected_video.get("url"):
                continue

            video_url = selected_video["url"]
            width = selected_video.get("width")
            height = selected_video.get("height")
            resolution = f"{width}x{height}" if width and height else None
            duration = float(hit.get("duration", 0))

            candidate = Candidate(
                provider=self.name,
                source_id=hit_id,
                source_url=video_url,
                license_type="pixabay_free",
                media_type="video",
                duration_sec=duration if duration > 0 else None,
                resolution=resolution,
                metadata={
                    "tags": [t.strip() for t in hit.get("tags", "").split(",") if t.strip()],
                    "page_url": hit.get("pageURL"),
                    "views": hit.get("views"),
                    "downloads": hit.get("downloads"),
                },
            )
            candidates.append(candidate)
            if len(candidates) >= max_results:
                break

        return candidates

    def search_photos(
        self,
        keyword: str,
        max_results: int = 20,
        orientation: Optional[str] = "horizontal",
    ) -> list[Candidate]:
        """Search Pixabay for still photos/images."""
        if not self.api_key:
            raise ValueError(
                "Pixabay API key not found. Set PIXABAY_API_KEY in environment or .env"
            )

        norm_orientation = self._normalize_orientation(orientation)
        params = {
            "key": self.api_key,
            "q": keyword,
            "per_page": min(max(max_results, 3), 200),
            "image_type": "photo",
        }
        if norm_orientation != "all":
            params["orientation"] = norm_orientation

        try:
            resp = requests.get(self.BASE_IMAGE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Pixabay photo search failed for query '{keyword}': {e}")
            raise

        candidates: list[Candidate] = []
        for hit in data.get("hits", []):
            hit_id = str(hit.get("id"))
            img_url = hit.get("largeImageURL") or hit.get("webformatURL") or hit.get("imageURL")
            if not img_url:
                continue

            width = hit.get("imageWidth")
            height = hit.get("imageHeight")
            resolution = f"{width}x{height}" if width and height else None

            candidates.append(
                Candidate(
                    provider=self.name,
                    source_id=hit_id,
                    source_url=img_url,
                    license_type="pixabay_free",
                    media_type="image",
                    duration_sec=None,
                    resolution=resolution,
                    metadata={
                        "tags": [t.strip() for t in hit.get("tags", "").split(",") if t.strip()],
                        "page_url": hit.get("pageURL"),
                        "views": hit.get("views"),
                        "downloads": hit.get("downloads"),
                    },
                )
            )
            if len(candidates) >= max_results:
                break

        return candidates
