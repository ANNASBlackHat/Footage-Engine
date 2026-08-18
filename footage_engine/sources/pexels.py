"""Pexels provider adapter."""

import logging
from typing import Optional
import requests

from footage_engine.config import get_settings
from footage_engine.sources.base import Candidate

logger = logging.getLogger(__name__)


class PexelsAdapter:
    name: str = "pexels"
    BASE_URL: str = "https://api.pexels.com/videos/search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_settings().PEXELS_API_KEY

    def _normalize_orientation(self, orientation: Optional[str]) -> str:
        if not orientation:
            return "landscape"
        o = orientation.lower()
        if o in ("landscape", "horizontal"):
            return "landscape"
        elif o in ("portrait", "vertical"):
            return "portrait"
        elif o == "square":
            return "square"
        return "all"

    def search(
        self,
        keyword: str,
        max_results: int = 20,
        media_type: str = "video",
        orientation: Optional[str] = "landscape",
    ) -> list[Candidate]:
        if not self.api_key:
            raise ValueError(
                "Pexels API key not found. Set PEXELS_API_KEY in environment or .env"
            )

        if media_type in ("image", "photo"):
            return self.search_photos(keyword=keyword, max_results=max_results, orientation=orientation)

        norm_orientation = self._normalize_orientation(orientation)
        headers = {"Authorization": self.api_key}
        params = {
            "query": keyword,
            "per_page": min(max(max_results, 1), 80),
        }
        if norm_orientation != "all":
            params["orientation"] = norm_orientation

        try:
            resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Pexels search failed for query '{keyword}': {e}")
            raise

        candidates: list[Candidate] = []
        for video in data.get("videos", []):
            video_id = str(video.get("id"))
            duration = float(video.get("duration", 0))
            video_files = video.get("video_files", [])

            # Select preferred file (HD or SD with MP4 link)
            selected_file = None
            for vf in video_files:
                if vf.get("file_type") == "video/mp4" and vf.get("quality") == "hd":
                    selected_file = vf
                    break
            if not selected_file and video_files:
                selected_file = video_files[0]

            if not selected_file or not selected_file.get("link"):
                continue

            width = selected_file.get("width") or video.get("width")
            height = selected_file.get("height") or video.get("height")
            resolution = f"{width}x{height}" if width and height else None

            candidates.append(
                Candidate(
                    provider=self.name,
                    source_id=video_id,
                    source_url=selected_file["link"],
                    license_type="pexels_free",
                    media_type="video",
                    duration_sec=duration if duration > 0 else None,
                    resolution=resolution,
                    metadata={
                        "url": video.get("url"),
                        "user": video.get("user", {}).get("name"),
                        "tags": video.get("tags", []),
                    },
                )
            )
            if len(candidates) >= max_results:
                break

        return candidates

    def search_photos(
        self,
        keyword: str,
        max_results: int = 20,
        orientation: Optional[str] = "landscape",
    ) -> list[Candidate]:
        """Search Pexels for still photos/images."""
        if not self.api_key:
            raise ValueError("Pexels API key not found. Set PEXELS_API_KEY in environment or .env")

        norm_orientation = self._normalize_orientation(orientation)
        headers = {"Authorization": self.api_key}
        params = {
            "query": keyword,
            "per_page": min(max(max_results, 1), 80),
        }
        if norm_orientation != "all":
            params["orientation"] = norm_orientation

        try:
            resp = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Pexels photo search failed for query '{keyword}': {e}")
            raise

        candidates: list[Candidate] = []
        for photo in data.get("photos", []):
            photo_id = str(photo.get("id"))
            src = photo.get("src", {})
            img_url = src.get("large2x") or src.get("large") or src.get("original")
            if not img_url:
                continue

            width = photo.get("width")
            height = photo.get("height")
            resolution = f"{width}x{height}" if width and height else None

            candidates.append(
                Candidate(
                    provider=self.name,
                    source_id=photo_id,
                    source_url=img_url,
                    license_type="pexels_free",
                    media_type="image",
                    duration_sec=None,
                    resolution=resolution,
                    metadata={
                        "url": photo.get("url"),
                        "photographer": photo.get("photographer"),
                        "alt": photo.get("alt"),
                    },
                )
            )
            if len(candidates) >= max_results:
                break

        return candidates
