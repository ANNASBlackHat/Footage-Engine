"""Unsplash provider adapter."""

import logging
from typing import Optional
import requests

from footage_engine.config import get_settings
from footage_engine.sources.base import Candidate

logger = logging.getLogger(__name__)


class UnsplashAdapter:
    """Searches Unsplash for still photos.

    Unsplash only hosts images, so video searches return no candidates.
    Per the API guidelines the image URLs returned by the API must be used
    directly (hotlinked) and the ``ixid`` parameter must be preserved.
    """

    name: str = "unsplash"
    BASE_URL: str = "https://api.unsplash.com/search/photos"
    TRACK_DOWNLOAD_URL: str = "https://api.unsplash.com/photos/{id}/download"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_settings().UNSPLASH_ACCESS_KEY

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Client-ID {self.api_key}",
            "Accept-Version": "v1",
        }

    def _normalize_orientation(self, orientation: Optional[str]) -> str:
        if not orientation:
            return "all"
        o = orientation.lower()
        if o in ("landscape", "horizontal"):
            return "landscape"
        if o in ("portrait", "vertical"):
            return "portrait"
        if o in ("square", "squarish"):
            return "squarish"
        return "all"

    def search(
        self,
        keyword: str,
        max_results: int = 20,
        media_type: str = "image",
        orientation: Optional[str] = None,
    ) -> list[Candidate]:
        if media_type not in ("image", "photo", "all"):
            logger.info("Unsplash does not host videos; skipping for video search.")
            return []

        if not self.api_key:
            raise ValueError(
                "Unsplash access key not found. Set UNSPLASH_ACCESS_KEY in environment or .env"
            )

        norm_orientation = self._normalize_orientation(orientation)
        params = {
            "query": keyword,
            "per_page": min(max(max_results, 1), 30),
        }
        if norm_orientation != "all":
            params["orientation"] = norm_orientation

        try:
            resp = requests.get(
                self.BASE_URL, headers=self._headers(), params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Unsplash search failed for query '{keyword}': {e}")
            raise

        candidates: list[Candidate] = []
        for photo in data.get("results", []):
            photo_id = str(photo.get("id"))
            urls = photo.get("urls", {})
            img_url = urls.get("regular") or urls.get("full") or urls.get("raw")
            if not img_url:
                continue

            width = photo.get("width")
            height = photo.get("height")
            resolution = f"{width}x{height}" if width and height else None
            links = photo.get("links", {})
            user = photo.get("user", {})

            candidates.append(
                Candidate(
                    provider=self.name,
                    source_id=photo_id,
                    source_url=img_url,
                    license_type="unsplash_free",
                    media_type="image",
                    duration_sec=None,
                    resolution=resolution,
                    metadata={
                        "page_url": links.get("html"),
                        "download_location": links.get("download_location"),
                        "photographer": user.get("name"),
                        "photographer_username": user.get("username"),
                        "photographer_url": (user.get("links") or {}).get("html"),
                        "alt_description": photo.get("alt_description"),
                        "description": photo.get("description"),
                        "color": photo.get("color"),
                        "blur_hash": photo.get("blur_hash"),
                        "raw_url": urls.get("raw"),
                    },
                )
            )
            if len(candidates) >= max_results:
                break

        return candidates

    def track_download(
        self, source_id: str, download_location: Optional[str] = None
    ) -> bool:
        """Registers a download event with Unsplash, as required by their API guidelines.

        Call this once each time the application actually downloads a photo.
        """
        if not self.api_key:
            raise ValueError(
                "Unsplash access key not found. Set UNSPLASH_ACCESS_KEY in environment or .env"
            )

        url = download_location or self.TRACK_DOWNLOAD_URL.format(id=source_id)
        try:
            resp = requests.get(url, headers=self._headers(), timeout=15)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Unsplash download tracking failed for photo '{source_id}': {e}")
            return False
