"""Coverr provider adapter."""

import logging
from typing import Optional
import requests

from footage_engine.config import get_settings
from footage_engine.sources.base import Candidate

logger = logging.getLogger(__name__)


class CoverrAdapter:
    name: str = "coverr"
    BASE_URL: str = "https://api.coverr.co/videos"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_settings().COVERR_API_KEY

    def _normalize_orientation(self, orientation: Optional[str]) -> str:
        if not orientation:
            return "landscape"
        o = orientation.lower()
        if o in ("landscape", "horizontal"):
            return "landscape"
        elif o in ("portrait", "vertical"):
            return "portrait"
        return "all"

    def search(
        self,
        keyword: str,
        max_results: int = 20,
        media_type: str = "video",
        orientation: Optional[str] = "landscape",
    ) -> list[Candidate]:
        if not self.api_key:
            # If no API key, log warning or provide fallback
            logger.warning("COVERR_API_KEY not configured.")
            return []

        if media_type in ("image", "photo"):
            logger.info("Coverr does not host still photos; skipping for image search.")
            return []

        norm_orientation = self._normalize_orientation(orientation)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        # Fetch slightly more to ensure enough items after orientation filter
        fetch_size = min(max(max_results * 2, 10), 50)
        params = {
            "query": keyword,
            "page_size": fetch_size,
            "urls": "true",
        }

        try:
            resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Coverr search failed for query '{keyword}': {e}")
            raise

        candidates: list[Candidate] = []
        hits = data.get("hits") or data.get("videos") or []
        for hit in hits:
            # Orientation filtering
            is_vert = hit.get("is_vertical")
            aspect = str(hit.get("aspect_ratio") or "")
            max_w = hit.get("max_width")
            max_h = hit.get("max_height")

            if norm_orientation == "landscape":
                if is_vert is True or aspect == "9:16" or (max_h and max_w and max_h > max_w):
                    continue
            elif norm_orientation == "portrait":
                if is_vert is False and aspect != "9:16" and not (max_h and max_w and max_h > max_w):
                    continue

            hit_id = str(hit.get("id") or hit.get("video_id") or hit.get("objectID"))
            urls = hit.get("urls", {})
            mp4_url = (
                urls.get("mp4")
                or urls.get("mp4_download")
                or urls.get("mp4_preview")
                or hit.get("video_url")
            )
            if not mp4_url:
                base_name = hit.get("base_filename")
                if base_name:
                    mp4_url = f"https://cdn.coverr.co/videos/{base_name}/1080p.mp4"

            if not mp4_url:
                continue

            duration_raw = hit.get("duration")
            duration_sec = float(duration_raw) if duration_raw is not None else None
            resolution = f"{max_w}x{max_h}" if max_w and max_h else None

            candidates.append(
                Candidate(
                    provider=self.name,
                    source_id=hit_id,
                    source_url=mp4_url,
                    license_type="coverr_free",
                    media_type="video",
                    duration_sec=duration_sec,
                    resolution=resolution,
                    metadata={
                        "title": hit.get("title"),
                        "tags": hit.get("tags", []),
                        "description": hit.get("description"),
                        "aspect_ratio": aspect or ("9:16" if is_vert else "16:9"),
                    },
                )
            )
            if len(candidates) >= max_results:
                break

        return candidates
