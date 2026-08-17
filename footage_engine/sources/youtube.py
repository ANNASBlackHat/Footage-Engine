"""YouTube provider adapter powered by yt-dlp with optional cookie authentication."""

import base64
import hashlib
import logging
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
from typing import Any, Optional

import requests

from footage_engine.config import get_settings
from footage_engine.sources.base import Candidate

logger = logging.getLogger(__name__)

YOUTUBE_REGEX = re.compile(
    r"(?:https?:\/\/)?(?:www\.|m\.)?(?:youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([\w-]{11})",
    re.IGNORECASE,
)


def is_youtube_url(url: str) -> bool:
    """Checks whether a URL is a YouTube video, shorts, or embed link."""
    if not url or not isinstance(url, str):
        return False
    return bool(YOUTUBE_REGEX.search(url))


def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extracts the 11-character YouTube video ID from a URL."""
    if not url or not isinstance(url, str):
        return None
    match = YOUTUBE_REGEX.search(url)
    if match:
        return match.group(1)
    return None


def normalize_youtube_url(url: str) -> str:
    """Converts any valid YouTube URL variant into canonical https://www.youtube.com/watch?v=VIDEO_ID."""
    vid = extract_youtube_video_id(url)
    if vid:
        return f"https://www.youtube.com/watch?v={vid}"
    return url


def resolve_cookie_source(source: Optional[str], cache_dir: Optional[str] = None) -> Optional[str]:
    """Resolves a cookie source into a local file path usable by yt-dlp.
    
    Supported source formats:
      1. Direct local file path (e.g. '/path/to/cookies.txt', './cookies.txt')
      2. Remote HTTP/HTTPS URL (e.g. 'https://my-bucket/cookies.txt' -> downloaded & cached)
      3. Raw Netscape cookie content string (contains '# Netscape' or tab-separated cookie lines)
      4. Base64-encoded cookie string
    """
    if not source or not isinstance(source, str):
        return None

    trimmed = source.strip()
    if not trimmed:
        return None

    # Case 1: Remote HTTP/HTTPS URL
    if trimmed.startswith(("http://", "https://")):
        target_dir = Path(cache_dir or os.path.join(tempfile.gettempdir(), "footage_engine_cookies"))
        target_dir.mkdir(parents=True, exist_ok=True)
        url_hash = hashlib.sha256(trimmed.encode("utf-8")).hexdigest()[:12]
        cached_cookie_path = target_dir / f"cookies_{url_hash}.txt"

        try:
            resp = requests.get(trimmed, timeout=30)
            resp.raise_for_status()
            with open(cached_cookie_path, "wb") as f:
                f.write(resp.content)
            return str(cached_cookie_path)
        except Exception as e:
            logger.error(f"Failed to download cookies from URL '{trimmed}': {e}")
            if cached_cookie_path.exists():
                logger.info(f"Falling back to previously cached cookies at {cached_cookie_path}")
                return str(cached_cookie_path)
            return None

    # Case 2: Direct local file path
    if os.path.isfile(trimmed):
        return os.path.abspath(trimmed)

    # Case 3: Raw Netscape cookie string or tab-separated cookie content
    if "# Netscape" in trimmed or "\t" in trimmed or "\n" in trimmed:
        tmp = tempfile.NamedTemporaryFile(suffix="_cookies.txt", delete=False, mode="w", encoding="utf-8")
        tmp.write(trimmed)
        tmp.close()
        return tmp.name

    # Case 4: Base64-encoded cookie file string
    try:
        decoded_bytes = base64.b64decode(trimmed, validate=True)
        decoded_text = decoded_bytes.decode("utf-8", errors="ignore")
        if "# Netscape" in decoded_text or "\t" in decoded_text or "\n" in decoded_text:
            tmp = tempfile.NamedTemporaryFile(suffix="_cookies.txt", delete=False, mode="wb")
            tmp.write(decoded_bytes)
            tmp.close()
            return tmp.name
    except Exception:
        pass

    return None


class YouTubeAdapter:
    """Source adapter for querying, extracting metadata, and streaming/downloading YouTube videos.
    
    Supports optional cookies via:
      - `cookies`: local file path, remote download URL, raw Netscape text, or base64 string
      - `cookies_from_browser`: name of browser to extract active session cookies from (e.g. 'chrome', 'firefox', 'brave')
    """

    name: str = "youtube"

    def __init__(
        self,
        cookies: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
        cookies_file: Optional[str] = None,  # Backward compatibility alias
    ):
        settings = get_settings()
        raw_cookie_source = cookies or cookies_file or settings.YOUTUBE_COOKIES
        self.cookies_source = raw_cookie_source
        self.cookies_from_browser = cookies_from_browser or settings.YOUTUBE_COOKIES_FROM_BROWSER
        self.resolved_cookie_file = resolve_cookie_source(self.cookies_source)

    def _get_ydl_opts(self, extra_opts: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "no_color": True,
            "noplaylist": True,
            "remote_components": ["ejs:github"],
        }

        # Auto-detect deno JS runtime if available
        deno_bin = os.path.expanduser("~/.deno/bin/deno")
        if os.path.exists(deno_bin):
            opts["js_runtimes"] = {"deno": {"path": deno_bin}}

        # Apply cookie file if resolved
        if self.resolved_cookie_file and os.path.exists(self.resolved_cookie_file):
            opts["cookiefile"] = self.resolved_cookie_file

        # Apply browser cookie extraction if configured
        if self.cookies_from_browser:
            opts["cookiesfrombrowser"] = (self.cookies_from_browser,)

        if extra_opts:
            opts.update(extra_opts)
        return opts

    def extract_info(self, url: str) -> dict[str, Any]:
        """Extracts complete video metadata without downloading the video payload."""
        import yt_dlp

        opts = self._get_ydl_opts({"skip_download": True})
        canonical_url = normalize_youtube_url(url)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(canonical_url, download=False)
            return info or {}

    def url_to_candidate(self, url: str, metadata: Optional[dict[str, Any]] = None) -> Candidate:
        """Extracts metadata from a YouTube URL and converts it to a standard Candidate."""
        canonical_url = normalize_youtube_url(url)
        video_id = extract_youtube_video_id(url) or ""

        try:
            info = self.extract_info(canonical_url)
        except Exception as e:
            logger.warning(f"Failed to fetch detailed info for YouTube video {url}: {e}")
            info = {}

        vid = info.get("id") or video_id
        duration = float(info.get("duration", 0.0) or 0.0)
        width = info.get("width")
        height = info.get("height")
        resolution = f"{width}x{height}" if width and height else None
        title = info.get("title", "")
        license_str = str(info.get("license") or "")
        license_type = "creative_commons" if "creative commons" in license_str.lower() else "youtube_standard"

        item_meta = {
            "title": title,
            "description": info.get("description"),
            "uploader": info.get("uploader") or info.get("channel"),
            "channel_id": info.get("channel_id"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "tags": info.get("tags") or [],
            "categories": info.get("categories") or [],
            "thumbnail": info.get("thumbnail"),
        }
        if metadata:
            item_meta.update(metadata)

        return Candidate(
            provider="youtube",
            source_url=canonical_url,
            source_id=vid,
            license_type=license_type,
            media_type="video",
            duration_sec=duration if duration > 0 else None,
            resolution=resolution,
            metadata=item_meta,
        )

    def search(self, keyword: str, max_results: int = 10) -> list[Candidate]:
        """Searches YouTube using keyword query and returns standardized candidates."""
        import yt_dlp

        query = f"ytsearch{max_results}:{keyword}"
        opts = self._get_ydl_opts({"skip_download": True})
        candidates: list[Candidate] = []

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                result = ydl.extract_info(query, download=False)
                entries = result.get("entries", []) if result else []
                for entry in entries:
                    if not entry:
                        continue
                    video_id = entry.get("id", "")
                    if not video_id:
                        continue
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    duration = float(entry.get("duration", 0.0) or 0.0)
                    w, h = entry.get("width"), entry.get("height")
                    res = f"{w}x{h}" if w and h else None
                    cand = Candidate(
                        provider="youtube",
                        source_url=url,
                        source_id=video_id,
                        license_type="unknown",
                        media_type="video",
                        duration_sec=duration if duration > 0 else None,
                        resolution=res,
                        metadata={
                            "title": entry.get("title", ""),
                            "uploader": entry.get("uploader", "") or entry.get("channel", ""),
                            "view_count": entry.get("view_count"),
                            "description": entry.get("description", ""),
                            "thumbnail": entry.get("thumbnail"),
                        },
                    )
                    candidates.append(cand)
        except Exception as e:
            logger.error(f"YouTube search failed for keyword '{keyword}': {e}")
            raise
        return candidates

    def download_to_path(self, url: str, target_path: str) -> str:
        """Downloads YouTube video into target local file path (MP4 format)."""
        import yt_dlp

        target_dir = os.path.dirname(os.path.abspath(target_path))
        os.makedirs(target_dir, exist_ok=True)

        canonical_url = normalize_youtube_url(url)
        opts = self._get_ydl_opts({
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": target_path,
            "merge_output_format": "mp4",
            "overwrites": True,
        })
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([canonical_url])
        return target_path

    def download_bytes(self, url: str) -> bytes:
        """Downloads YouTube video and returns raw bytes in memory."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.download_to_path(url, tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
