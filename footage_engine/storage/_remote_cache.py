"""Shared helpers for resolving remote references to local cached files.

Every backend needs to turn a *reference* into something that OpenCV,
PySceneDetect and the embedding models can read from disk. That includes not
only files the backend itself stores, but also YouTube links and plain HTTP
URLs referenced by ``source_url``.

The retry/backoff behaviour, the ``Referer`` headers some hosts require, and
the AV1 refetch workaround used to be copy-pasted into each backend. They live
here once so adding a new backend does not mean adding a new copy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

# Media extensions that can be cached under their own name. Anything else is
# treated as a video and normalised to ``.mp4``.
KNOWN_MEDIA_EXTENSIONS = (
    ".mp4",
    ".webm",
    ".ogv",
    ".mov",
    ".mkv",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)

# Cached files smaller than this are treated as a failed/truncated download.
MIN_VALID_FILE_BYTES = 1000

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

# Some hosts reject requests that do not carry a matching Referer.
REFERER_OVERRIDES = (
    (("wikimedia.org", "wikipedia.org"), "https://commons.wikimedia.org/"),
    (("pexels.com",), "https://www.pexels.com/"),
)

_MAX_ATTEMPTS = 5
_DOWNLOAD_TIMEOUT_SEC = 60
_CHUNK_SIZE = 65536


def video_codec(path: str | Path) -> str:
    """Returns the fourcc codec tag of a local video file ('' if unreadable)."""
    try:
        import cv2
    except ImportError:
        return ""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return ""
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip().lower()
    finally:
        cap.release()


def _cached_filename(storage_path: str) -> str:
    """Derives a safe cache filename from an HTTP URL."""
    clean_name = storage_path.split("?")[0].split("/")[-1]
    if not clean_name.endswith(KNOWN_MEDIA_EXTENSIONS):
        clean_name += ".mp4"
    return clean_name


def _headers_for(url: str) -> dict[str, str]:
    """Browser-like headers, plus a Referer override when the host needs one."""
    headers = dict(BROWSER_HEADERS)
    for hosts, referer in REFERER_OVERRIDES:
        if any(host in url for host in hosts):
            headers["Referer"] = referer
            break
    return headers


def _download_to_cache(url: str, cached_file: Path) -> Path:
    """Streams ``url`` into ``cached_file``, retrying transient failures."""
    headers = _headers_for(url)
    tmp_file = cached_file.with_suffix(cached_file.suffix + ".tmp")

    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC)
            if resp.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "").lower()
            if "text/html" in ctype or "text/plain" in ctype:
                time.sleep(3 * (attempt + 1))
                continue
            with open(tmp_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            if tmp_file.stat().st_size > MIN_VALID_FILE_BYTES:
                tmp_file.replace(cached_file)
                return cached_file
        except Exception as e:
            if attempt == _MAX_ATTEMPTS - 1:
                if tmp_file.exists():
                    tmp_file.unlink()
                raise e
            time.sleep(3 * (attempt + 1))

    if tmp_file.exists():
        tmp_file.unlink()
    return cached_file


def _resolve_youtube(
    storage_path: str,
    cache_dir: Path,
    youtube_adapter: Any,
    extract_video_id: Any,
) -> str:
    """Downloads (or reuses) a cached YouTube video at the best available codec."""
    vid = extract_video_id(storage_path) or "yt"
    cached_file = cache_dir / f"youtube_{vid}.mp4"
    codec = ""

    if cached_file.exists() and cached_file.stat().st_size >= MIN_VALID_FILE_BYTES:
        codec = video_codec(cached_file)
        # An AV1 cached file decodes badly in OpenCV (frame errors) and slowly in
        # software — refetch it once with the non-AV1 format preference. The
        # marker stops an endless refetch loop if AV1 is the only format served.
        marker = cached_file.with_name(cached_file.name + ".av1_refetched")
        if codec == "av01" and not marker.exists():
            print(f"    → Cached file is AV1, re-downloading without AV1: {cached_file}", flush=True)
            marker.touch()
            cached_file.unlink()
            codec = ""

    if not cached_file.exists() or cached_file.stat().st_size < MIN_VALID_FILE_BYTES:
        print(f"    → YouTube download: {storage_path} → {cached_file}", flush=True)
        adapter = youtube_adapter()
        adapter.download_to_path(storage_path, str(cached_file))
    else:
        print(f"    → YouTube cache hit: {cached_file} (codec={codec})", flush=True)

    return str(cached_file)


def resolve_remote_reference(storage_path: str, cache_dir: Path) -> str | None:
    """Resolves a YouTube / HTTP(S) / ``file://`` reference to a local path.

    Returns ``None`` when ``storage_path`` is none of those, leaving the caller
    to resolve it against its own storage.
    """
    from footage_engine.sources.youtube import (
        YouTubeAdapter,
        extract_youtube_video_id,
        is_youtube_url,
    )

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    if is_youtube_url(storage_path):
        return _resolve_youtube(storage_path, cache_dir, YouTubeAdapter, extract_youtube_video_id)

    if storage_path.startswith(("http://", "https://")):
        cached_file = cache_dir / _cached_filename(storage_path)
        if not cached_file.exists() or cached_file.stat().st_size < MIN_VALID_FILE_BYTES:
            _download_to_cache(storage_path, cached_file)
        return str(cached_file)

    if storage_path.startswith("file://"):
        return storage_path[len("file://") :]

    return None


__all__ = [
    "KNOWN_MEDIA_EXTENSIONS",
    "MIN_VALID_FILE_BYTES",
    "resolve_remote_reference",
    "video_codec",
]
