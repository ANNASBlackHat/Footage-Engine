"""Ingest footage from a file of URLs.

Reads any text file, extracts image & video URLs via regex,
deduplicates, and feeds them into the footage engine ingestion pipeline.

Usage:
    python scripts/ingest_from_urls.py <url_file> [--provider PROVIDER] [--dry-run]

Examples:
    python scripts/ingest_from_urls.py urls.txt
    python scripts/ingest_from_urls.py bookmarks.html --provider wikimedia
    python scripts/ingest_from_urls.py links.md --dry-run
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import footage_engine as fe
from footage_engine.config import get_settings
from footage_engine.models.db import init_db
from footage_engine.vector import get_vector_store

# ---------------------------------------------------------------------------
# URL regex patterns — covers common image & video extensions as well as
# YouTube, Vimeo, and other known hosting patterns.
# ---------------------------------------------------------------------------

# File-extension based patterns (most reliable)
_EXT_VIDEO = r"\.(?:mp4|mov|mkv|webm|avi|ogv|m4v|flv|wmv)(?:\?[^\s\"'<>]*)?"
_EXT_IMAGE = r"\.(?:jpg|jpeg|png|webp|gif|bmp|tiff|tif|svg|avif)(?:\?[^\s\"'<>]*)?"

# Known hosting patterns (fallback for URLs without clear extensions)
_YOUTUBE = r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?[^\s\"'<>]*|youtu\.be/[^\s\"'<>]+)"
_VIMEO = r"(?:https?://)?(?:www\.)?vimeo\.com/\d+[^\s\"'<>]*"

# Generic http(s) URL as a catch-all — only used if extension/hosting patterns
# don't match, so we don't pull in navigation links etc.
_GENERIC_URL = r"https?://[^\s\"'<>]+\.(?:mp4|mov|mkv|webm|avi|ogv|m4v|jpg|jpeg|png|webp|gif|bmp|tiff|tif|svg|avif)(?:\?[^\s\"'<>]*)?"

# Combined master pattern
_URL_PATTERN = re.compile(
    "|".join([
        _YOUTUBE,
        _VIMEO,
        _GENERIC_URL,
        _EXT_VIDEO,
        _EXT_IMAGE,
    ]),
    re.IGNORECASE,
)

# More focused extractor: pull full http(s) URLs that contain a media extension
_MEDIA_URL_RE = re.compile(
    r'(?:https?://[^\s"\'<>]+)',
    re.IGNORECASE,
)

# Extension check for filtering
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".ogv", ".m4v", ".flv", ".wmv"})
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".svg", ".avif"})
_KNOWN_HOST_VIDEO = {"youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"}


def _extract_media_urls(text: str) -> list[str]:
    """Extract unique media URLs from freeform text."""
    candidates: list[str] = []

    for match in _MEDIA_URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:!?)\"'")
        low = url.lower().split("?")[0]

        # Check if it ends with a known media extension
        ext = os.path.splitext(low.split("/")[-1])[1] if "/" in low else ""
        host = low.split("/")[2] if low.startswith("http") else ""

        is_video = ext in _VIDEO_EXTS or any(h in host for h in _KNOWN_HOST_VIDEO)
        is_image = ext in _IMAGE_EXTS

        if is_video or is_image:
            candidates.append(url)

    return candidates


def _deduplicate(urls: list[str]) -> list[str]:
    """Remove duplicate URLs (case-insensitive, ignoring trailing slashes)."""
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        key = url.rstrip("/").lower()
        if key not in seen:
            seen.add(key)
            unique.append(url)
    return unique


def classify_url(url: str) -> str:
    """Return 'video', 'image', or 'unknown' based on URL."""
    low = url.lower().split("?")[0]
    ext = os.path.splitext(low.split("/")[-1])[1] if "/" in low else ""
    host = low.split("/")[2] if low.startswith("http") else ""

    if ext in _VIDEO_EXTS or any(h in host for h in _KNOWN_HOST_VIDEO):
        return "video"
    if ext in _IMAGE_EXTS:
        return "image"
    return "unknown"


def load_urls_from_file(filepath: str) -> list[str]:
    """Read a text file and extract all media URLs."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: {filepath} not found.", flush=True)
        sys.exit(1)

    content = path.read_text(encoding="utf-8")
    raw_urls = _extract_media_urls(content)
    urls = _deduplicate(raw_urls)
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Ingest media URLs from a text file into the footage engine."
    )
    parser.add_argument("url_file", help="Path to the file containing URLs")
    parser.add_argument(
        "--provider", default="manual",
        help="Provider label for ingested assets (default: manual)"
    )
    parser.add_argument(
        "--entity", "--entity-name", dest="entity", default=None,
        help="Canonical entity name to associate with all ingested URLs (e.g. 'USS Cyclops', 'Aye-aye')"
    )
    parser.add_argument(
        "--entity-type", default="other",
        help="Entity type if creating a new entity ('ship', 'animal', 'person', 'location', 'event', 'other')"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Extract and display URLs without actually ingesting"
    )
    parser.add_argument(
        "--max", type=int, default=0,
        help="Max number of URLs to ingest (0 = all)"
    )
    args = parser.parse_args()

    # 1. Extract URLs
    urls = load_urls_from_file(args.url_file)
    if not urls:
        print("No media URLs found in the file.", flush=True)
        sys.exit(0)

    if args.max > 0:
        urls = urls[: args.max]

    videos = sum(1 for u in urls if classify_url(u) == "video")
    images = sum(1 for u in urls if classify_url(u) == "image")

    print("=" * 80, flush=True)
    print("📎 URL File Ingestion", flush=True)
    print("=" * 80, flush=True)
    print(f"• Source file : {args.url_file}", flush=True)
    print(f"• Total URLs  : {len(urls)}", flush=True)
    print(f"  ├─ Videos   : {videos}", flush=True)
    print(f"  └─ Images   : {images}", flush=True)
    print(f"• Provider    : {args.provider}", flush=True)
    if args.entity:
        print(f"• Entity      : {args.entity} (type: {args.entity_type})", flush=True)
    print(f"• Dry run     : {args.dry_run}", flush=True)
    print("=" * 80, flush=True)

    # Show first few URLs as preview
    print("\n🔍 URL Preview:", flush=True)
    for i, url in enumerate(urls[:10], 1):
        print(f"   {i}. [{classify_url(url).upper()}] {url[:100]}", flush=True)
    if len(urls) > 10:
        print(f"   ... and {len(urls) - 10} more", flush=True)

    if args.dry_run:
        print("\n🏁 Dry run complete — no assets were ingested.", flush=True)
        return

    # 2. Ingest
    print(f"\n📥 Ingesting {len(urls)} URLs...", flush=True)
    cfg = get_settings()
    init_db(cfg.DATABASE_URL)

    orchestrator = fe.Orchestrator()
    results = orchestrator.ingest_url_list(
        urls=urls,
        provider=args.provider,
        entity_name=args.entity,
    )

    success = sum(1 for r in results if r.status.value != "failed")
    failed = len(results) - success

    print(f"\n{'=' * 80}", flush=True)
    print(f"✅ Ingestion complete!", flush=True)
    print(f"   ├─ Registered : {success}", flush=True)
    print(f"   └─ Failed     : {failed}", flush=True)

    # 3. Optionally process embeddings
    run_processing = os.environ.get("SKIP_PROCESSING") != "1"
    if run_processing and success > 0:
        print("\n🧠 Running Batch Processor (scene detection, embeddings, vector indexing)...", flush=True)
        vec_store = get_vector_store(cfg)
        processor = fe.BatchProcessor(vector_store=vec_store)
        stats = processor.process_all_pending()
        print(f"\n✓ Batch processing results: {stats}", flush=True)

    print(f"\n✨ Done!", flush=True)


if __name__ == "__main__":
    main()
