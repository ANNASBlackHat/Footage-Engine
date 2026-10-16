#!/usr/bin/env python3
"""Migrate video chunks to Google Drive by re-slicing from cached source media.

Why this exists:
Cloudinary accounts can get locked or throw 401 ACL deny on free-tier rotations,
rendering old chunk URLs inaccessible. Because all chunk time boundaries (start_ts,
end_ts), captions, and embeddings remain safely in PostgreSQL, this script restores
full accessibility by:
  1. Downloading the master video once per media item (via YouTube / provider adapter).
  2. Slicing the exact chunk MP4 using ffmpeg (no scene re-detection needed).
  3. Uploading the slice to Google Drive via GoogleDriveStorageBackend.
  4. Updating chunk.storage_path in the database with the resulting Drive file ID.

Usage:
    # 1. Dry run: inspect the 5 latest Cloudinary chunks that would be migrated
    ./.venv/bin/python scripts/migrate_chunks_to_gdrive.py --dry-run --limit 5

    # 2. Migrate the 5 latest chunks to Google Drive
    ./.venv/bin/python scripts/migrate_chunks_to_gdrive.py --limit 5

    # 3. Migrate all chunks for a specific media item (e.g. latest video)
    ./.venv/bin/python scripts/migrate_chunks_to_gdrive.py --media-item-id 9b016ecf-50da-4e71-9e11-57cd72ebe0d4

    # 4. Migrate the latest 50 chunks
    ./.venv/bin/python scripts/migrate_chunks_to_gdrive.py --limit 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Ensure repository root is on sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sqlalchemy import asc, desc, select, text
from tqdm import tqdm

from footage_engine.chunking.detector import extract_video_clip
from footage_engine.config import get_settings
from footage_engine.models.db import get_engine, get_session_factory
from footage_engine.models.media import Chunk, MediaItem
from footage_engine.sources.youtube import YouTubeAdapter, is_youtube_url
from footage_engine.storage import get_storage_backend
from footage_engine.storage.gdrive import GoogleDriveStorageBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_gdrive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of chunks to migrate (default: all matching).",
    )
    parser.add_argument(
        "--media-item-id",
        type=str,
        default=None,
        help="Filter chunks to a specific media_item_id.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded source videos (default: /tmp/footage_engine_cache).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the workflow without uploading to Drive or updating the database.",
    )
    parser.add_argument(
        "--batch-commit",
        type=int,
        default=5,
        help="Commit database changes every N chunks (default: 5).",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to YouTube cookies file (Netscape format) or raw cookie string for yt-dlp.",
    )
    parser.add_argument(
        "--oldest-first",
        action="store_true",
        help="Process from oldest to newest chunks (useful when running parallel to another worker working newest to oldest).",
    )
    parser.add_argument(
        "--all-broken",
        action="store_true",
        help="Inspect all non-Drive chunks (including ImageKit) and migrate any that return non-200 or connection errors.",
    )
    return parser.parse_args()


def get_master_video_path(
    media_item: MediaItem,
    cache_dir: Path,
    youtube_adapter: YouTubeAdapter,
) -> Path:
    """Ensures the source video for media_item is downloaded and available locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    master_path = cache_dir / f"master_{media_item.id[:8]}.mp4"

    if master_path.exists() and master_path.stat().st_size > 1024 * 1024:
        return master_path

    source_url = media_item.source_url
    logger.info(f"Downloading master video for item {media_item.id[:8]} from: {source_url}")

    if is_youtube_url(source_url):
        youtube_adapter.download_to_path(source_url, str(master_path))
    else:
        # Fallback to general download if not YouTube
        import requests

        with requests.get(source_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(master_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    if not master_path.exists() or master_path.stat().st_size < 1000:
        raise RuntimeError(f"Failed to download source video for item {media_item.id[:8]}")

    logger.info(f"Master video cached at: {master_path} ({master_path.stat().st_size / (1024*1024):.1f} MB)")
    return master_path


def is_url_accessible(path: str) -> tuple[bool, str]:
    """Probes whether a storage path is actually accessible and serving video media."""
    import requests
    if not path:
        return False, "path is empty"
    if path.startswith(("http://", "https://")):
        url = path
    elif path.startswith("/footage_engine/raw/"):
        # ImageKit relative path
        url = f"https://ik.imagekit.io/annassocial{path}"
    else:
        # Default to Cloudinary endpoint if relative
        url = f"https://res.cloudinary.com/qdfnsveu/video/upload/{path.lstrip('/')}"
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True)
        if resp.status_code == 200:
            ctype = resp.headers.get("content-type", "").lower()
            if any(ctype.startswith(prefix) for prefix in ("video/", "application/octet-stream")):
                return True, f"HTTP 200 ({ctype})"
            return False, f"HTTP 200 but not media ({ctype})"
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:60]}"


def main() -> int:
    args = parse_args()
    cfg = get_settings()

    cache_dir = Path(args.cache_dir or os.path.join(tempfile.gettempdir(), "footage_engine_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Google Drive storage backend
    if not args.dry_run:
        storage = get_storage_backend()
        if not isinstance(storage, GoogleDriveStorageBackend):
            print(
                f"Error: Current configured backend is '{type(storage).__name__}', not GoogleDriveStorageBackend.\n"
                f"Set STORAGE_BACKEND=gdrive in .env before running this script.",
                file=sys.stderr,
            )
            return 1
    else:
        storage = None

    SessionLocal = get_session_factory(cfg.DATABASE_URL)
    yt_adapter = YouTubeAdapter(cookies=args.cookies)
    if args.cookies:
        logger.info(f"Using YouTube cookies from: {args.cookies}")

    print("=" * 72)
    print("MIGRATE CHUNKS TO GOOGLE DRIVE")
    target_desc = "All broken non-Drive chunks (including ImageKit)" if args.all_broken else "Cloudinary chunks"
    order_desc = "oldest -> newest" if args.oldest_first else "newest -> oldest"
    print(f"Target: {target_desc} (ordered: {order_desc})")
    if args.media_item_id:
        print(f"Filter media_item_id: {args.media_item_id}")
    if args.limit:
        print(f"Limit: {args.limit} chunk(s)")
    if args.dry_run:
        print("MODE: DRY RUN (no uploads or DB updates)")
    print("=" * 72)

    with SessionLocal() as session:
        if args.all_broken:
            # Query all non-Drive chunks that have an explicit path (including ImageKit)
            query = (
                select(Chunk)
                .join(MediaItem, Chunk.media_item_id == MediaItem.id)
                .where(
                    Chunk.storage_path.isnot(None),
                    Chunk.storage_path != "",
                    ~Chunk.storage_path.like("%drive.google.com%"),
                    ~Chunk.storage_path.like("%googleusercontent.com%"),
                    (text("LENGTH(chunks.storage_path) > 44 OR chunks.storage_path LIKE '%/%' OR chunks.storage_path LIKE '%.%'")),
                )
            )
        else:
            # Query chunks with broken Cloudinary paths, excluding any that are already in Drive
            query = (
                select(Chunk)
                .join(MediaItem, Chunk.media_item_id == MediaItem.id)
                .where(
                    (
                        (Chunk.storage_path.like("%cloudinary%"))
                        | (Chunk.storage_path.like("%qdfnsveu%"))
                        | (Chunk.storage_path.like("footage_engine/raw/chunks/%"))
                    )
                    & (~Chunk.storage_path.like("%drive.google.com%"))
                    & (~Chunk.storage_path.like("%googleusercontent.com%"))
                )
            )
        if args.media_item_id:
            query = query.where(Chunk.media_item_id == args.media_item_id)

        # Order chunks (default: newest first; or oldest first)
        if args.oldest_first:
            query = query.order_by(asc(Chunk.created_at), asc(Chunk.id))
        else:
            query = query.order_by(desc(Chunk.created_at), desc(Chunk.id))

        if args.limit:
            query = query.limit(args.limit)

        chunks_to_migrate: list[Chunk] = list(session.execute(query).scalars().all())

        if not chunks_to_migrate:
            print("No matching candidate chunks found to migrate.")
            return 0

        print(f"Found {len(chunks_to_migrate)} candidate chunk(s) to inspect.\n")

        migrated_count = 0
        skipped_count = 0
        failed_count = 0

        active_item_id = None
        active_master_path: Optional[Path] = None

        # Group work or process sequentially
        with tempfile.TemporaryDirectory() as tmp_slice_dir:
            for i, chunk in enumerate(chunks_to_migrate, start=1):
                item = chunk.media_item
                start_ts = chunk.start_ts
                end_ts = chunk.end_ts or (chunk.start_ts + 5.0)

                # Clean up previous master video if we moved to a new item
                if active_item_id and active_item_id != item.id and active_master_path and active_master_path.exists():
                    try:
                        active_master_path.unlink()
                        logger.info(f"Cleaned up previous master video: {active_master_path.name}")
                    except Exception as err:
                        logger.debug(f"Could not remove old master {active_master_path}: {err}")
                    active_master_path = None

                active_item_id = item.id

                # Step 0: Double check current path accessibility
                accessible, reason = is_url_accessible(chunk.storage_path or "")
                if accessible:
                    print(
                        f"[{i}/{len(chunks_to_migrate)}] Skipping chunk {chunk.id[:8]} "
                        f"- already accessible ({reason})"
                    )
                    skipped_count += 1
                    continue

                print(
                    f"[{i}/{len(chunks_to_migrate)}] Chunk {chunk.id[:8]} "
                    f"(item {item.id[:8]} @ {start_ts:.2f}s - {end_ts:.2f}s) | "
                    f"Inaccessible ({reason}) | Current: {chunk.storage_path[:50]}..."
                )

                if args.dry_run:
                    print(f"    [DRY RUN] Would download master for item {item.id[:8]}")
                    print(f"    [DRY RUN] Would cut clip {start_ts:.2f}s - {end_ts:.2f}s")
                    print(f"    [DRY RUN] Would upload to Google Drive and update DB")
                    migrated_count += 1
                    continue

                try:
                    # 1. Download master video once per item
                    master_path = get_master_video_path(item, cache_dir, yt_adapter)
                    active_master_path = master_path

                    # 2. Slice chunk MP4
                    slice_filename = f"{item.id[:8]}_{chunk.id[:8]}.mp4"
                    slice_path = os.path.join(tmp_slice_dir, slice_filename)
                    extract_video_clip(str(master_path), start_ts, end_ts, slice_path)

                    if not os.path.exists(slice_path) or os.path.getsize(slice_path) < 100:
                        raise RuntimeError(f"FFmpeg slice generated empty file: {slice_path}")

                    slice_size_kb = os.path.getsize(slice_path) / 1024
                    print(f"    → Sliced {slice_filename} ({slice_size_kb:.1f} KB)")

                    # 3. Upload to Google Drive
                    remote_name = f"chunks/{slice_filename}"
                    with open(slice_path, "rb") as f:
                        file_id = storage.save_file(
                            file_data=f,
                            filename=remote_name,
                            content_type="video/mp4",
                            media_type="video",
                        )

                    # 4. Update database
                    chunk.storage_path = file_id
                    migrated_count += 1
                    drive_url = storage.get_url(file_id)
                    print(f"    ✓ Uploaded to Drive: fileId={file_id}")
                    print(f"    ✓ URL: {drive_url}")

                    # Clean up temporary slice
                    if os.path.exists(slice_path):
                        os.remove(slice_path)

                    # Commit periodically
                    if migrated_count % args.batch_commit == 0:
                        session.commit()
                        logger.info(f"Committed batch ({migrated_count} chunks saved).")

                except Exception as e:
                    logger.error(f"Failed migrating chunk {chunk.id}: {e}", exc_info=True)
                    failed_count += 1

            if not args.dry_run and migrated_count > 0:
                session.commit()

    print("\n" + "=" * 72)
    print("MIGRATION SUMMARY")
    print("=" * 72)
    print(f"  Successfully processed: {migrated_count}")
    print(f"  Failed:                  {failed_count}")
    print(f"  Master cache location:   {cache_dir}")
    print("=" * 72)
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
