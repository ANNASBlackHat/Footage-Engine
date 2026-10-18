#!/usr/bin/env python3
"""Caption uncaptioned video chunks using a Qwen-VL model.

Footage-Engine version of the pipeline's ``caption_chunks.py``. Instead of raw
psycopg2 it uses the SQLAlchemy session so it works against SQLite or
PostgreSQL (whatever ``DATABASE_URL`` points at), and instead of yt-dlp it
resolves the source video through the storage backend (local, ImageKit,
Cloudinary, or Google Drive) — no separate download step.

Writes generated captions/tags back to ``Chunk.caption`` / ``Chunk.tags`` so
the retrieval layer (``retrieval/api.py``, ``retrieval/llm.py``) can surface
them.

Usage:
    # Caption the media item with the most uncaptioned chunks
    uv run python scripts/caption_chunks.py

    # Target a specific item, up to the first 10 minutes
    uv run python scripts/caption_chunks.py \
        --media-item-id e4ea8da0-ed79-4286-b0da-f9b4a47d7655 \
        --max-start-ts 600.0

    # Dry run (list what would be captioned without loading the model)
    uv run python scripts/caption_chunks.py --dry-run --limit 5
"""

import argparse
import sys
import time
from pathlib import Path

# Repo root so `import footage_engine` works regardless of cwd
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sqlalchemy import func, select

from footage_engine.captions import QwenCaptioner
from footage_engine.config import get_settings
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaStatus
from footage_engine.storage import get_storage_backend


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Caption uncaptioned video chunks with Qwen-VL.")
    ap.add_argument(
        "--media-item-id",
        default="auto",
        help="Target media_item ID, or 'auto' to pick the item with the most uncaptioned chunks (default: auto)",
    )
    ap.add_argument(
        "--max-start-ts",
        type=float,
        default=None,
        help="Only caption chunks whose start_ts is at or below this many seconds.",
    )
    ap.add_argument(
        "--model-id",
        default=None,
        help="Override the Qwen caption model (default: QWEN_CAPTION_MODEL from settings).",
    )
    ap.add_argument(
        "--frames-per-chunk",
        type=int,
        default=None,
        help="Frames sampled per chunk (default: QWEN_CAPTION_FRAMES from settings).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Recaption chunks that already have a caption.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List the chunk intervals that would be captioned; do not load the model or write.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of chunk intervals processed in this run.",
    )
    args, _unknown = ap.parse_known_args()  # tolerate kernel argv (e.g. Colab's -f flag)
    return args


def find_next_uncaptioned(
    database_url: str,
    session_factory=None,
) -> str | None:
    """Returns the media_item.id with the most uncaptioned chunks, or None."""
    with get_db_session(database_url) as session:
        uncaptioned = (
            select(Chunk.media_item_id)
            .where((Chunk.caption.is_(None)) | (Chunk.caption == ""))
            .group_by(Chunk.media_item_id)
        )
        stmt = (
            select(
                Chunk.media_item_id,
                func.count(Chunk.id).label("n"),
            )
            .where((Chunk.caption.is_(None)) | (Chunk.caption == ""))
            .group_by(Chunk.media_item_id)
            .order_by(func.count(Chunk.id).desc())
            .limit(1)
        )
        row = session.execute(stmt).first()
        return row[0] if row else None


def fetch_chunk_intervals(
    session,
    media_item_id: str,
    max_start_ts: float | None = None,
    overwrite: bool = False,
    limit: int | None = None,
) -> list[Chunk]:
    """Returns distinct chunks to caption for a media item, ordered by start_ts."""
    conditions = [Chunk.media_item_id == media_item_id, Chunk.end_ts.isnot(None)]
    if max_start_ts is not None:
        conditions.append(Chunk.start_ts <= max_start_ts)
    if not overwrite:
        conditions.append((Chunk.caption.is_(None)) | (Chunk.caption == ""))

    stmt = (
        select(Chunk)
        .where(*conditions)
        .order_by(Chunk.start_ts.asc())
    )
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = session.execute(stmt).scalars().all()

    # De-duplicate overlapping intervals (the same shot can appear in several
    # chunk rows under different embedding models); caption the first row of
    # each distinct (start_ts, end_ts) pair and leave the rest for a follow-up.
    seen: set[tuple[float, float]] = set()
    unique: list[Chunk] = []
    for chunk in rows:
        key = (chunk.start_ts, chunk.end_ts)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def resolve_video_path(
    storage,
    session,
    media_item_id: str,
) -> str:
    """Resolves the local path for the media item's source video.

    Tries ``storage_path`` first (the uploaded master or the raw source), then
    falls back to ``source_url``.
    """
    item = session.get(MediaItem, media_item_id)
    if not item:
        raise ValueError(f"MediaItem with id '{media_item_id}' not found in database.")
    try:
        return storage.get_local_path(item.storage_path)
    except Exception:
        return storage.get_local_path(item.source_url)


def process_item(
    database_url: str,
    media_item_id: str,
    model_id: str,
    num_frames: int,
    max_start_ts: float | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    progress_callback=None,
) -> int:
    """Captions the missing chunks for one media item.

    Returns the number of chunk intervals captioned in this run.
    """
    cfg = get_settings()
    storage = get_storage_backend(cfg)

    # Build the captioner only when we actually have work to do.
    if not dry_run:
        captioner = QwenCaptioner(
            model_name=model_id,
            device=cfg.QWEN_CAPTION_DEVICE,
            num_frames=num_frames,
        )
    else:
        captioner = None

    with get_db_session(database_url) as session:
        intervals = fetch_chunk_intervals(
            session,
            media_item_id,
            max_start_ts=max_start_ts,
            overwrite=overwrite,
            limit=limit,
        )
        if not intervals:
            print(f"  ✓ No uncaptioned chunks for item {media_item_id[:8]}.")
            return 0

        if dry_run:
            print(f"  [dry-run] Would caption {len(intervals)} chunk interval(s) for {media_item_id[:8]}:")
            for c in intervals[:10]:
                print(f"    [{c.start_ts:.1f}s - {c.end_ts:.1f}s] ({(c.end_ts or 0) - c.start_ts:.1f}s)")
            if len(intervals) > 10:
                print(f"    ... and {len(intervals) - 10} more.")
            return len(intervals)

        video_path = resolve_video_path(storage, session, media_item_id)
        print(f"  → Video: {video_path}")

        done = 0
        t0 = time.time()
        for idx, chunk in enumerate(intervals, 1):
            dur = max(0.1, (chunk.end_ts or 0) - chunk.start_ts)
            caption, tags = captioner.caption_chunk(
                video_path=video_path,
                start_ts=chunk.start_ts,
                end_ts=chunk.end_ts,
                num_frames=num_frames,
            )
            chunk.caption = caption
            chunk.tags = tags
            session.flush()
            done += 1
            if progress_callback:
                progress_callback(done, len(intervals))
            print(
                f"    [{idx:03d}/{len(intervals)}] {chunk.start_ts:8.1f}s → {chunk.end_ts:8.1f}s "
                f"({dur:5.1f}s) | 💬 {caption[:80]}"
                + (f" | 🏷️ {', '.join(tags[:5])}" if tags else ""),
                flush=True,
            )
        print(f"  ✓ Captioned {done}/{len(intervals)} chunk(s) in {time.time() - t0:.1f}s.")
        return done


def main() -> int:
    args = parse_args()
    cfg = get_settings()
    init_db(cfg.DATABASE_URL)

    media_item_id = args.media_item_id
    if media_item_id and media_item_id.lower() == "auto":
        media_item_id = find_next_uncaptioned(cfg.DATABASE_URL)
        if not media_item_id:
            print("🎉 No uncaptioned media items found. Everything is captioned.")
            return 0
        print(f"🎯 Auto-selected next media item with uncaptioned chunks: {media_item_id}")

    model_id = args.model_id or cfg.QWEN_CAPTION_MODEL
    num_frames = args.frames_per_chunk or cfg.QWEN_CAPTION_FRAMES

    print("=" * 80, flush=True)
    print("📝 Footage Engine — Qwen-VL Chunk Captioning", flush=True)
    print("=" * 80, flush=True)
    print(f"• Media Item  : {media_item_id}", flush=True)
    print(f"• Model       : {model_id}", flush=True)
    print(f"• Frames/Chnk : {num_frames}", flush=True)
    if args.max_start_ts is not None:
        print(f"• Max Start TS: {args.max_start_ts}s ({args.max_start_ts / 60:.1f} min)", flush=True)
    if args.overwrite:
        print("• Overwrite   : yes", flush=True)
    if args.dry_run:
        print("• Mode        : dry-run (no model load, no writes)", flush=True)
    print("=" * 80, flush=True)

    n = process_item(
        database_url=cfg.DATABASE_URL,
        media_item_id=media_item_id,
        model_id=model_id,
        num_frames=num_frames,
        max_start_ts=args.max_start_ts,
        overwrite=args.overwrite,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(f"\n✅ Done. {n} chunk interval(s) {'would be' if args.dry_run else ''}captioned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
