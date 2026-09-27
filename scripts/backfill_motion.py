#!/usr/bin/env python3
"""Backfill motion scores (motion_mean, motion_std) for video chunks.

Computes normalized frame-difference variance to score video dynamism and
detect static loops. Scores are persisted to the database.

Usage:
    # Dry run on 5 chunks
    python scripts/backfill_motion.py --dry-run --limit 5

    # Process 50 chunks locally with 4 workers
    python scripts/backfill_motion.py --limit 50 --workers 4

    # Process only Pexels footage
    python scripts/backfill_motion.py --provider pexels --workers 4

    # Run continuously until all chunks are scored
    python scripts/backfill_motion.py --workers 4
"""

import argparse
import os
import sys
import time

# Suppress low-level FFmpeg C-library warnings on early stream termination
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from sqlalchemy import desc, select

from footage_engine.models.db import get_engine, get_session_factory, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaType
from footage_engine.processing.motion import compute_motion_score
from footage_engine.storage import get_storage_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill motion scores for video chunks.")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process (default: all unscored).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads (default: 4).")
    parser.add_argument("--batch-size", type=int, default=20, help="DB commit batch size (default: 20).")
    parser.add_argument("--provider", type=str, default=None, help="Filter by provider (e.g. pexels, pixabay).")
    parser.add_argument("--dry-run", action="store_true", help="Calculate scores without updating DB.")
    return parser.parse_args()


def resolve_video_source(storage, raw_path: Optional[str]) -> Optional[str]:
    """Resolves raw storage path to a local file or playable HTTP stream URL."""
    if not raw_path:
        return None
    raw = str(raw_path).strip()
    if raw.startswith(("http://", "https://")):
        return raw
    try:
        local_path = storage.get_local_path(raw)
        if Path(local_path).exists():
            return local_path
    except Exception:
        pass
    # Check relative to base_dir
    base_dir = getattr(storage, "base_dir", None)
    if base_dir:
        cand = Path(base_dir) / raw.lstrip("/\\")
        if cand.exists():
            return str(cand)
    return raw


def process_single_chunk(task_data: dict) -> dict:
    """Worker function to compute motion score for a single chunk."""
    chunk_id = task_data["chunk_id"]
    video_source = task_data["video_source"]
    start_ts = task_data["start_ts"]
    end_ts = task_data["end_ts"]

    score = compute_motion_score(
        video_source=video_source,
        start_ts=start_ts,
        end_ts=end_ts,
        sample_fps=4,
        resize_width=320,
        max_duration_sec=30.0,
    )
    return {
        "chunk_id": chunk_id,
        "score": score,
    }


def main():
    args = parse_args()

    # Ensure DB schema is up to date (motion columns exist)
    init_db()
    session_factory = get_session_factory()
    storage = get_storage_backend()

    # 1. Query unscored video chunks
    print("🔍 Querying unscored video chunks from database...")
    query_session = session_factory()
    try:
        stmt = (
            select(
                Chunk.id,
                Chunk.start_ts,
                Chunk.end_ts,
                Chunk.storage_path.label("chunk_storage"),
                MediaItem.storage_path.label("media_storage"),
                MediaItem.source_url,
                MediaItem.provider,
            )
            .join(MediaItem, Chunk.media_item_id == MediaItem.id)
            .where(
                Chunk.motion_mean.is_(None),
                Chunk.media_type == MediaType.VIDEO,
            )
            .order_by(desc(Chunk.usage_count), desc(Chunk.created_at))
        )

        if args.provider:
            stmt = stmt.where(MediaItem.provider == args.provider.lower().strip())

        if args.limit:
            stmt = stmt.limit(args.limit)

        rows = query_session.execute(stmt).all()
    finally:
        query_session.close()

    total_found = len(rows)
    if total_found == 0:
        print("✨ No unscored chunks found! Everything is already scored.")
        return

    print(f"📋 Found {total_found} unscored chunk(s) to process.")
    if args.dry_run:
        print("⚠️  DRY-RUN mode enabled: scores will be computed and displayed, but NOT written to DB.")

    # 2. Prepare task payloads
    tasks = []
    for row in rows:
        c_id, s_ts, e_ts, c_store, m_store, s_url, prov = row
        raw_target = c_store or m_store or s_url
        video_src = resolve_video_source(storage, raw_target)
        if video_src:
            tasks.append({
                "chunk_id": c_id,
                "video_source": video_src,
                "start_ts": s_ts or 0.0,
                "end_ts": e_ts,
                "provider": prov,
            })

    # 3. Process with ThreadPoolExecutor
    succeeded = 0
    failed = 0
    all_means = []
    pending_updates = []

    progress = tqdm(total=len(tasks), desc="Scoring motion", unit="chunk")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_chunk, t): t for t in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            c_id = res["chunk_id"]
            score = res["score"]

            if score is not None:
                succeeded += 1
                all_means.append(score["motion_mean"])
                pending_updates.append((c_id, score["motion_mean"], score["motion_std"]))
            else:
                failed += 1

            # Batch write to DB if not dry run
            if not args.dry_run and len(pending_updates) >= args.batch_size:
                _save_batch(session_factory, pending_updates)
                pending_updates.clear()

            avg_mean = round(sum(all_means) / len(all_means), 2) if all_means else 0.0
            progress.set_postfix({
                "ok": succeeded,
                "failed": failed,
                "avg_mean": avg_mean,
            })
            progress.update(1)

    # Save remaining batch
    if not args.dry_run and pending_updates:
        _save_batch(session_factory, pending_updates)
        pending_updates.clear()

    progress.close()

    print("\n" + "=" * 50)
    print("🎉 Backfill Completed Summary:")
    print(f"   Total Processed: {len(tasks)}")
    print(f"   Succeeded:       {succeeded}")
    print(f"   Failed / Blank:  {failed}")
    if all_means:
        print(f"   Avg Motion Mean: {round(sum(all_means) / len(all_means), 2)}")
    print("=" * 50)


def _save_batch(session_factory, updates: list[tuple[str, float, float]]) -> None:
    """Persists a batch of motion scores to the database."""
    session = session_factory()
    try:
        for chunk_id, m_mean, m_std in updates:
            chunk = session.get(Chunk, chunk_id)
            if chunk:
                chunk.motion_mean = m_mean
                chunk.motion_std = m_std
        session.commit()
    except Exception as exc:
        session.rollback()
        print(f"⚠️  Error saving batch to DB: {exc}", file=sys.stderr)
    finally:
        session.close()


if __name__ == "__main__":
    main()
