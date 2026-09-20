"""Backfill Qwen embeddings for already-processed media items.

Single SQL query finds items that have X-CLIP chunks but NO Qwen chunks, then
embeds the missing ranges in parallel. Existing X-CLIP rows and the base
collection are only read, never modified.

Usage:
    python scripts/backfill_qwen.py --dry-run --limit 2
    python scripts/backfill_qwen.py --limit 5
    python scripts/backfill_qwen.py --media-ids <id1,id2>
    python scripts/backfill_qwen.py --workers 4 --limit 50
"""

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from sqlalchemy import text

from footage_engine.config import get_settings
from footage_engine.embeddings import collection_name_for, get_embedder
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaStatus, MediaType
from footage_engine.storage import get_storage_backend
from footage_engine.vector import get_vector_store
from footage_engine.vector.base import VectorRecord


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill Qwen embeddings for DONE media items.")
    ap.add_argument("--limit", type=int, default=None, help="Max media items to process.")
    ap.add_argument("--media-ids", default=None, help="Comma-separated MediaItem IDs (subset).")
    ap.add_argument("--dry-run", action="store_true", help="Report what would be done without writing.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel video processing workers (default: 1).")
    args, _unknown = ap.parse_known_args()  # tolerate kernel argv (e.g. Colab's -f flag)
    return args


def find_items_needing_backfill(
    database_url: str,
    xclip_model: str,
    qwen_model: str,
    limit: Optional[int] = None,
    media_ids: Optional[list[str]] = None,
) -> list[str]:
    """Single SQL query: find DONE items with X-CLIP chunks but no Qwen chunks."""
    query = """
        SELECT mi.id
        FROM media_items mi
        WHERE mi.status = 'done'
          AND EXISTS (
            SELECT 1 FROM chunks c
            WHERE c.media_item_id = mi.id
              AND c.embedding_model = :xclip_model
          )
          AND NOT EXISTS (
            SELECT 1 FROM chunks c
            WHERE c.media_item_id = mi.id
              AND c.embedding_model = :qwen_model
          )
    """
    params: dict = {"xclip_model": xclip_model, "qwen_model": qwen_model}

    if media_ids:
        placeholders = ", ".join(f":mid_{i}" for i in range(len(media_ids)))
        query += f" AND mi.id IN ({placeholders})"
        for i, mid in enumerate(media_ids):
            params[f"mid_{i}"] = mid

    query += " ORDER BY mi.ingested_at ASC"

    if limit:
        query += " LIMIT :lim"
        params["lim"] = limit

    with get_db_session(database_url) as session:
        rows = session.execute(text(query), params).scalars().all()
    return list(rows)


def get_xclip_ranges(database_url: str, item_id: str, xclip_model: str) -> list[tuple[float, Optional[float]]]:
    """Load X-CLIP chunk time ranges for a single item."""
    with get_db_session(database_url) as session:
        item = session.get(MediaItem, item_id)
        if not item:
            return []
        return [
            (c.start_ts, c.end_ts)
            for c in item.chunks
            if c.embedding_model == xclip_model
        ]


def process_item(
    item_id: str,
    database_url: str,
    xclip_model: str,
    embedder,
    storage,
    vector_store,
    collection: str,
    dry_run: bool = False,
    progress_counter: Optional[dict] = None,
    progress_lock: Optional[threading.Lock] = None,
) -> int:
    """Embed a single item's missing Qwen chunks. Returns number of new chunks created."""
    ranges = get_xclip_ranges(database_url, item_id, xclip_model)
    if not ranges:
        return 0

    if dry_run:
        return len(ranges)

    with get_db_session(database_url) as session:
        item = session.get(MediaItem, item_id)
        if not item:
            return 0

        try:
            local_path = storage.get_local_path(item.storage_path)
        except Exception:
            local_path = storage.get_local_path(item.source_url)

        if item.media_type == MediaType.IMAGE:
            vecs = [embedder.embed_image(local_path) for _ in ranges]
        elif hasattr(embedder, "embed_video_batch"):
            vecs = embedder.embed_video_batch(
                video_path=local_path,
                chunk_ranges=ranges,
                batch_size=8,
            )
        else:
            vecs = [
                embedder.embed_video(video_path=local_path, start_ts=s, end_ts=e)
                for s, e in ranges
            ]

        records: list[VectorRecord] = []
        for (s_ts, e_ts), vec in zip(ranges, vecs):
            chunk = Chunk(
                media_item_id=item.id,
                start_ts=s_ts,
                end_ts=e_ts,
                media_type=item.media_type,
                embedding_model=embedder.model_name,
                embedding_version=embedder.version,
            )
            session.add(chunk)
            session.flush()  # assign chunk.id
            chunk.vector_id = chunk.id
            records.append(
                VectorRecord(
                    id=chunk.id,
                    vector=vec,
                    chunk_id=chunk.id,
                    media_item_id=item.id,
                    provider=item.provider,
                    media_type=chunk.media_type.value,
                    duration_sec=e_ts - s_ts if e_ts else None,
                    embedding_model=chunk.embedding_model,
                    embedding_version=chunk.embedding_version,
                )
            )

        vector_store.upsert(collection, records)
        session.flush()

    if progress_counter is not None and progress_lock is not None:
        with progress_lock:
            progress_counter["done"] += 1
            done = progress_counter["done"]
            total = progress_counter["total"]
            pct = int((done / total) * 100) if total else 100
            print(f"  [{done}/{total}] ({pct}%) {item_id[:8]} indexed {len(records)} Qwen vector(s).", flush=True)

    return len(records)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    init_db(settings.DATABASE_URL)

    embedder = get_embedder(settings, backend="qwen")
    storage = get_storage_backend(settings)
    vector_store = get_vector_store(settings, backend="qwen")
    collection = collection_name_for(settings, "qwen")

    xclip_model = settings.DEFAULT_EMBEDDING_MODEL  # 'microsoft/xclip-base-patch32'
    qwen_model = embedder.model_name

    print(f"Qwen backfill: model={qwen_model} dim={embedder.dimension} collection={collection}")
    print(f"  X-CLIP model: {xclip_model}")
    print(f"  Workers: {args.workers}")

    # 1. Single SQL query — find items needing backfill (no per-item scan)
    media_ids = [i.strip() for i in args.media_ids.split(",") if i.strip()] if args.media_ids else None
    item_ids = find_items_needing_backfill(
        database_url=settings.DATABASE_URL,
        xclip_model=xclip_model,
        qwen_model=qwen_model,
        limit=args.limit,
        media_ids=media_ids,
    )

    print(f"Found {len(item_ids)} item(s) with X-CLIP chunks but no Qwen chunks.")
    if not item_ids:
        return 0

    if args.dry_run:
        # Quick dry-run: just count ranges per item
        total_ranges = 0
        for iid in item_ids:
            ranges = get_xclip_ranges(settings.DATABASE_URL, iid, xclip_model)
            total_ranges += len(ranges)
            print(f"  [{iid[:8]}] {len(ranges)} range(s) to backfill")
        print(f"Would create {total_ranges} Qwen chunk(s).")
        return 0

    # 2. Process items — parallel or sequential
    progress = {"done": 0, "total": len(item_ids)}
    progress_lock = threading.Lock()
    total_new = 0
    failed = 0

    def _worker(iid: str) -> tuple[str, int, Optional[Exception]]:
        try:
            n = process_item(
                item_id=iid,
                database_url=settings.DATABASE_URL,
                xclip_model=xclip_model,
                embedder=embedder,
                storage=storage,
                vector_store=vector_store,
                collection=collection,
                dry_run=False,
                progress_counter=progress,
                progress_lock=progress_lock,
            )
            return iid, n, None
        except Exception as e:
            return iid, 0, e

    if args.workers <= 1 or len(item_ids) <= 1:
        for iid in item_ids:
            try:
                n = process_item(
                    item_id=iid,
                    database_url=settings.DATABASE_URL,
                    xclip_model=xclip_model,
                    embedder=embedder,
                    storage=storage,
                    vector_store=vector_store,
                    collection=collection,
                    dry_run=False,
                    progress_counter=progress,
                    progress_lock=progress_lock,
                )
                total_new += n
            except Exception as e:
                failed += 1
                print(f"  [{iid[:8]}] FAILED: {type(e).__name__}: {e}", flush=True)
    else:
        print(f"Processing {len(item_ids)} items with {args.workers} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, iid): iid for iid in item_ids}
            for future in as_completed(futures):
                iid, n, exc = future.result()
                if exc:
                    failed += 1
                    print(f"  [{iid[:8]}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                else:
                    total_new += n

    print(f"Created {total_new} Qwen chunk(s), {failed} failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
