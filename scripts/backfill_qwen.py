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

from sqlalchemy import exists, select

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
    """ORM query: find DONE items with X-CLIP chunks but no Qwen chunks."""
    has_xclip = exists().where(
        Chunk.media_item_id == MediaItem.id,
        Chunk.embedding_model == xclip_model,
    )
    has_qwen = exists().where(
        Chunk.media_item_id == MediaItem.id,
        Chunk.embedding_model == qwen_model,
    )

    stmt = (
        select(MediaItem.id)
        .where(
            MediaItem.status == MediaStatus.DONE,
            has_xclip,
            ~has_qwen,
        )
        .order_by(MediaItem.ingested_at.asc())
    )
    if media_ids:
        stmt = stmt.where(MediaItem.id.in_(media_ids))
    if limit:
        stmt = stmt.limit(limit)

    with get_db_session(database_url) as session:
        rows = session.execute(stmt).scalars().all()
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
            print(f"    → [{item.id[:8]}] source=storage_path ({item.storage_path}) → {local_path}", flush=True)
        except Exception as e:
            print(
                f"    → [{item.id[:8]}] storage_path failed ({item.storage_path}): "
                f"{type(e).__name__}: {e} — falling back to source_url ({item.source_url})",
                flush=True,
            )
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

    # 2. Process items — parallel or sequential with continuous progress and failure skipping
    total = len(item_ids)
    progress_lock = threading.Lock()
    done_count = 0
    successes: list[dict] = []
    failures: list[dict] = []

    def _record_result(iid: str, n: int, exc: Optional[Exception]):
        nonlocal done_count
        with progress_lock:
            done_count += 1
            pct = int((done_count / total) * 100) if total else 100
            if exc is not None:
                err_str = f"{type(exc).__name__}: {exc}"
                failures.append({"id": iid, "error": err_str})
                print(f"  [{done_count}/{total}] ({pct}%) [{iid[:8]}] SKIPPED (FAILED): {err_str}", flush=True)
            else:
                successes.append({"id": iid, "chunks": n})
                print(f"  [{done_count}/{total}] ({pct}%) [{iid[:8]}] SUCCESS: indexed {n} Qwen chunk(s).", flush=True)

    def _run_item(iid: str) -> tuple[str, int, Optional[Exception]]:
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
            )
            return iid, n, None
        except Exception as e:
            return iid, 0, e

    if args.workers <= 1 or len(item_ids) <= 1:
        for iid in item_ids:
            _iid, n, exc = _run_item(iid)
            _record_result(_iid, n, exc)
    else:
        print(f"Processing {total} items with {args.workers} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_item, iid): iid for iid in item_ids}
            for future in as_completed(futures):
                iid, n, exc = future.result()
                _record_result(iid, n, exc)

    total_chunks = sum(s["chunks"] for s in successes)
    print("\n" + "=" * 65)
    print("                    BACKFILL SUMMARY REPORT")
    print("=" * 65)
    print(f"  Total items evaluated:   {total}")
    print(f"  Successfully processed:  {len(successes)} item(s) ({total_chunks} chunk(s) indexed)")
    print(f"  Failed / Skipped:        {len(failures)} item(s)")

    if failures:
        print("\n  Failures breakdown:")
        for f in failures:
            print(f"    - [{f['id'][:8]}] {f['error']}")
        print("\n  Note: Failed items were skipped without halting the process.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    sys.exit(main())
