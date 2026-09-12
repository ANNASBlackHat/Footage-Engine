"""Backfill Qwen embeddings for already-processed media items.

Reads DONE items and their existing X-CLIP chunk time ranges, embeds the same
ranges with Qwen3-VL-Embedding-2B, and writes NEW Chunk rows + vectors into the
Qwen collection. Existing X-CLIP rows and the base collection are only read,
never modified.

Usage:
    python scripts/backfill_qwen.py --dry-run --limit 2
    python scripts/backfill_qwen.py --limit 5
    python scripts/backfill_qwen.py --media-ids <id1,id2>
"""

import argparse
import sys

from sqlalchemy import select

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
    args, _unknown = ap.parse_known_args()  # tolerate kernel argv (e.g. Colab's -f flag)
    return args


def main() -> int:
    args = parse_args()
    settings = get_settings()
    init_db(settings.DATABASE_URL)

    embedder = get_embedder(settings, backend="qwen")
    storage = get_storage_backend(settings)
    vector_store = get_vector_store(settings, backend="qwen")
    collection = collection_name_for(settings, "qwen")
    print(f"Qwen backfill: model={embedder.model_name} dim={embedder.dimension} collection={collection}")

    with get_db_session(settings.DATABASE_URL) as session:
        stmt = (
            select(MediaItem.id)
            .where(MediaItem.status == MediaStatus.DONE)
            .order_by(MediaItem.ingested_at.asc())
        )
        if args.media_ids:
            ids = [i.strip() for i in args.media_ids.split(",") if i.strip()]
            stmt = stmt.where(MediaItem.id.in_(ids))
        if args.limit:
            stmt = stmt.limit(args.limit)
        item_ids = list(session.execute(stmt).scalars().all())

    print(f"Found {len(item_ids)} DONE media item(s).")
    if not item_ids:
        return 0

    total_new = 0
    failed = 0
    for n, iid in enumerate(item_ids, 1):
        try:
            with get_db_session(settings.DATABASE_URL) as session:
                item = session.get(MediaItem, iid)
                if not item:
                    continue
                src_ranges = [
                    (c.start_ts, c.end_ts)
                    for c in item.chunks
                    if c.embedding_model == settings.DEFAULT_EMBEDDING_MODEL
                ]
                if not src_ranges and item.chunks:
                    # Fall back to any existing ranges (e.g. processed under another model name)
                    src_ranges = [(c.start_ts, c.end_ts) for c in item.chunks]
                done_ranges = {
                    (c.start_ts, c.end_ts)
                    for c in item.chunks
                    if c.embedding_model == embedder.model_name
                }
                missing = [r for r in src_ranges if r not in done_ranges]
                print(f"[{item.id[:8]}] ({n}/{len(item_ids)}) {item.media_type.value} "
                      f"ranges={len(src_ranges)} missing_qwen={len(missing)}")

                if not missing:
                    continue
                if args.dry_run:
                    total_new += len(missing)
                    continue

                try:
                    local_path = storage.get_local_path(item.storage_path)
                except Exception:
                    local_path = storage.get_local_path(item.source_url)

                if item.media_type == MediaType.IMAGE:
                    vecs = [embedder.embed_image(local_path) for _ in missing]
                elif hasattr(embedder, "embed_video_batch"):
                    vecs = embedder.embed_video_batch(
                        video_path=local_path,
                        chunk_ranges=missing,
                        batch_size=settings.EMBEDDING_BATCH_SIZE,
                    )
                else:
                    vecs = [
                        embedder.embed_video(video_path=local_path, start_ts=s, end_ts=e)
                        for s, e in missing
                    ]

                records: list[VectorRecord] = []
                for (s_ts, e_ts), vec in zip(missing, vecs):
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
                total_new += len(records)
                print(f"[{item.id[:8]}] ({n}/{len(item_ids)}) indexed {len(records)} Qwen vector(s).")
        except Exception as e:
            failed += 1
            print(f"[{iid[:8]}] ({n}/{len(item_ids)}) FAILED, skipping: {type(e).__name__}: {e}")

    print(f"{'Would create' if args.dry_run else 'Created'} {total_new} Qwen chunk(s), {failed} failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
