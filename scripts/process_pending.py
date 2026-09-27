"""Standalone script to process all pending media items (Chunking + Embedding + Vector Indexing).

Supports dual-indexing: Base (X-CLIP) and Qwen multimodal embeddings (if configured via env).

Usage:
    python scripts/process_pending.py
    python scripts/process_pending.py --limit 5 --workers 2
    python scripts/process_pending.py --skip-qwen
    python scripts/process_pending.py --media-ids <id1,id2>
"""

import argparse
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import footage_engine as fe
from footage_engine.config import get_settings
from footage_engine.embeddings import collection_name_for
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import MediaItem
from footage_engine.vector import get_vector_store


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    cfg = get_settings()

    default_workers = int(
        os.environ.get("NUM_WORKERS")
        or os.environ.get("MAX_WORKERS")
        or cfg.PROCESS_NUM_WORKERS
    )
    default_batch_size = int(
        os.environ.get("BATCH_SIZE")
        or os.environ.get("EMBEDDING_BATCH_SIZE")
        or cfg.EMBEDDING_BATCH_SIZE
    )
    default_limit = int(os.environ["LIMIT"]) if "LIMIT" in os.environ else None
    default_skip_qwen = os.environ.get("SKIP_QWEN", "").lower() in ("1", "true", "yes")

    parser = argparse.ArgumentParser(
        description="Batch process pending media items into vector store (X-CLIP + optional Qwen)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=default_limit,
        help=f"Max media items to process (default: {default_limit}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel video worker threads (default: {default_workers}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help=f"Embedding batch size per forward pass (default: {default_batch_size}).",
    )
    parser.add_argument(
        "--media-ids",
        default=os.environ.get("MEDIA_IDS", None),
        help="Comma-separated MediaItem IDs to process.",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        default=default_skip_qwen,
        help="Skip Qwen embeddings even if Qwen Zilliz credentials or collection are set in environment.",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, _ = parse_args()
    cfg = get_settings()
    init_db(cfg.DATABASE_URL)
    vec_store = get_vector_store(cfg)

    workers = args.workers
    batch_size = args.batch_size
    limit = args.limit
    media_ids = [m.strip() for m in args.media_ids.split(",") if m.strip()] if args.media_ids else None

    # Detect Qwen configuration
    has_qwen_configured = bool(
        (cfg.QWEN_ZILLIZ_URI and cfg.QWEN_ZILLIZ_TOKEN)
        or cfg.QWEN_ZILLIZ_COLLECTION_NAME
    )
    run_qwen = has_qwen_configured and not args.skip_qwen

    print("=" * 80, flush=True)
    print("🧠 Footage Engine — Batch Processor for Pending Footage", flush=True)
    print("=" * 80, flush=True)
    print(f"• Database       : {cfg.DATABASE_URL}", flush=True)
    print(f"• Storage        : {cfg.STORAGE_BACKEND} ({cfg.LOCAL_STORAGE_DIR})", flush=True)
    print(f"• Base Store     : {cfg.VECTOR_STORE} (collection: {cfg.ZILLIZ_COLLECTION_NAME})", flush=True)
    print(f"• Base Model     : {cfg.DEFAULT_EMBEDDING_MODEL} (Device: {cfg.EMBEDDING_DEVICE})", flush=True)
    if run_qwen:
        qwen_collection = collection_name_for(cfg, "qwen")
        print(f"• Qwen Model     : {cfg.QWEN_MODEL_NAME} (collection: {qwen_collection})", flush=True)
    elif has_qwen_configured and args.skip_qwen:
        print("• Qwen Model     : Skipped (--skip-qwen flag active)", flush=True)
    else:
        print("• Qwen Model     : Not configured (missing Qwen Zilliz env credentials)", flush=True)

    print(f"• Batch Size     : {batch_size} chunks / forward pass", flush=True)
    print(f"• Worker Threads : {workers} parallel video worker(s)", flush=True)
    if limit:
        print(f"• Limit          : {limit} item(s)", flush=True)
    if media_ids:
        print(f"• Target Media   : {len(media_ids)} item ID(s) specified", flush=True)
    print("=" * 80, flush=True)

    start_time = time.time()
    processor = fe.BatchProcessor(vector_store=vec_store)

    # Phase 1: Base (X-CLIP) Chunking & Embedding
    print("\n⏳ Phase 1: Finding and processing pending media items (X-CLIP)...", flush=True)
    if media_ids:
        print(f"  → Processing {len(media_ids)} specific media item(s)...", flush=True)
        stats = {"total": len(media_ids), "succeeded": 0, "failed": 0, "succeeded_ids": []}
        for mid in media_ids:
            ok = processor.process_item(mid)
            if ok:
                stats["succeeded"] += 1
                stats["succeeded_ids"].append(mid)
            else:
                stats["failed"] += 1
    else:
        stats = processor.process_all_pending(limit=limit, max_workers=workers)

    print(
        f"✓ Phase 1 Completed: {stats['succeeded']}/{stats['total']} succeeded "
        f"({stats['failed']} failed).",
        flush=True,
    )

    # Phase 2: Qwen Embedding (if enabled)
    qwen_stats = {"total": 0, "succeeded": 0, "failed": 0, "chunks": 0}
    if run_qwen:
        print("\n" + "-" * 80, flush=True)
        print(f"🚀 Phase 2: Qwen Multimodal Embeddings ({cfg.QWEN_MODEL_NAME})", flush=True)
        print("-" * 80, flush=True)

        try:
            from footage_engine.embeddings import get_embedder
            from footage_engine.storage import get_storage_backend
            from scripts.backfill_qwen import (
                find_items_needing_backfill,
                process_item as process_qwen_item,
            )

            qwen_collection = collection_name_for(cfg, "qwen")
            t_load = time.time()
            print(f"• Initializing Qwen embedder and vector store...", flush=True)
            qwen_embedder = get_embedder(cfg, backend="qwen")
            storage = get_storage_backend(cfg)
            qwen_vec_store = get_vector_store(cfg, backend="qwen")
            print(f"✓ Qwen Embedder ready in {time.time() - t_load:.1f}s.", flush=True)

            # Candidates:
            # 1) Items that succeeded in Phase 1 of this run
            succeeded_ids = stats.get("succeeded_ids", [])
            # 2) Any DONE items in DB with X-CLIP chunks but missing Qwen chunks
            backfill_candidates = find_items_needing_backfill(
                database_url=cfg.DATABASE_URL,
                xclip_model=cfg.DEFAULT_EMBEDDING_MODEL,
                qwen_model=qwen_embedder.model_name,
                limit=limit,
                media_ids=media_ids,
            )
            # Deduplicate preserving order (newly processed items first)
            target_qwen_ids = list(dict.fromkeys(succeeded_ids + backfill_candidates))
            if limit:
                target_qwen_ids = target_qwen_ids[:limit]

            total_q = len(target_qwen_ids)
            qwen_stats["total"] = total_q

            if total_q == 0:
                print("✓ No items needing Qwen embedding (all up to date).", flush=True)
            else:
                print(
                    f"Found {total_q} item(s) to embed with Qwen into collection '{qwen_collection}'...",
                    flush=True,
                )
                progress_lock = threading.Lock()
                done_q = 0

                def _record_qwen_result(iid: str, n_chunks: int, exc: Optional[Exception]):
                    nonlocal done_q
                    with progress_lock:
                        done_q += 1
                        pct = int((done_q / total_q) * 100) if total_q else 100
                        if exc is not None:
                            qwen_stats["failed"] += 1
                            print(
                                f"  [{done_q}/{total_q}] ({pct}%) [{iid[:8]}] ❌ FAILED: {type(exc).__name__}: {exc}",
                                flush=True,
                            )
                        else:
                            qwen_stats["succeeded"] += 1
                            qwen_stats["chunks"] += n_chunks
                            print(
                                f"  [{done_q}/{total_q}] ({pct}%) [{iid[:8]}] ✓ SUCCESS: indexed {n_chunks} Qwen chunk(s).",
                                flush=True,
                            )

                def _run_single_qwen(iid: str) -> tuple[str, int, Optional[Exception]]:
                    try:
                        with get_db_session(cfg.DATABASE_URL) as session:
                            db_item = session.get(MediaItem, iid)
                            already_has_qwen = any(
                                c.embedding_model == qwen_embedder.model_name
                                for c in (db_item.chunks if db_item else [])
                            )
                        if already_has_qwen:
                            return iid, 0, None

                        n = process_qwen_item(
                            item_id=iid,
                            database_url=cfg.DATABASE_URL,
                            xclip_model=cfg.DEFAULT_EMBEDDING_MODEL,
                            embedder=qwen_embedder,
                            storage=storage,
                            vector_store=qwen_vec_store,
                            collection=qwen_collection,
                            dry_run=False,
                        )
                        return iid, n, None
                    except Exception as e:
                        return iid, 0, e

                if workers <= 1 or total_q <= 1:
                    for iid in target_qwen_ids:
                        _id, n, exc = _run_single_qwen(iid)
                        _record_qwen_result(_id, n, exc)
                else:
                    print(f"🚀 Processing Qwen embeddings with {workers} worker thread(s)...", flush=True)
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {executor.submit(_run_single_qwen, iid): iid for iid in target_qwen_ids}
                        for future in as_completed(futures):
                            _id, n, exc = future.result()
                            _record_qwen_result(_id, n, exc)

        except Exception as qwen_err:
            print(
                f"⚠️ Phase 2 Qwen processing encountered an error: {qwen_err} "
                f"(Base X-CLIP embeddings remain intact and fully indexed).",
                flush=True,
            )

    elapsed = time.time() - start_time
    print("\n" + "=" * 80, flush=True)
    print(f"✨ Batch Processing Completed in {elapsed:.1f}s!", flush=True)
    print(f"  • Base (X-CLIP) items : {stats['succeeded']}/{stats['total']} succeeded ({stats['failed']} failed)")
    if run_qwen:
        print(f"  • Qwen embeddings     : {qwen_stats['succeeded']}/{qwen_stats['total']} items ({qwen_stats['chunks']} chunks indexed, {qwen_stats['failed']} failed)")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
