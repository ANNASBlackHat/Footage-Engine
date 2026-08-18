"""Standalone script to process all pending media items (Chunking + X-CLIP Embedding + Vector Indexing)."""

import os
import sys
import time

import footage_engine as fe
from footage_engine.config import get_settings
from footage_engine.models.db import init_db
from footage_engine.vector import get_vector_store


def main():
    cfg = get_settings()
    init_db(cfg.DATABASE_URL)
    vec_store = get_vector_store(cfg)

    workers = int(os.environ.get("NUM_WORKERS") or os.environ.get("MAX_WORKERS") or cfg.PROCESS_NUM_WORKERS)
    batch_size = int(os.environ.get("BATCH_SIZE") or os.environ.get("EMBEDDING_BATCH_SIZE") or cfg.EMBEDDING_BATCH_SIZE)

    print("=" * 80, flush=True)
    print("🧠 Footage Engine — High-Speed Batch Processor for Pending Footage", flush=True)
    print("=" * 80, flush=True)
    print(f"• Database       : {cfg.DATABASE_URL}", flush=True)
    print(f"• Storage        : {cfg.STORAGE_BACKEND} ({cfg.LOCAL_STORAGE_DIR})", flush=True)
    print(f"• Vector Store   : {cfg.VECTOR_STORE}", flush=True)
    print(f"• Model & Device : {cfg.DEFAULT_EMBEDDING_MODEL} (Device: {cfg.EMBEDDING_DEVICE})", flush=True)
    print(f"• Batch Size     : {batch_size} chunks / forward pass", flush=True)
    print(f"• Worker Threads : {workers} parallel video worker(s)", flush=True)
    print("=" * 80, flush=True)

    limit = int(os.environ["LIMIT"]) if "LIMIT" in os.environ else None
    if limit:
        print(f"• Processing limit: {limit} items", flush=True)

    start_time = time.time()
    processor = fe.BatchProcessor(vector_store=vec_store)
    
    print("\n⏳ Finding and processing pending media items...", flush=True)
    stats = processor.process_all_pending(limit=limit, max_workers=workers)

    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80, flush=True)
    print(f"✓ Batch Processing Completed in {elapsed:.1f}s!", flush=True)
    print(f"  • Total items processed : {stats['total']}", flush=True)
    print(f"  • Succeeded            : {stats['succeeded']}", flush=True)
    print(f"  • Failed               : {stats['failed']}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
