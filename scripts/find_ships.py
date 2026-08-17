"""Script to search and ingest footage about ships from Pixabay, process embeddings, and query them."""

import os
import sys
import footage_engine as fe
from footage_engine.retrieval.models import SearchFilters


def main():
    print("=" * 75)
    print("🚢 Footage Engine — Searching Pixabay for 'ships'")
    print("=" * 75)

    cfg = fe.get_settings()
    print(f"\nConfiguration:")
    print(f"  • Database URL: {cfg.DATABASE_URL.split('@')[-1] if '@' in cfg.DATABASE_URL else cfg.DATABASE_URL}")
    print(f"  • Storage Backend: {cfg.STORAGE_BACKEND}")
    print(f"  • Pixabay Key Configured: {'Yes' if cfg.PIXABAY_API_KEY else 'No'}")

    if not cfg.PIXABAY_API_KEY:
        print("\n❌ Error: PIXABAY_API_KEY is not set in your .env file.")
        sys.exit(1)

    # 1. Search and Ingest from Pixabay
    keyword = "ships"
    print(f"\n[1/4] Querying Pixabay API for '{keyword}' (max 5 results)...")
    try:
        items = fe.search_and_ingest(keyword=keyword, provider="pixabay", max_results=5)
    except Exception as e:
        print(f"❌ Failed to search/ingest from Pixabay: {e}")
        sys.exit(1)

    print(f"✓ Found and ingested {len(items)} media items into PostgreSQL:")
    for idx, item in enumerate(items, 1):
        print(f"  [{idx}] ID: {item.id}")
        print(f"      Source ID: {item.source_id} | Duration: {item.duration_sec}s | Resolution: {item.resolution}")
        print(f"      Status: {item.status.value}")
        print(f"      Storage Path: {item.storage_path}")
        print(f"      Source URL: {item.source_url}")

    # 2. Run Batch Processing Pipeline
    print("\n[2/4] Running Batch Processor (PySceneDetect + X-CLIP Embeddings)...")
    from footage_engine.vector import get_vector_store
    from footage_engine.models.db import get_db_session
    from footage_engine.models.media import MediaItem, MediaStatus

    # Reset any DONE items to PENDING for in-memory re-indexing
    with get_db_session() as s:
        for it in s.query(MediaItem).all():
            it.status = MediaStatus.PENDING
        s.commit()

    vec_store = get_vector_store()
    processor = fe.BatchProcessor(vector_store=vec_store)
    stats = processor.process_all_pending()
    print(f"✓ Batch processing results: {stats}")

    # 3. Multimodal Search Queries
    search_queries = [
        "cruise ship sailing in harbour ocean",
        "container cargo shipping terminal port",
        "aerial view of large vessel in water",
    ]
    retrieval_api = fe.RetrievalAPI(vector_store=vec_store)
    search_query = "large cargo ship sailing in deep blue ocean water"
    print(f"\n[3/4] Searching with natural language query: '{search_query}'...")
    results = retrieval_api.search(query=search_query, top_k=5)

    print(f"\nTop {len(results)} matches retrieved:")
    for i, res in enumerate(results, 1):
        print(f"\n  Rank #{i}: Similarity Score: {res.score:.4f}")
        print(f"  • Chunk ID: {res.chunk_id} (Media Item: {res.media_item_id})")
        print(f"  • Range: [{res.start_ts:.2f}s - {res.end_ts:.2f}s] (Duration: {res.duration_sec:.2f}s)")
        print(f"  • Resolution: {res.resolution} | Provider: {res.provider} | License: {res.license_type}")
        print(f"  • Storage URL: {res.storage_url}")
        if res.tags:
            print(f"  • Tags: {', '.join(res.tags[:6])}")

    # 4. Fine-grained Action Sub-clip Localization
    if results:
        top_chunk = results[0]
        print(f"\n[4/4] Pinpointing exact action window inside winning chunk {top_chunk.chunk_id[:8]}...")
        sub_start, sub_end = retrieval_api.fine_localize(top_chunk.chunk_id, query=search_query)
        print(f"✓ Original Chunk Window : [{top_chunk.start_ts:.2f}s - {top_chunk.end_ts:.2f}s]")
        print(f"✓ Refined Sub-clip Cut  : [{sub_start:.2f}s - {sub_end:.2f}s] (Duration: {sub_end - sub_start:.2f}s)")

    print("\n" + "=" * 75)
    print("✨ Ingestion, embedding, and retrieval pipeline finished successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
