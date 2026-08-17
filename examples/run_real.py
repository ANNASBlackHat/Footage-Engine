"""Script to run Footage Retrieval Engine with REAL data and REAL X-CLIP embeddings."""

import os
import footage_engine as fe
from footage_engine.retrieval.models import SearchFilters


def main():
    print("=" * 75)
    print("🎬 Running Footage Retrieval Engine with REAL Data & X-CLIP Embeddings")
    print("=" * 75)

    cfg = fe.get_settings()
    print(f"\nConfiguration:")
    print(f"  • DB: {cfg.DATABASE_URL}")
    print(f"  • Storage: {cfg.STORAGE_BACKEND} ({cfg.LOCAL_STORAGE_DIR})")
    print(f"  • Model: {cfg.DEFAULT_EMBEDDING_MODEL}")
    print(f"  • Device: {cfg.EMBEDDING_DEVICE}")

    # 1. Ingestion
    print("\n[Step 1/4] Ingesting Real Footage...")
    
    # Check if Pexels API key is present for live keyword search
    if cfg.PEXELS_API_KEY:
        print("  → Searching Pexels for 'nature drone shots'...")
        items = fe.search_and_ingest("nature drone", provider="pexels", max_results=3)
    elif cfg.PIXABAY_API_KEY:
        print("  → Searching Pixabay for 'waterfall'...")
        items = fe.search_and_ingest("waterfall", provider="pixabay", max_results=3)
    else:
        print("  → No provider API key found in .env; using public creative-commons video URLs...")
        sample_urls = [
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
        ]
        items = fe.ingest_url_list(sample_urls, provider="direct_feed")

    for item in items:
        print(f"  ✓ Ingested: {item.id} | Provider: {item.provider} | Duration: {item.duration_sec}s | Status: {item.status.value}")

    # 2. Resumable Batch Processing
    print("\n[Step 2/4] Running Batch Processor (PySceneDetect + X-CLIP Embeddings + Vector Indexing)...")
    processor = fe.BatchProcessor()
    stats = processor.process_all_pending()
    print(f"  ✓ Batch processing stats: {stats}")

    # 3. Semantic Search
    print("\n[Step 3/4] Running Semantic Search...")
    queries = [
        "scenic outdoor mountains and blue sky",
        "action running fast movement",
    ]

    for q in queries:
        print(f"\n  🔍 Query: '{q}'")
        results = fe.search(query=q, top_k=3)
        for i, res in enumerate(results, 1):
            print(f"    [{i}] Score: {res.score:.4f} | Range: [{res.start_ts:.1f}s - {res.end_ts:.1f}s]")
            print(f"        Chunk ID: {res.chunk_id} | Provider: {res.provider}")
            print(f"        URL: {res.storage_url}")

    # 4. Fine Localization
    if results:
        top_chunk = results[0]
        print(f"\n[Step 4/4] Fine Localizing query within Chunk {top_chunk.chunk_id[:8]}...")
        sub_start, sub_end = fe.fine_localize(top_chunk.chunk_id, query=queries[0])
        print(f"  ✓ Refined timeline window: [{sub_start:.2f}s - {sub_end:.2f}s] (ready for timeline cut)")

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
