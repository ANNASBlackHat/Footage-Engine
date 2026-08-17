"""Ingest curated footage across narrative segments from keywords.txt using Pexels, Pixabay & Coverr."""

import sys
import time
from footage_engine.config import get_settings
from footage_engine.models.db import init_db, get_db_session
from footage_engine.models.media import MediaItem
from footage_engine.vector import get_vector_store
import footage_engine as fe

def main():
    cfg = get_settings()
    init_db(cfg.DATABASE_URL)
    vec_store = get_vector_store()

    print("=" * 80)
    print("🎬 Footage Retrieval Engine — Multi-Provider Narrative Ingestion & Search")
    print("=" * 80)
    print(f"• Vector Store : {cfg.VECTOR_STORE} ({cfg.ZILLIZ_COLLECTION_NAME})")
    print(f"• Database     : PostgreSQL ({cfg.DATABASE_URL.split('@')[-1].split('/')[0]})")
    print(f"• Providers    : Pixabay, Pexels (Videos & Photos), Coverr")

    # Curated narrative segment experiments from keywords.txt
    narrative_experiments = [
        # Segment 1: Exterior & Scale (Videos)
        {"keyword": "container ship aerial", "provider": "pexels", "media_type": "video", "max": 2, "segment": "1. Exterior & Scale"},
        {"keyword": "shipping containers drone", "provider": "pixabay", "media_type": "video", "max": 2, "segment": "1. Exterior & Scale"},
        
        # Segment 2: Interior Cabin & Details (Photos & Videos)
        {"keyword": "ship cabin interior", "provider": "pexels", "media_type": "video", "max": 2, "segment": "2. Interior Cabin"},
        {"keyword": "coffee mug on desk", "provider": "pexels", "media_type": "image", "max": 2, "segment": "2. Interior Details"},
        {"keyword": "looking out ship window ocean", "provider": "pexels", "media_type": "video", "max": 2, "segment": "2. Cabin View"},

        # Segment 3: Navigation & Bridge (Videos)
        {"keyword": "ship navigation bridge", "provider": "pexels", "media_type": "video", "max": 2, "segment": "3. Navigation & Bridge"},
        {"keyword": "radar screen ship", "provider": "pixabay", "media_type": "video", "max": 2, "segment": "3. Radar & Bridge"},

        # Segment 4: Operations, Admin & Safe (Photos & Videos)
        {"keyword": "us dollar bills cash stack", "provider": "pexels", "media_type": "image", "max": 2, "segment": "4. Cash & Safe"},
        {"keyword": "passports stack on desk", "provider": "pexels", "media_type": "image", "max": 2, "segment": "4. Documents"},
        {"keyword": "signing contract document paperwork", "provider": "pexels", "media_type": "video", "max": 2, "segment": "4. Operations"},

        # Segment 5: Engine Room & Rough Seas (Videos)
        {"keyword": "stormy rough ocean waves", "provider": "coverr", "media_type": "video", "max": 2, "segment": "5. Engine Room & Rough Seas"},
        {"keyword": "cargo ship engine room", "provider": "pexels", "media_type": "video", "max": 2, "segment": "5. Engine Room"},

        # Segment 6: Solitude & Modern Tech (Videos & Photos)
        {"keyword": "man sitting alone staring window", "provider": "pexels", "media_type": "video", "max": 2, "segment": "6. Solitude"},
        {"keyword": "laptop video call night", "provider": "pexels", "media_type": "video", "max": 2, "segment": "6. Modern Tech"},
    ]

    total_ingested = []
    print("\n" + "-" * 80)
    print("📥 [1/3] Searching & Ingesting Curated Narrative Assets (Pre-Spend Deduplication Active)")
    print("-" * 80)

    for exp in narrative_experiments:
        seg = exp["segment"]
        kw = exp["keyword"]
        prov = exp["provider"]
        mtype = exp["media_type"]
        max_r = exp["max"]
        print(f"\n📂 [{seg}] Querying {prov.upper()} for '{kw}' ({mtype}s)...")
        try:
            items = fe.search_and_ingest(keyword=kw, provider=prov, max_results=max_r, media_type=mtype)
            print(f"   ✓ Ingested/deduped {len(items)} item(s).")
            total_ingested.extend(items)
        except Exception as e:
            print(f"   ⚠️  Failed to ingest '{kw}' from {prov}: {e}")

    # 2. Batch Processing Pipeline
    print("\n" + "-" * 80)
    print("🧠 [2/3] Computing X-CLIP Embeddings & Indexing into Zilliz Cloud")
    print("-" * 80)
    processor = fe.BatchProcessor(vector_store=vec_store)
    stats = processor.process_all_pending()
    print(f"\n✓ Batch processing results: {stats}")

    # 3. Multimodal Search Queries across the entire ingested library
    print("\n" + "-" * 80)
    print("🔍 [3/3] Testing Semantic Video & Image Retrieval across Narrative Moments")
    print("-" * 80)

    test_queries = [
        "aerial drone view of gigantic cargo vessel loaded with containers",
        "lonely captain in dark cabin looking out rainy ocean window",
        "stacks of hundred dollar bills cash on office table",
        "violent dark sea storm huge waves crashing",
        "man typing on modern laptop late at night alone",
    ]

    retrieval_api = fe.RetrievalAPI(vector_store=vec_store)

    for q in test_queries:
        print(f"\n🔎 Query: \"{q}\"")
        results = retrieval_api.search(query=q, top_k=3)
        for rank, res in enumerate(results, 1):
            if res.start_ts is not None and res.end_ts is not None:
                dur = res.end_ts - res.start_ts
                dur_str = f"{dur:.1f}s"
                ts_str = f"[{res.start_ts:.1f}s - {res.end_ts:.1f}s]"
            else:
                dur_str = "still photo"
                ts_str = "[photo]"
            print(f"   Rank #{rank} (Score: {res.score:.4f}) | {res.provider.upper()} ({res.media_type}) | {ts_str} ({dur_str})")
            print(f"      🔗 URL: {res.storage_url or res.source_url}")

    # Fine localization on the top result of the first query
    first_res = retrieval_api.search(query=test_queries[0], top_k=1)
    if first_res and first_res[0].media_type == "video":
        winner = first_res[0]
        print(f"\n✂️  [Fine Localization Demo] Pinpointing sub-second cut for chunk {winner.chunk_id[:8]}...")
        sub_s, sub_e = retrieval_api.fine_localize(winner.chunk_id, query=test_queries[0])
        print(f"   • Original Chunk Window : [{winner.start_ts:.2f}s - {winner.end_ts:.2f}s]")
        print(f"   • Refined Sub-clip Cut  : [{sub_s:.2f}s - {sub_e:.2f}s] (Duration: {sub_e - sub_s:.2f}s)")

    print("\n" + "=" * 80)
    print("✨ Narrative Footage Retrieval Experiment Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
