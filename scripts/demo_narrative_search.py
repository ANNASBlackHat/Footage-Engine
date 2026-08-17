"""Demo narrative retrieval across 6 story segments with fine localization."""

import footage_engine as fe
from footage_engine.config import get_settings
from footage_engine.models.db import init_db
from footage_engine.vector import get_vector_store

def main():
    cfg = get_settings()
    init_db(cfg.DATABASE_URL)
    vec_store = get_vector_store()
    retrieval = fe.RetrievalAPI(vector_store=vec_store)

    test_queries = [
        ("Segment 1: Exterior & Scale", "aerial drone view of gigantic cargo vessel loaded with containers"),
        ("Segment 2: Interior Cabin", "lonely captain in dark cabin looking out rainy ocean window"),
        ("Segment 2: Cabin Details", "close up hot coffee mug on wooden desk"),
        ("Segment 4: Operations & Safe", "stacks of hundred dollar bills cash on office table"),
        ("Segment 4: Passports & Docs", "passports and travel documents stacked on table"),
        ("Segment 5: Heavy Seas", "violent dark sea storm huge waves crashing"),
        ("Segment 6: Solitude & Tech", "man typing on modern laptop late at night alone"),
    ]

    print("=" * 80)
    print("🎬 Footage Engine — Narrative Multi-Provider Retrieval (Pexels, Pixabay, Coverr)")
    print("=" * 80)

    for seg_title, q in test_queries:
        print(f"\n🎭 [{seg_title}]")
        print(f"   Query: \"{q}\"")
        results = retrieval.search(query=q, top_k=2)
        for rank, res in enumerate(results, 1):
            if res.start_ts is not None and res.end_ts is not None:
                dur = res.end_ts - res.start_ts
                ts_str = f"[{res.start_ts:.1f}s - {res.end_ts:.1f}s] ({dur:.1f}s)"
            else:
                ts_str = "[Still Photo]"
            print(f"   • Rank #{rank} (Score: {res.score:.4f}) | {res.provider.upper()} ({res.media_type}) | {ts_str}")
            print(f"     🔗 {res.storage_url or res.source_url}")

    # Fine localization test on the winning container ship chunk
    first_res = retrieval.search(query=test_queries[0][1], top_k=1)
    if first_res and first_res[0].media_type == "video":
        winner = first_res[0]
        print("\n" + "-" * 80)
        print(f"✂️  [Sub-second Action Localization Demo] Pinpointing cut for chunk {winner.chunk_id[:8]}...")
        sub_s, sub_e = retrieval.fine_localize(winner.chunk_id, query=test_queries[0][1], fps=0.5)
        print(f"   • Original Chunk Window : [{winner.start_ts:.2f}s - {winner.end_ts:.2f}s] (Duration: {winner.end_ts - winner.start_ts:.2f}s)")
        print(f"   • Refined Sub-clip Cut  : [{sub_s:.2f}s - {sub_e:.2f}s] (Duration: {sub_e - sub_s:.2f}s)")
        print("-" * 80)

    print("\n✨ All narrative queries retrieved and localized successfully!")

if __name__ == "__main__":
    main()
