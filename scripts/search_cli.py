"""Interactive CLI search tool for querying embedded video footage."""

import sys
import footage_engine as fe
from footage_engine.retrieval.models import SearchFilters


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("\n🔍 Enter search query: ").strip()

    if not query:
        print("Empty query. Exiting.")
        return

    print("=" * 80)
    print(f"🎬 Footage Engine — Semantic Video Search")
    print(f"🔎 Query: \"{query}\"")
    print("=" * 80)

    retrieval = fe.get_retrieval_api()
    results = retrieval.search(query=query, top_k=5)

    if not results:
        print("\nNo matching video chunks found. (Make sure you have embedded footage in the DB).")
        return

    for rank, res in enumerate(results, 1):
        if res.start_ts is not None and res.end_ts is not None:
            dur = res.end_ts - res.start_ts
            time_str = f"[{res.start_ts:.1f}s - {res.end_ts:.1f}s] ({dur:.1f}s)"
        else:
            time_str = "[Still Photo]"

        print(f"\n🏆 Rank #{rank} (Similarity Score: {res.score:.4f})")
        print(f"   • Chunk ID   : {res.chunk_id}")
        print(f"   • Provider   : {res.provider.upper()} ({res.media_type})")
        print(f"   • Time Range : {time_str}")
        print(f"   • Stream URL : {res.storage_url or res.source_url}")

    # Fine localization on top result
    top_video = next((r for r in results if r.media_type == "video"), None)
    if top_video:
        print("\n" + "-" * 80)
        print(f"✂️  Running Sub-Second Fine Localization on Rank #1 chunk ({top_video.chunk_id[:8]})...")
        try:
            sub_s, sub_e = retrieval.fine_localize(top_video.chunk_id, query=query, fps=1.0)
            print(f"   • Full Scene Window   : [{top_video.start_ts:.2f}s - {top_video.end_ts:.2f}s]")
            print(f"   • Refined Sub-clip Cut: [{sub_s:.2f}s - {sub_e:.2f}s] (Duration: {sub_e - sub_s:.2f}s)")
        except Exception as e:
            print(f"   (Fine localization notice: {e})")
        print("-" * 80)


if __name__ == "__main__":
    main()
