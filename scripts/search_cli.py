import argparse
import sys
import footage_engine as fe
from footage_engine.retrieval.models import SearchFilters


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI search tool for querying embedded video footage."
    )
    parser.add_argument("query", nargs="*", default=None, help="Search query text")
    parser.add_argument(
        "--beat", action="store_true",
        help="Treat query as a narrative voiceover beat (runs Multi-Query expansion + entity detection)"
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Enable optional LLM judge to rerank candidates against the original beat text"
    )
    parser.add_argument(
        "--confidence-floor", type=float, default=None,
        help="Minimum confidence/similarity threshold (filters out lower scoring clips)"
    )
    parser.add_argument(
        "--entity", "--entity-name", dest="entity", default=None,
        help="Scope search to a specific canonical entity (e.g. 'USS Cyclops', 'Aye-aye')"
    )
    parser.add_argument(
        "--provider", default=None,
        help="Filter by provider ('pexels', 'pixabay', 'coverr', 'youtube', 'manual')"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)"
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip() if args.query else ""
    if not query:
        prompt_text = "Enter voiceover script beat" if args.beat else "Enter search query"
        query = input(f"\n🔍 {prompt_text}: ").strip()

    if not query:
        print("Empty query. Exiting.")
        return

    print("=" * 80)
    print("🎬 Footage Engine — " + ("Script Beat Search (Multi-Query)" if args.beat else "Semantic Video Search"))
    print(f"🔎 {'Beat' if args.beat else 'Query'}: \"{query}\"")
    if args.entity:
        print(f"🏷️  Entity Filter: {args.entity}")
    if args.provider:
        print(f"📦 Provider Filter: {args.provider}")
    if args.beat and args.rerank:
        print(f"🧠 LLM Judge Reranker: Active")
    if args.confidence_floor is not None:
        print(f"🛡️  Confidence Floor: {args.confidence_floor}")
    print("=" * 80)

    filters = None
    if args.entity or args.provider:
        filters = SearchFilters(
            entity_name=args.entity,
            provider=args.provider,
        )

    retrieval = fe.get_retrieval_api()

    if args.beat:
        results = retrieval.search_beat(
            beat_text=query,
            top_k=args.top_k,
            filters=filters,
            rerank=args.rerank,
            confidence_floor=args.confidence_floor,
        )
    else:
        results = retrieval.search(query=query, top_k=args.top_k, filters=filters)

    if not results:
        print("\nNo matching video chunks found. (Try adjusting filters or confidence floor).")
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
        if res.entity_name:
            print(f"   • Entity     : {res.entity_name}")
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
