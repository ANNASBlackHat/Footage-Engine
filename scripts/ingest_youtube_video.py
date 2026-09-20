import argparse
import sys
import time

import footage_engine as fe
from footage_engine.config import get_settings
from footage_engine.models.db import init_db
from footage_engine.vector import get_vector_store


def main():
    parser = argparse.ArgumentParser(
        description="Ingest and process a single YouTube video into retrievable scene chunks."
    )
    parser.add_argument("url", nargs="?", default=None, help="YouTube Video URL")
    parser.add_argument(
        "--entity", "--entity-name", dest="entity", default=None,
        help="Canonical entity name to associate with this footage (e.g. 'USS Cyclops', 'Aye-aye')"
    )
    parser.add_argument(
        "--entity-type", default="other",
        help="Entity type if creating a new entity ('ship', 'animal', 'person', 'location', 'event', 'other')"
    )
    args = parser.parse_args()

    url = args.url
    if not url:
        url = input("\n🎬 Enter YouTube Video URL: ").strip()

    if not url:
        print("Error: No URL provided.")
        return

    cfg = get_settings()
    init_db(cfg.DATABASE_URL)
    vec_store = get_vector_store(cfg)

    print("=" * 80, flush=True)
    print("🎬 Footage Engine — Long YouTube Video Ingestion & Chunking", flush=True)
    print("=" * 80, flush=True)
    print(f"• URL          : {url}", flush=True)
    if args.entity:
        print(f"• Entity       : {args.entity} (type: {args.entity_type})", flush=True)
    print(f"• Database     : {cfg.DATABASE_URL}", flush=True)
    print(f"• Vector Store : {cfg.VECTOR_STORE}", flush=True)
    print(f"• Scene Split  : Adaptive Scene Detection + {cfg.SLIDING_WINDOW_SEC}s Sliding Window", flush=True)
    print("=" * 80, flush=True)

    start_time = time.time()

    # Step 1: Ingest & extract metadata
    print("\n[1/3] Fetching video metadata & registering in database...", flush=True)
    item = fe.ingest(
        source_url=url,
        entity_name=args.entity,
        entity_type=args.entity_type,
    )
    
    dur_str = f"{item.duration_sec:.1f}s ({item.duration_sec / 60:.1f} min)" if item.duration_sec else "unknown"
    print(f"  ✓ MediaItem ID : {item.id}", flush=True)
    print(f"  ✓ Title        : {item.item_metadata.get('title', 'N/A')}", flush=True)
    print(f"  ✓ Duration     : {dur_str}", flush=True)
    print(f"  ✓ Resolution   : {item.resolution or 'Probing on download'} ({item.orientation})", flush=True)
    print(f"  ✓ Status       : {item.status.value}", flush=True)

    if item.status == fe.MediaStatus.DONE:
        print(f"\n✨ This video has ALREADY been processed and indexed! (Found {len(item.chunks)} chunks in DB).", flush=True)
        print("You can search it immediately using: uv run python scripts/search_cli.py \"<your search query>\"")
        return

    # Step 2: Batch Process (Scene chunking + X-CLIP embedding + Vector indexing)
    print(f"\n[2/3] Processing video: Running PySceneDetect + X-CLIP multimodal embeddings...", flush=True)
    processor = fe.BatchProcessor(vector_store=vec_store)
    ok = processor.process_item(item.id)

    if ok:
        elapsed = time.time() - start_time
        print("\n" + "=" * 80, flush=True)
        print(f"✨ Successfully indexed in {elapsed:.1f}s!", flush=True)
        print("=" * 80, flush=True)
        print(f"You can now search inside this video with semantic queries:")
        print(f"  uv run python scripts/search_cli.py \"describe any scene in the video\"")
        print("=" * 80, flush=True)
    else:
        print(f"\n❌ Processing failed. Check logs above for details.", flush=True)


if __name__ == "__main__":
    main()
