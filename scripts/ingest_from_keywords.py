"""Batch ingest footage from Pexels, Pixabay & Coverr using keywords from keywords.txt."""

import os
import sys
import time
from pathlib import Path

import footage_engine as fe
from footage_engine.config import get_settings
from footage_engine.models.db import init_db
from footage_engine.vector import get_vector_store


def load_keywords(file_path: str = "keywords.txt") -> list[str]:
    """Parses keywords from a comma-separated or multi-line text file."""
    p = Path(file_path)
    if not p.exists():
        print(f"Error: {file_path} not found.", flush=True)
        return []

    content = p.read_text(encoding="utf-8")
    # Split by comma or newline
    raw_tokens = content.replace("\n", ",").split(",")
    keywords = [t.strip().strip("*`\"'") for t in raw_tokens if t.strip()]
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique_keywords.append(kw)
    return unique_keywords


def main():
    cfg = get_settings()
    init_db(cfg.DATABASE_URL)
    vec_store = get_vector_store(cfg)

    # 1. Load keywords from keywords.txt
    keywords = load_keywords("keywords.txt")
    if not keywords:
        print("No keywords found to process.", flush=True)
        return

    # Providers to query (excluding YouTube)
    providers = ["pexels", "pixabay", "coverr"]
    max_per_kw = int(os.environ.get("MAX_RESULTS_PER_KEYWORD", "2"))
    media_type = os.environ.get("MEDIA_TYPE", cfg.DEFAULT_MEDIA_TYPE).lower()
    orientation = (os.environ.get("ORIENTATION") or os.environ.get("ASSET_ORIENTATION") or cfg.DEFAULT_ORIENTATION).lower()

    print("=" * 80, flush=True)
    print("🎬 Footage Engine — Multi-Keyword Ingestion (Pexels, Pixabay & Coverr)", flush=True)
    print("=" * 80, flush=True)
    print(f"• Total unique keywords: {len(keywords)}", flush=True)
    print(f"• Providers to query   : {', '.join(providers).upper()}", flush=True)
    print(f"• Max items per search : {max_per_kw}", flush=True)
    print(f"• Media type           : {media_type.upper()}", flush=True)
    print(f"• Asset orientation    : {orientation.upper()}", flush=True)
    print(f"• Storage backend      : {cfg.STORAGE_BACKEND}", flush=True)
    print(f"• Vector store         : {cfg.VECTOR_STORE}", flush=True)
    print("=" * 80, flush=True)

    total_ingested = 0
    start_time = time.time()

    # Ingestion Pass
    for idx, kw in enumerate(keywords, 1):
        print(f"\n[{idx}/{len(keywords)}] Keyword: '{kw}'", flush=True)
        for prov in providers:
            try:
                items = fe.search_and_ingest(
                    keyword=kw,
                    provider=prov,
                    max_results=max_per_kw,
                    media_type=media_type,
                    orientation=orientation,
                )
                if items:
                    print(f"   ✓ {prov.upper():<8}: Ingested/found {len(items)} asset(s)", flush=True)
                    total_ingested += len(items)
                else:
                    print(f"   - {prov.upper():<8}: No assets found or provider skipped", flush=True)
            except Exception as e:
                err_msg = str(e)
                if "API key not found" in err_msg or "401" in err_msg:
                    print(f"   ⚠️  {prov.upper():<8}: API key missing in .env", flush=True)
                else:
                    print(f"   ⚠️  {prov.upper():<8}: {e}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"📥 Ingestion phase complete! Total candidate items processed: {total_ingested}", flush=True)
    print("=" * 80, flush=True)

    # Batch Processing Pass (Chunking, X-CLIP embeddings, Vector Indexing)
    run_processing = os.environ.get("SKIP_PROCESSING") != "1"
    if run_processing:
        print("\n🧠 Running Batch Processor (Scene detection, X-CLIP embedding & Vector indexing)...", flush=True)
        processor = fe.BatchProcessor(vector_store=vec_store)
        stats = processor.process_all_pending()
        print(f"\n✓ Batch processing results: {stats}", flush=True)
    else:
        print("\n(Skipping batch processing because SKIP_PROCESSING=1)", flush=True)

    elapsed = time.time() - start_time
    print(f"\n✨ All done in {elapsed:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
