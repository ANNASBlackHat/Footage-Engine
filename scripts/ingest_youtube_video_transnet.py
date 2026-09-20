"""Ingest and process a single long YouTube video using TransNetV2 with fast parallel clipping."""

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

import footage_engine as fe
from footage_engine.chunking.detector import create_chunks_in_db
from footage_engine.chunking.transnet_detector import (
    parallel_extract_clips,
    preprocess_media_transnet,
)
from footage_engine.config import get_settings
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import MediaItem, MediaStatus
from footage_engine.storage import get_storage_backend
from footage_engine.vector import get_vector_store


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube video into Footage Engine using TransNetV2 deep scene detection."
    )
    parser.add_argument("url", nargs="?", default=None, help="YouTube video URL or local video path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="TransNetV2 cut probability threshold (0.5 default, 0.7 for conservative cuts)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent ffmpeg workers for parallel clipping (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional local directory to export sliced MP4 scene clips (e.g. /content/clips_transnet)",
    )
    parser.add_argument(
        "--stream-copy",
        action="store_true",
        help="Use '-c copy' for instant clipping without re-encoding (cuts at keyframes)",
    )
    parser.add_argument(
        "--use-nvenc",
        action="store_true",
        help="Use NVIDIA GPU hardware encoder (h264_nvenc) for fast re-encoding",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to cookies.txt, remote URL, or raw Netscape/base64 cookie string",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Extract cookies from browser (e.g. 'chrome', 'firefox', 'brave', 'safari')",
    )
    parser.add_argument(
        "--entity", "--entity-name", dest="entity", default=None,
        help="Canonical entity name to associate with this footage (e.g. 'USS Cyclops', 'Aye-aye')",
    )
    parser.add_argument(
        "--entity-type", default="other",
        help="Entity type if creating a new entity ('ship', 'animal', 'person', 'location', 'event', 'other')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    url = args.url

    if args.cookies:
        os.environ["YOUTUBE_COOKIES"] = args.cookies
    if args.cookies_from_browser:
        os.environ["YOUTUBE_COOKIES_FROM_BROWSER"] = args.cookies_from_browser

    if not url:
        url = input("\n🎬 Enter YouTube Video URL or video path: ").strip()

    if not url:
        print("Error: No URL or file path provided.")
        return

    cfg = get_settings()
    init_db(cfg.DATABASE_URL)
    storage = get_storage_backend(cfg)
    vec_store = get_vector_store(cfg) if not args.skip_index else None

    print("=" * 80, flush=True)
    print("🎬 Footage Engine — TransNetV2 Ingestion & Multi-Threaded Clipping", flush=True)
    print("=" * 80, flush=True)
    print(f"• URL / Path   : {url}", flush=True)
    print(f"• Scene Model  : TransNetV2 (Deep Learning Shot Boundary Detection)", flush=True)
    print(f"• Threshold    : {args.threshold}", flush=True)
    print(f"• Workers      : {args.workers} concurrent ffmpeg threads", flush=True)
    print(f"• Fast Seek    : Active (-ss before -i)", flush=True)
    if args.entity:
        print(f"• Entity       : {args.entity} (type: {args.entity_type})", flush=True)
    if args.cookies or cfg.YOUTUBE_COOKIES:
        print(f"• Cookies      : Active ({args.cookies or cfg.YOUTUBE_COOKIES})", flush=True)
    if args.cookies_from_browser or cfg.YOUTUBE_COOKIES_FROM_BROWSER:
        print(f"• Cookies From : {args.cookies_from_browser or cfg.YOUTUBE_COOKIES_FROM_BROWSER}", flush=True)
    if args.output_dir:
        print(f"• Export Clips : {args.output_dir}", flush=True)
    if args.stream_copy:
        print(f"• Stream Copy  : Enabled (-c copy)", flush=True)
    print(f"• Vector Store : {cfg.VECTOR_STORE if not args.skip_index else 'Skipped'}", flush=True)
    print("=" * 80, flush=True)

    start_time = time.time()

    # Step 1: Ingest & fetch metadata
    print("\n[1/4] Fetching video metadata & registering in database...", flush=True)
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

    if item.status == fe.MediaStatus.DONE and not args.output_dir:
        print(f"\n✨ This video has ALREADY been processed and indexed! (Found {len(item.chunks)} chunks in DB).", flush=True)
        print("You can search it immediately using: uv run python scripts/search_cli.py \"<your search query>\"")
        return

    # Resolve local path
    try:
        local_path = storage.get_local_path(item.storage_path)
    except Exception:
        local_path = storage.get_local_path(item.source_url)

    # Step 2: TransNetV2 Scene Detection & Chunking
    print(f"\n[2/4] Running TransNetV2 scene boundary detection (threshold={args.threshold})...", flush=True)
    with get_db_session(cfg.DATABASE_URL) as session:
        db_item = session.get(MediaItem, item.id)
        if not db_item:
            print(f"Error: MediaItem {item.id} not found in DB.")
            return

        chunks = db_item.chunks
        if not chunks:
            candidates = preprocess_media_transnet(
                media_item=db_item,
                storage=storage,
                threshold=args.threshold,
                chunk_threshold_sec=cfg.CHUNK_THRESHOLD_SEC,
                window_sec=cfg.SLIDING_WINDOW_SEC,
                overlap_ratio=cfg.SLIDING_OVERLAP_RATIO,
            )
            print(f"  ✓ TransNetV2 identified {len(candidates)} chunk candidate(s).", flush=True)

            chunks = create_chunks_in_db(
                media_item=db_item,
                candidates=candidates,
                session=session,
            )
            session.commit()
            print(f"  ✓ Stored {len(chunks)} chunk records in database.", flush=True)
        else:
            print(f"  ✓ Found {len(chunks)} existing chunks in database.", flush=True)

    # Step 3: Multi-threaded parallel clipping
    need_local_export = bool(args.output_dir)
    need_cloud_upload = bool(cfg.UPLOAD_CHUNKS_TO_STORAGE)

    if need_local_export or need_cloud_upload:
        print(f"\n[3/4] Parallel clipping {len(chunks)} scene(s) with {args.workers} workers...", flush=True)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        clip_tasks: list[tuple[int, float, float, str]] = []
        tmp_files_to_cleanup: list[str] = []

        for idx, chunk in enumerate(chunks):
            if not chunk.end_ts:
                continue
            if args.output_dir:
                clip_out = os.path.join(args.output_dir, f"scene_{idx+1:03d}.mp4")
            else:
                tmp_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                clip_out = tmp_f.name
                tmp_f.close()
                tmp_files_to_cleanup.append(clip_out)

            clip_tasks.append((idx, chunk.start_ts, chunk.end_ts, clip_out))

        def on_progress(done, total):
            pct = int((done / total) * 100) if total else 100
            print(f"    → Sliced {done}/{total} clips ({pct}%)...", end="\r", flush=True)

        t_clip_start = time.time()
        results = parallel_extract_clips(
            video_path=local_path,
            clips=clip_tasks,
            max_workers=args.workers,
            stream_copy=args.stream_copy,
            use_nvenc=args.use_nvenc,
            progress_callback=on_progress,
        )
        print(f"\n  ✓ Parallel clipping finished in {time.time() - t_clip_start:.2f}s.", flush=True)

        # Handle Cloud/Storage Upload if configured
        if need_cloud_upload:
            print(f"  → Uploading sliced clips to storage backend...", flush=True)
            with get_db_session(cfg.DATABASE_URL) as session:
                db_item = session.get(MediaItem, item.id)
                for (idx, out_path, success), chunk in zip(results, db_item.chunks):
                    if success and os.path.exists(out_path):
                        with open(out_path, "rb") as cf:
                            chunk_bytes = cf.read()
                        saved_path = storage.save_file(
                            chunk_bytes, f"chunks/{db_item.id[:8]}_{chunk.id[:8]}.mp4"
                        )
                        chunk.storage_path = saved_path
                session.commit()
            print("  ✓ Sliced clips uploaded to storage.", flush=True)

        # Cleanup temporary files if created for cloud-only upload
        for tmp_p in tmp_files_to_cleanup:
            if os.path.exists(tmp_p):
                os.remove(tmp_p)
    else:
        print(f"\n[3/4] Physical chunk slicing skipped (UPLOAD_CHUNKS_TO_STORAGE=False and no --output-dir).")

    # Step 4: Batch Processing (Embeddings + Vector Indexing)
    if not args.skip_index:
        print(f"\n[4/4] Computing X-CLIP embeddings & indexing into {cfg.VECTOR_STORE}...", flush=True)
        processor = fe.BatchProcessor(
            storage=storage,
            vector_store=vec_store,
        )
        ok = processor.process_item(item.id)
        if ok:
            elapsed = time.time() - start_time
            print("\n" + "=" * 80, flush=True)
            print(f"✨ Successfully indexed with TransNetV2 in {elapsed:.1f}s!", flush=True)
            print("=" * 80, flush=True)
            print("You can now search inside this video with semantic queries:")
            print('  uv run python scripts/search_cli.py "describe any scene in the video"')
            print("=" * 80, flush=True)
        else:
            print("\n❌ Embedding/indexing failed. Check logs above for details.", flush=True)
    else:
        print("\n✨ Clipping complete. Vector indexing skipped as requested (--skip-index).")


if __name__ == "__main__":
    main()
