"""Ingest and process a single long YouTube video using TransNetV2 with fast parallel clipping."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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

# ImageKit free-plan per-video upload cap (paid plans are higher: Lite 300MB, Pro 2GB).
# Source: https://imagekit.io/docs/api-reference/upload-file/upload-file (File size limit)
IMAGEKIT_FREE_PLAN_VIDEO_LIMIT_MB = 100

# ---------------------------------------------------------------------------
# Telegram reporting
#
# Credentials come from whichever runtime we happen to be in, because the same
# script runs locally, on Kaggle, and on Colab and each exposes secrets
# differently. Resolution order is first-non-empty-wins:
#   1. runtime secrets  (kaggle_secrets on Kaggle, google.colab.userdata on Colab)
#   2. .env file in the repo root (hand-parsed, so no extra import is required)
#   3. os.environ
# If neither TELEGRAM_TOKEN nor TELEGRAM_CHAT_ID resolves, reporting is skipped.
# ---------------------------------------------------------------------------

_RUNTIME = None  # cached: "kaggle" | "colab" | None
_DOTENV = None  # cached parsed .env dict


def _detect_runtime():
    """Identify the hosted runtime, if any. Kaggle wins over Colab."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME or None

    def _has_module(name):
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    if _has_module("kaggle_secrets") or os.path.isdir("/kaggle/working"):
        _RUNTIME = "kaggle"
    elif _has_module("google.colab") or os.path.isdir("/content"):
        _RUNTIME = "colab"
    else:
        _RUNTIME = ""
    return _RUNTIME or None


def _load_dotenv():
    """Parse repo_root/.env once into a dict. Missing file is not an error."""
    global _DOTENV
    if _DOTENV is not None:
        return _DOTENV

    values: dict[str, str] = {}
    env_path = Path(repo_root) / ".env"
    try:
        with open(env_path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, _, value = line.partition("=")
                value = value.strip().strip("'\"")
                if key.strip() and value:
                    values[key.strip()] = value
    except OSError:
        pass  # no .env here — fall through to os.environ

    _DOTENV = values
    return _DOTENV


def _resolve_env(name: str):
    """First non-empty value of `name` across runtime secrets, .env, os.environ."""
    runtime = _detect_runtime()

    if runtime == "kaggle":
        try:
            from kaggle_secrets import UserSecretClient

            value = UserSecretClient().get_secret(name)
            if value:
                return value.strip()
        except Exception:
            pass  # secret not attached, or kaggle_secrets is a stub

    elif runtime == "colab":
        try:
            # userdata.get raises SecretNotFoundError when unset or not granted.
            from google.colab import userdata

            value = userdata.get(name)
            if value:
                return value.strip()
        except Exception:
            pass

    value = _load_dotenv().get(name)
    if value:
        return value

    value = os.environ.get(name)
    return value.strip() if value else None


def _send_telegram(text: str) -> bool:
    """Best-effort report to Telegram. Never raises; returns False on any problem."""
    token = _resolve_env("TELEGRAM_TOKEN")
    chat_id = _resolve_env("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        missing = [
            k for k, v in (("TELEGRAM_TOKEN", token), ("TELEGRAM_CHAT_ID", chat_id)) if not v
        ]
        print(f"  ℹ️ Telegram report skipped (not set: {', '.join(missing)}).", flush=True)
        return False

    masked = f"{token[:8]}…" if len(token) > 8 else "***"
    try:
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        if resp.ok:
            print("  📤 Telegram report sent.", flush=True)
            return True
        print(f"  ⚠️ Telegram send failed (HTTP {resp.status_code}): {resp.text[:200]}", flush=True)
    except Exception as err:
        print(f"  ⚠️ Telegram send failed ({type(err).__name__}: {err}) [token {masked}]", flush=True)
    return False


def _item_line(summary: dict) -> str:
    bits = []
    if summary.get("item_id"):
        bits.append(f"🆔 {summary['item_id'][:8]}")
    if summary.get("duration"):
        bits.append(f"⏳ {summary['duration']:.1f}s")
    return " · ".join(bits) or "🆔 -"


def _esc(text: str) -> str:
    """Minimal HTML escape for the HTML parse_mode message body."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_report(summary: dict) -> str:
    """Build a compact HTML message from the run summary."""
    status = summary.get("status")
    title = summary.get("title") or "Untitled"

    if status == "skipped":
        return f"✅ <b>Already indexed</b>\n{_item_line(summary)}"

    # YouTube titles routinely contain & < >, which would abort an HTML parse.
    title = _esc(title)
    error = _esc(summary.get("error", ""))

    icon = "✅" if status == "success" else "❌"
    lines = [f"{icon} <b>{title}</b>", _item_line(summary)]

    if summary.get("chunks"):
        lines.append(
            f"🎞 {summary['chunks']} chunk(s) · {summary.get('uploaded', 0)} uploaded "
            f"· {summary.get('upload_failed', 0)} failed"
        )
    if summary.get("qwen"):
        lines.append("🧠 Qwen embeddings: on")
    if summary.get("elapsed"):
        lines.append(f"⏱ {summary['elapsed']:.1f}s")
    lines.append(f"▶️ {summary.get('url', '')}")

    if summary.get("error"):
        lines.append(f"\n<pre>{error}</pre>")

    return "\n".join(lines)


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
        "--skip-index",
        action="store_true",
        help="Skip X-CLIP embedding & vector indexing (only detect scenes and slice clips)",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip Qwen2-VL embeddings even if Qwen Zilliz credentials or collection are set in environment",
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
        "--upload-chunks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Upload sliced scene chunks to cloud storage (overrides UPLOAD_CHUNKS_TO_STORAGE env)",
    )
    parser.add_argument(
        "--entity-type", default="other",
        help="Entity type if creating a new entity ('ship', 'animal', 'person', 'location', 'event', 'other')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    summary: dict = {"url": args.url or "", "status": "failed"}

    try:
        _run(args, summary)
    except Exception as err:
        summary["status"] = "failed"
        summary["error"] = f"{type(err).__name__}: {err}"[:900]
        summary.setdefault("elapsed", time.time() - summary.get("_t0", time.time()))
        print(f"\n❌ Ingestion failed: {type(err).__name__}: {err}", flush=True)
        _send_telegram(_format_report(summary))
        raise
    else:
        _send_telegram(_format_report(summary))


def _run(args, summary: dict):
    url = args.url
    summary["_t0"] = time.time()

    if args.cookies:
        os.environ["YOUTUBE_COOKIES"] = args.cookies
    if args.cookies_from_browser:
        os.environ["YOUTUBE_COOKIES_FROM_BROWSER"] = args.cookies_from_browser

    if not url:
        url = input("\n🎬 Enter YouTube Video URL or video path: ").strip()

    if not url:
        print("Error: No URL or file path provided.")
        summary["status"] = "failed"
        summary["error"] = "No URL or file path provided."
        return

    summary["url"] = url

    cfg = get_settings()
    init_db(cfg.DATABASE_URL)
    storage = get_storage_backend(cfg)
    vec_store = get_vector_store(cfg) if not args.skip_index else None

    # Auto-detect if Qwen vector credentials / collection are configured
    has_qwen_configured = bool(
        (cfg.QWEN_ZILLIZ_URI and cfg.QWEN_ZILLIZ_TOKEN)
        or cfg.QWEN_ZILLIZ_COLLECTION_NAME
    )
    run_qwen = has_qwen_configured and not args.skip_qwen and not args.skip_index

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
    if run_qwen:
        qwen_col = cfg.QWEN_ZILLIZ_COLLECTION_NAME or cfg.ZILLIZ_COLLECTION_NAME
        print(f"• Qwen Embed   : Enabled (Collection: {qwen_col})", flush=True)
    elif has_qwen_configured and args.skip_qwen:
        print(f"• Qwen Embed   : Skipped (--skip-qwen flag)", flush=True)
    else:
        print(f"• Qwen Embed   : Not configured (QWEN_ZILLIZ_URI/TOKEN or QWEN_ZILLIZ_COLLECTION_NAME not set)", flush=True)
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

    summary["item_id"] = item.id
    summary["title"] = item.item_metadata.get("title", "Untitled")
    summary["duration"] = item.duration_sec
    summary["elapsed"] = time.time() - summary["_t0"]

    if item.status == fe.MediaStatus.DONE and not args.output_dir:
        with get_db_session(cfg.DATABASE_URL) as session:
            db_item = session.get(MediaItem, item.id)
            has_qwen_chunks = any(
                c.embedding_model == cfg.QWEN_MODEL_NAME
                for c in (db_item.chunks if db_item else [])
            )
        if run_qwen and not has_qwen_chunks:
            print(f"\n✨ Video is indexed with X-CLIP, but Qwen embeddings are missing.", flush=True)
            print("  Proceeding to compute Qwen embeddings...", flush=True)
        else:
            print(f"\n✨ This video has ALREADY been processed and indexed! (Found {len(item.chunks)} chunks in DB).", flush=True)
            print("You can search it immediately using: uv run python scripts/search_cli.py \"<your search query>\"")
            summary["status"] = "skipped"
            summary["chunks"] = len(item.chunks)
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
            summary["status"] = "failed"
            summary["error"] = f"MediaItem {item.id} not found in DB."
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
                embedding_model=cfg.DEFAULT_EMBEDDING_MODEL,
                embedding_version=cfg.DEFAULT_EMBEDDING_VERSION,
            )
            session.commit()
            print(f"  ✓ Stored {len(chunks)} chunk records in database.", flush=True)
        else:
            print(f"  ✓ Found {len(chunks)} existing chunks in database.", flush=True)

    summary["chunks"] = len(chunks)
    summary["qwen"] = run_qwen

    # Step 3: Multi-threaded parallel clipping
    need_local_export = bool(args.output_dir)
    need_cloud_upload = bool(args.upload_chunks if args.upload_chunks is not None else cfg.UPLOAD_CHUNKS_TO_STORAGE)

    if need_local_export or need_cloud_upload:
        print(f"\n[3/4] Parallel clipping {len(chunks)} scene(s) with {args.workers} workers...", flush=True)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        clip_tasks: list[tuple[int, float, float, str]] = []
        tmp_files_to_cleanup: list[str] = []

        for idx, chunk in enumerate(chunks):
            if not chunk.end_ts:
                continue
            # If only cloud upload is needed (no local export) and chunk was already uploaded, skip slicing
            if need_cloud_upload and not need_local_export and chunk.storage_path:
                continue

            if args.output_dir:
                clip_out = os.path.join(args.output_dir, f"scene_{idx+1:03d}.mp4")
            else:
                tmp_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                clip_out = tmp_f.name
                tmp_f.close()
                tmp_files_to_cleanup.append(clip_out)

            clip_tasks.append((idx, chunk.start_ts, chunk.end_ts, clip_out))

        try:
            results: list[tuple[int, str, bool]] = []
            if clip_tasks:
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
            else:
                print("  ✓ All required clips are already sliced or uploaded.", flush=True)

            # Handle Cloud/Storage Upload if configured
            if need_cloud_upload:
                results_by_idx = {r[0]: (r[1], r[2]) for r in results}
                with get_db_session(cfg.DATABASE_URL) as session:
                    db_item = session.get(MediaItem, item.id)
                    uploaded_bytes = 0
                    uploaded_count = 0
                    failed_count = 0
                    already_uploaded = sum(1 for c in db_item.chunks if c.storage_path)

                    if already_uploaded > 0:
                        print(f"  → Found {already_uploaded}/{len(db_item.chunks)} chunk(s) already uploaded to storage.", flush=True)
                    print(f"  → Uploading remaining sliced clip(s) to storage backend...", flush=True)

                    for idx, chunk in enumerate(db_item.chunks):
                        if chunk.storage_path:
                            continue

                        if idx not in results_by_idx:
                            continue

                        out_path, success = results_by_idx[idx]
                        if not success or not os.path.exists(out_path):
                            failed_count += 1
                            continue

                        try:
                            with open(out_path, "rb") as cf:
                                chunk_bytes = cf.read()
                            chunk_filename = f"chunks/{db_item.id[:8]}_{chunk.id[:8]}.mp4"
                            size_mb = len(chunk_bytes) / (1024 * 1024)
                            uploaded_bytes += len(chunk_bytes)
                            print(
                                f"    → Uploading chunk {idx + 1}/{len(db_item.chunks)} | {chunk_filename} | "
                                f"{size_mb:.2f} MB | session total {uploaded_bytes / (1024 * 1024):.2f} MB",
                                flush=True,
                            )
                            if size_mb > IMAGEKIT_FREE_PLAN_VIDEO_LIMIT_MB:
                                print(
                                    f"      ⚠ {size_mb:.2f} MB exceeds the "
                                    f"{IMAGEKIT_FREE_PLAN_VIDEO_LIMIT_MB} MB free-plan per-video upload limit.",
                                    flush=True,
                                )

                            saved_path = storage.save_file(chunk_bytes, chunk_filename)
                            chunk.storage_path = saved_path
                            session.commit()  # Incremental commit per chunk so progress is never lost
                            uploaded_count += 1

                        except Exception as upload_err:
                            session.rollback()
                            failed_count += 1
                            print(
                                f"      ❌ Failed to upload chunk {idx + 1}: {upload_err}",
                                flush=True,
                            )
                            err_str = str(upload_err).lower()
                            if any(w in err_str for w in ["quota", "limit", "full", "space", "storage", "out of memory"]):
                                print(
                                    f"\n      ⚠️ Storage quota exceeded or storage full. Aborting further uploads.",
                                    flush=True,
                                )
                                print(
                                    f"      Progress preserved: {uploaded_count + already_uploaded} chunk(s) saved in DB.",
                                    flush=True,
                                )
                                break

                    print(
                        f"  ✓ Sliced clips upload summary: {uploaded_count} uploaded, "
                        f"{already_uploaded} previously uploaded, {failed_count} failed/skipped.",
                        flush=True,
                    )
                    summary["uploaded"] = uploaded_count + already_uploaded
                    summary["upload_failed"] = failed_count

        finally:
            # Cleanup temporary files if created for cloud-only upload
            for tmp_p in tmp_files_to_cleanup:
                try:
                    if os.path.exists(tmp_p):
                        os.remove(tmp_p)
                except OSError:
                    pass
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
            print("  ✓ X-CLIP indexing completed successfully.", flush=True)

            # Step 4b: Optional Qwen Embedding
            if run_qwen:
                try:
                    from footage_engine.embeddings import collection_name_for, get_embedder
                    from scripts.backfill_qwen import process_item as process_qwen_item

                    qwen_collection = collection_name_for(cfg, "qwen")
                    print(
                        f"\n  → Computing Qwen embeddings ({cfg.QWEN_MODEL_NAME}) "
                        f"& indexing into collection '{qwen_collection}'...",
                        flush=True,
                    )
                    t_qwen_start = time.time()
                    qwen_embedder = get_embedder(cfg, backend="qwen")
                    qwen_vec_store = get_vector_store(cfg, backend="qwen")

                    with get_db_session(cfg.DATABASE_URL) as session:
                        db_item = session.get(MediaItem, item.id)
                        already_has_qwen = any(
                            c.embedding_model == qwen_embedder.model_name
                            for c in (db_item.chunks if db_item else [])
                        )

                    if already_has_qwen:
                        print("  ✓ Qwen chunks already exist in DB for this item. Skipping Qwen embedding.", flush=True)
                    else:
                        n_qwen = process_qwen_item(
                            item_id=item.id,
                            database_url=cfg.DATABASE_URL,
                            xclip_model=cfg.DEFAULT_EMBEDDING_MODEL,
                            embedder=qwen_embedder,
                            storage=storage,
                            vector_store=qwen_vec_store,
                            collection=qwen_collection,
                            dry_run=False,
                        )
                        print(f"  ✓ Indexed {n_qwen} Qwen chunk(s) in {time.time() - t_qwen_start:.1f}s.", flush=True)
                except Exception as qwen_err:
                    print(
                        f"  ⚠️ Qwen embedding failed: {qwen_err} "
                        f"(X-CLIP embeddings remain intact and searchable).",
                        flush=True,
                    )

            elapsed = time.time() - start_time
            print("\n" + "=" * 80, flush=True)
            print(f"✨ Successfully indexed with TransNetV2 in {elapsed:.1f}s!", flush=True)
            print("=" * 80, flush=True)
            print("You can now search inside this video with semantic queries:")
            print('  uv run python scripts/search_cli.py "describe any scene in the video"')
            print("=" * 80, flush=True)
            summary["status"] = "success"
        else:
            print("\n❌ Embedding/indexing failed. Check logs above for details.", flush=True)
            summary["status"] = "failed"
            summary["error"] = "Embedding/indexing failed (see logs)."
    else:
        print("\n✨ Clipping complete. Vector indexing skipped as requested (--skip-index).")
        summary["status"] = "success"

    summary["elapsed"] = time.time() - summary["_t0"]


if __name__ == "__main__":
    main()
