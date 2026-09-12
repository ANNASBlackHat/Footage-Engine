"""Run the Qwen backfill on Kaggle (clone repo, run script, done).

Kaggle setup (one time, ~2 minutes):
  1. New Notebook, enable **Internet** (right panel) and **GPU** accelerator
     (P100/T4). T4 (~2x) is fastest if available.
  2. Add-ons -> Secrets, attach these names (values from your local `.env`):
       DATABASE_URL, VECTOR_STORE, STORAGE_BACKEND,
       ZILLIZ_URI, ZILLIZ_TOKEN, ZILLIZ_COLLECTION_NAME,
       QWEN_ZILLIZ_URI, QWEN_ZILLIZ_TOKEN, QWEN_ZILLIZ_COLLECTION_NAME,
       IMAGEKIT_PUBLIC_KEY, IMAGEKIT_PRIVATE_KEY, IMAGEKIT_URL_ENDPOINT
     (Kaggle exposes attached secrets as environment variables.)
  3. Push this repo first so the clone has the latest code:
       git push origin main
  4. In a notebook cell (or terminal):
       !git clone https://github.com/ANNASBlackHat/Footage-Engine.git
       %cd Footage-Engine
       !python scripts/kaggle_backfill.py --limit 100
     For a private repo use a token URL instead:
       !git clone https://<GITHUB_TOKEN>@github.com/ANNASBlackHat/Footage-Engine.git

Notes:
  - The run continues server-side if you close the browser (within Kaggle's
    per-session time limits). If it stops early, just re-run: finished items
    are skipped, so batches resume where the last run ended.
  - Start with `--dry-run --limit 5` to validate secrets/connectivity fast.

Usage:
    python scripts/kaggle_backfill.py [--limit N] [--dry-run] [--media-ids A,B] [--skip-deps]
"""

import argparse
import os
import subprocess
import sys

REQUIRED_ENV = [
    "DATABASE_URL",
    "VECTOR_STORE",
    "ZILLIZ_URI",
    "ZILLIZ_TOKEN",
    "ZILLIZ_COLLECTION_NAME",
    "QWEN_ZILLIZ_URI",
    "QWEN_ZILLIZ_TOKEN",
    "QWEN_ZILLIZ_COLLECTION_NAME",
]

# (module, pip-spec): installed only when the import fails (Kaggle pre-ships torch/cv2).
DEP_GROUPS = [
    ("torch", "torch"),
    ("transformers", "transformers>=4.57"),
    ("sentence_transformers", "sentence-transformers"),
    ("sqlalchemy", "sqlalchemy"),
    ("pydantic_settings", "pydantic-settings"),
    ("pymilvus", "pymilvus"),
    ("psycopg2", "psycopg2-binary"),
    ("imagekitio", "imagekitio"),
    ("PIL", "pillow"),
]


def check_env() -> list[str]:
    return [k for k in REQUIRED_ENV if not os.environ.get(k)]


def ensure_deps() -> None:
    missing: list[str] = []
    for module, spec in DEP_GROUPS:
        try:
            __import__(module)
        except ImportError:
            missing.append(spec)
    if not missing:
        print("[kaggle] alldeps present, skipping installs", flush=True)
        return
    print(f"[kaggle] installing: {' '.join(missing)}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *missing])
    print("[kaggle] installs done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen backfill runner for Kaggle.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--media-ids", default=None)
    ap.add_argument("--skip-deps", action="store_true", help="Skip dependency check/install.")
    args = ap.parse_args()

    missing = check_env()
    if missing:
        print(f"[kaggle] ERROR: missing Kaggle secrets (env vars): {', '.join(missing)}", flush=True)
        print("[kaggle] Add-ons -> Secrets -> attach each name, then re-run.", flush=True)
        return 2

    if not args.skip_deps:
        ensure_deps()

    # Make the repo importable when run from its root (clone dir).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import torch

    print(f"[kaggle] cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}",
          flush=True)

    from scripts.backfill_qwen import main as backfill_main

    argv = ["backfill_qwen.py"]
    if args.dry_run:
        argv.append("--dry-run")
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.media_ids:
        argv += ["--media-ids", args.media_ids]
    sys.argv = argv
    return backfill_main()


if __name__ == "__main__":
    sys.exit(main())
