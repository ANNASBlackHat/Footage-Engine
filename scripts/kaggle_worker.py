"""Run the async search-job worker on Kaggle (clone repo, run, done).

Kaggle setup (one time, ~2 minutes):
  1. New Notebook, enable **Internet** (right panel) and **GPU** accelerator
     (P100/T4). T4 (~2x) is fastest if available.
  2. Add-ons -> Secrets, attach these names (values from your local `.env`):
       DATABASE_URL, VECTOR_STORE, STORAGE_BACKEND,
       ZILLIZ_URI, ZILLIZ_TOKEN, ZILLIZ_COLLECTION_NAME,
       QWEN_ZILLIZ_URI, QWEN_ZILLIZ_TOKEN, QWEN_ZILLIZ_COLLECTION_NAME
     Optional, only needed for script-beat search (Multi-Query + LLM judge):
       GEMINI_API_KEY (or LLM_API_KEY)
     (Kaggle exposes attached secrets as environment variables.)
  3. Push this repo first so the clone has the latest code:
       git push origin main
  4. In a notebook cell (or terminal):
       !git clone https://github.com/ANNASBlackHat/Footage-Engine.git
       %cd Footage-Engine
       !python scripts/kaggle_worker.py --backend qwen --idle-exit 900
     For a private repo use a token URL instead:
       !git clone https://<GITHUB_TOKEN>@github.com/ANNASBlackHat/Footage-Engine.git

Notes:
  - Callers submit jobs from anywhere (see scripts/submit_job.py); this process
    only drains the shared queue, so it is safe to run next to a laptop worker.
  - A remote worker is only meaningful with VECTOR_STORE=zilliz. With the
    default in_memory store every process has a private index and finds nothing.
  - --idle-exit keeps the session from burning quota once the queue is drained.

Usage:
    python scripts/kaggle_worker.py [--backend qwen|xclip] [--tasks A,B]
        [--concurrency N] [--max-jobs N] [--idle-exit SEC] [--dry-run] [--skip-deps]
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

# Only required for search_script_beat (Multi-Query expansion + LLM judge).
OPTIONAL_ENV = ["GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"]

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
    ("cv2", "opencv-python-headless"),  # fine_localize_clip frame sampling
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
        print("[kaggle] all deps present, skipping installs", flush=True)
        return
    print(f"[kaggle] installing: {' '.join(missing)}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *missing])
    print("[kaggle] installs done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Async search-job worker runner for Kaggle.")
    ap.add_argument("--backend", choices=["xclip", "qwen"], default="qwen")
    ap.add_argument("--tasks", default=None, help="Comma-separated task allowlist (default: all).")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--idle-exit", type=int, default=900, dest="idle_exit",
                    help="Exit after N idle seconds (default: 900).")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--skip-deps", action="store_true", dest="skip_deps")
    args, _unknown = ap.parse_known_args()

    missing = check_env()
    if missing:
        print(f"[kaggle] ERROR: missing Kaggle secrets (env vars): {', '.join(missing)}", flush=True)
        print("[kaggle] Add-ons -> Secrets -> attach each name, then re-run.", flush=True)
        return 2

    if not any(os.environ.get(k) for k in OPTIONAL_ENV):
        print("[kaggle] NOTE: no LLM key found; search_script_beat will skip "
              "Multi-Query expansion/reranking.", flush=True)

    if not args.skip_deps and not args.dry_run:
        ensure_deps()

    # Make the repo importable when run from its root (clone dir).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import torch

        print(f"[kaggle] cuda={torch.cuda.is_available()} "
              f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}",
              flush=True)
    except ImportError:
        print("[kaggle] torch unavailable; worker will run on CPU", flush=True)

    from footage_engine.worker.runner import main as worker_main

    argv = ["run_worker.py", "--backend", args.backend, "--concurrency", str(args.concurrency)]
    if args.tasks:
        argv += ["--tasks", args.tasks]
    if args.max_jobs is not None:
        argv += ["--max-jobs", str(args.max_jobs)]
    if args.idle_exit is not None:
        argv += ["--idle-exit", str(args.idle_exit)]
    if args.dry_run:
        argv.append("--dry-run")
    sys.argv = argv
    return worker_main()


if __name__ == "__main__":
    sys.exit(main())
