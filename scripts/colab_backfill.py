"""Run the Qwen backfill on a Colab GPU VM from your laptop.

`colab run script.py` alone fails because a fresh VM has neither this repo
nor its dependencies. This driver handles the whole flow:

  1. Bundles repo code (never `.env`) into a tarball + env JSON from settings.
  2. Creates (or reuses) a GPU session, installs deps with visible progress.
  3. Uploads the bundle, runs scripts/backfill_qwen.py with your args.
  4. Stops the VM unless --keep is given.

Usage:
    uv run python scripts/colab_backfill.py --limit 20
    uv run python scripts/colab_backfill.py --dry-run --limit 2
    uv run python scripts/colab_backfill.py --limit 100 --keep
    uv run python scripts/colab_backfill.py --bundle-only   # build only, no Colab
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REMOTE_SETUP = '''
"""Install backfill deps on the Colab VM with visible progress."""
import subprocess
import sys

groups = [
    ["torch"],
    ["transformers>=4.57", "sentence-transformers", "pillow", "numpy"],
    ["sqlalchemy", "pydantic", "pydantic-settings", "python-dotenv"],
    ["pymilvus", "psycopg2-binary", "imagekitio", "requests", "opencv-python-headless"],
]
for group in groups:
    print(f"[setup] pip install {' '.join(group)} ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *group])
print("[setup] done", flush=True)
'''

REMOTE_RUN = '''
"""Unpack bundle, load env, run backfill with the given argv."""
import json
import os
import runpy
import sys
import tarfile

with tarfile.open("/content/fe_backfill.tar.gz", "r:gz") as tf:
    tf.extractall("/content/fe_repo")
sys.path.insert(0, "/content/fe_repo")

with open("/content/fe_env.json") as f:
    os.environ.update({k: str(v) for k, v in json.load(f).items()})

sys.argv = ["backfill_qwen.py", *EXTRA_ARGV]
runpy.run_path("/content/fe_repo/scripts/backfill_qwen.py", run_name="__main__")
print("[run] done", flush=True)
'''


def sh(*cmd: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(list(cmd))


def build_bundle(tmpdir: str) -> tuple[str, str]:
    """Create code tarball + env JSON from local settings. Returns both paths."""
    from footage_engine.config import get_settings

    tarball = os.path.join(tmpdir, "fe_backfill.tar.gz")
    with tarfile.open(tarball, "w:gz") as tf:
        for rel in ("footage_engine", "pyproject.toml", "scripts/backfill_qwen.py"):
            full = os.path.join(REPO_ROOT, rel)
            tf.add(full, arcname=rel,
                   filter=lambda ti: None if "__pycache__" in ti.name else ti)

    settings = get_settings()
    env = {k: v for k, v in settings.model_dump().items() if v is not None}
    env_path = os.path.join(tmpdir, "fe_env.json")
    with open(env_path, "w") as f:
        json.dump(env, f)

    setup_path = os.path.join(tmpdir, "fe_remote_setup.py")
    with open(setup_path, "w") as f:
        f.write(REMOTE_SETUP)
    return tarball, env_path, setup_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Qwen backfill on a Colab GPU VM.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--media-ids", default=None)
    ap.add_argument("--session", "-s", default="embed-backfill")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--keep", action="store_true", help="Leave the VM running afterwards.")
    ap.add_argument("--skip-setup", action="store_true", help="Reuse session, skip pip installs.")
    ap.add_argument("--bundle-only", action="store_true", help="Only build the bundle, no Colab.")
    args = ap.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="fe_colab_")
    tarball, env_path, setup_path = build_bundle(tmpdir)
    print(f"[bundle] {tarball} ({os.path.getsize(tarball)} bytes)")
    if args.bundle_only:
        print(f"[bundle] env keys: {sorted(json.load(open(env_path)).keys())}")
        return 0

    run_path = os.path.join(tmpdir, "fe_remote_run.py")
    extra_argv: list[str] = []
    if args.dry_run:
        extra_argv.append("--dry-run")
    if args.limit is not None:
        extra_argv += ["--limit", str(args.limit)]
    if args.media_ids:
        extra_argv += ["--media-ids", args.media_ids]
    with open(run_path, "w") as f:
        f.write(f"EXTRA_ARGV = {extra_argv!r}\n" + REMOTE_RUN)

    sh("colab", "new", "-s", args.session, "--gpu", args.gpu)
    try:
        if not args.skip_setup:
            sh("colab", "exec", "--timeout", "3600", "-s", args.session, "-f", setup_path)
        sh("colab", "upload", "-s", args.session, tarball, "/content/fe_backfill.tar.gz")
        sh("colab", "upload", "-s", args.session, env_path, "/content/fe_env.json")
        sh("colab", "exec", "--timeout", "3600", "-s", args.session, "-f", run_path)
    finally:
        if not args.keep:
            subprocess.run(["colab", "stop", "-s", args.session])
    return 0


if __name__ == "__main__":
    sys.exit(main())
