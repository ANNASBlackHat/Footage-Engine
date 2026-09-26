"""Run the async search-job worker on a Colab GPU VM from your laptop.

`colab run script.py` alone fails because a fresh VM has neither this repo nor
its dependencies. This driver handles the whole flow:

  1. Bundles repo code (never `.env`) into a tarball + env JSON from settings.
  2. Creates (or reuses) a GPU session, installs deps with visible progress.
  3. Uploads the bundle and runs scripts/run_worker.py with your args.
  4. Stops the VM unless --keep is given.

Usage:
    uv run python scripts/colab_worker.py --backend qwen --idle-exit 900
    uv run python scripts/colab_worker.py --dry-run --backend qwen
    uv run python scripts/colab_worker.py --idle-exit 0 --keep   # run forever
    uv run python scripts/colab_worker.py --bundle-only          # build only
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
"""Install worker deps on the Colab VM with visible progress."""
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
"""Unpack bundle, load env, run the worker with the given argv."""
import json
import os
import runpy
import sys
import tarfile

with tarfile.open("/content/fe_worker.tar.gz", "r:gz") as tf:
    tf.extractall("/content/fe_repo")
sys.path.insert(0, "/content/fe_repo")

with open("/content/fe_env.json") as f:
    os.environ.update({k: str(v) for k, v in json.load(f).items()})

sys.argv = ["run_worker.py", *EXTRA_ARGV]
runpy.run_path("/content/fe_repo/scripts/run_worker.py", run_name="__main__")
print("[run] done", flush=True)
'''


def sh(*cmd: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(list(cmd))


def build_bundle(tmpdir: str) -> tuple[str, str, str]:
    """Create code tarball + env JSON + remote setup script. Returns the paths."""
    from footage_engine.config import get_settings

    tarball = os.path.join(tmpdir, "fe_worker.tar.gz")
    with tarfile.open(tarball, "w:gz") as tf:
        for rel in ("footage_engine", "pyproject.toml", "scripts/run_worker.py"):
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
    ap = argparse.ArgumentParser(description="Run the async search-job worker on a Colab GPU VM.")
    ap.add_argument("--backend", choices=["xclip", "qwen"], default="qwen")
    ap.add_argument("--tasks", default=None, help="Comma-separated task allowlist (default: all).")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--idle-exit", type=int, default=900, dest="idle_exit",
                    help="Exit after N idle seconds; 0 = run forever (default: 900).")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--session", "-s", default="fe-worker")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--keep", action="store_true", help="Leave the VM running afterwards.")
    ap.add_argument("--skip-setup", action="store_true", help="Reuse session, skip pip installs.")
    ap.add_argument("--bundle-only", action="store_true", help="Only build the bundle, no Colab.")
    ap.add_argument("--exec-timeout", type=int, default=3600, help="colab exec timeout in seconds.")
    args = ap.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="fe_colab_worker_")
    tarball, env_path, setup_path = build_bundle(tmpdir)
    print(f"[bundle] {tarball} ({os.path.getsize(tarball)} bytes)")
    if args.bundle_only:
        print(f"[bundle] env keys: {sorted(json.load(open(env_path)).keys())}")
        return 0

    extra_argv: list[str] = ["--backend", args.backend, "--concurrency", str(args.concurrency)]
    if args.tasks:
        extra_argv += ["--tasks", args.tasks]
    if args.max_jobs is not None:
        extra_argv += ["--max-jobs", str(args.max_jobs)]
    if args.idle_exit is not None:
        extra_argv += ["--idle-exit", str(args.idle_exit)]
    if args.dry_run:
        extra_argv.append("--dry-run")

    run_path = os.path.join(tmpdir, "fe_remote_run.py")
    with open(run_path, "w") as f:
        f.write(f"EXTRA_ARGV = {extra_argv!r}\n" + REMOTE_RUN)

    sh("colab", "new", "-s", args.session, "--gpu", args.gpu)
    try:
        if not args.skip_setup:
            sh("colab", "exec", "--timeout", "3600", "-s", args.session, "-f", setup_path)
        sh("colab", "upload", "-s", args.session, tarball, "/content/fe_worker.tar.gz")
        sh("colab", "upload", "-s", args.session, env_path, "/content/fe_env.json")
        sh("colab", "exec", "--timeout", str(args.exec_timeout), "-s", args.session, "-f", run_path)
    finally:
        if not args.keep:
            subprocess.run(["colab", "stop", "-s", args.session])
    return 0


if __name__ == "__main__":
    sys.exit(main())
