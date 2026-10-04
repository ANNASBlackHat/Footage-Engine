"""Bundle .env into a private Kaggle dataset so kernels can read it without UI secrets.

Why this exists: `kaggle kernels push` cannot attach secrets. Kaggle secrets must
be added by hand in the notebook UI (Add-ons -> Secrets), which is fine for one
kernel and unworkable when the secret list is fourteen entries and changes often.
A **private** dataset is the only CLI-driven alternative — `kaggle datasets create`
is private unless `-u` is passed.

SECURITY: this is a downgrade from UI secrets, not an equivalent. Kaggle masks
secret *values* in notebook logs; dataset file contents are ordinary files that
can be printed, committed by a later notebook version, or downloaded by anyone
the dataset is shared with. Mitigations applied here:

  * the generated file is gitignored, so it cannot be committed by accident
  * the dataset is created private
  * only the keys needed on Kaggle are exported, not the whole local .env
  * the notebook never prints values, only key names

Treat the dataset as a credential store: share it with nobody, and rotate
anything that ever lands in a public one.

Usage:
    python scripts/build_kaggle_dataset.py                    # build from ./.env
    python scripts/build_kaggle_dataset.py --set KEY=VALUE    # add/override
    python scripts/build_kaggle_dataset.py --from-file other.env
    python scripts/build_kaggle_dataset.py --show             # key names only
    kaggle datasets create -p kaggle/dataset                 # then upload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "kaggle" / "dataset"
OUTPUT_NAME = "runtime.env"

# Everything the yt_ingest kernel reads. Keeping the allowlist explicit means a
# stray key in a local .env (a Pexels token, a Cloudinary secret) is not shipped
# to a third party just because it sits in the same file.
REQUIRED_KEYS = [
    "DATABASE_URL",
    "VECTOR_STORE",
    "ZILLIZ_URI",
    "ZILLIZ_TOKEN",
    "ZILLIZ_COLLECTION_NAME",
    "QWEN_ZILLIZ_URI",
    "QWEN_ZILLIZ_TOKEN",
    "QWEN_ZILLIZ_COLLECTION_NAME",
    "STORAGE_BACKEND",
    "GDRIVE_OAUTH_CREDENTIALS_JSON",
    "GDRIVE_FOLDER_ID",
    "UPLOAD_RAW_TO_STORAGE",
    "UPLOAD_CHUNKS_TO_STORAGE",
    "YT_COOKIES_URL",
    "INGEST_URLS",
]

# Written even when absent in the source .env, so the kernel's own defaults do
# not silently differ from local behaviour.
DEFAULTS = {
    "EMBEDDING_BACKEND": "xclip",
    "EMBEDDING_DEVICE": "auto",
    "NUM_WORKERS": "2",
    "GDRIVE_SCOPES": "https://www.googleapis.com/auth/drive",
    "GDRIVE_URL_TEMPLATE": "https://lh3.googleusercontent.com/d/{file_id}",
}


def parse_env(path: Path) -> dict[str, str]:
    """Reads KEY=VALUE lines, ignoring comments and blanks."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip one layer of matching quotes; pydantic-settings handles the rest.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def build(source: Path, overrides: list[str]) -> tuple[dict[str, str], list[str]]:
    """Returns the exported values plus the list of required keys left empty.

    Export set = the allowlist, plus any key passed explicitly with --set, plus
    DEFAULTS. It is deliberately *not* "everything in the source .env": a local
    .env typically also holds Pexels, Coverr, ImageKit and Cloudinary
    credentials that the kernel has no use for and that should not be shipped
    to a third party.

    --set keys are honoured even when absent from REQUIRED_KEYS — restricting
    the output to the allowlist silently discarded overrides such as
    EMBEDDING_BACKEND, which made a qwen run quietly fall back to xclip.
    """
    values = parse_env(source)

    explicit: list[str] = []
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        key = key.strip()
        values[key] = value.strip()
        explicit.append(key)

    for key, value in DEFAULTS.items():
        values.setdefault(key, value)

    missing = [key for key in REQUIRED_KEYS if not values.get(key)]

    keys = list(REQUIRED_KEYS) + [k for k in explicit if k not in REQUIRED_KEYS]
    keys += [k for k in DEFAULTS if k not in keys]
    return {key: values.get(key, "") for key in keys}, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from-file", default=".env", help="Source env file (default: .env)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add or override a key. Repeatable.",
    )
    parser.add_argument("--show", action="store_true", help="Print key names and nothing else.")
    args = parser.parse_args(argv)

    source = Path(args.from_file).expanduser()
    if not source.is_file():
        raise SystemExit(f"source env not found: {source}")

    try:
        values, missing = build(source, args.set)
    except SystemExit as err:
        print(str(err), file=sys.stderr)
        return 1

    if missing:
        print("missing keys (supply with --set KEY=VALUE):", file=sys.stderr)
        for key in missing:
            print(f"  {key}", file=sys.stderr)
        return 1

    if args.show:
        for key in values:
            print(key)
        return 0

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASET_DIR / OUTPUT_NAME
    out.write_text(
        "# Generated by scripts/build_kaggle_dataset.py — DO NOT COMMIT.\n"
        "# Uploaded to a PRIVATE Kaggle dataset for kernel consumption.\n"
        + "".join(f"{key}={values[key]}\n" for key in sorted(values)),
        encoding="utf-8",
    )
    out.chmod(0o600)
    try:
        shown = out.relative_to(Path.cwd())
    except ValueError:
        shown = out
    print(f"wrote {out} ({len(values)} keys, {out.stat().st_size} bytes)")
    print("gitignored:", "yes" if "kaggle/dataset/runtime.env" in
          (source.parent / ".gitignore").read_text(encoding="utf-8") else "NO — do not commit")
    print()
    print("next:")
    print(f"  kaggle datasets create -p {shown.parent}")
    print("  # or, to update an existing dataset:")
    print(f"  kaggle datasets version -p {shown.parent} -m 'refresh secrets'")
    return 0


if __name__ == "__main__":
    sys.exit(main())