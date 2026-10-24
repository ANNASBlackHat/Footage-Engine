"""Print Google Drive storage quota for the configured credential identity.

Works whenever ``GDRIVE_SERVICE_ACCOUNT_*`` or ``GDRIVE_OAUTH_CREDENTIALS_*`` is
set, independent of ``STORAGE_BACKEND``. Reads ``.env`` via the app settings.

Usage:
    python scripts/gdrive_quota.py
    python scripts/gdrive_quota.py --json
"""

import argparse
import json
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from footage_engine.config import get_settings
from footage_engine.storage.gdrive import format_bytes, get_storage_quota


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show Google Drive total, used, and free storage for the configured credentials."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw quota as JSON (all values in bytes).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = get_settings()

    try:
        quota = get_storage_quota(cfg)
    except Exception as err:
        print(f"❌ Could not read Drive quota: {type(err).__name__}: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(quota, indent=2))
        return 0

    identity = "user OAuth" if quota["identity"] == "user_oauth" else "service account"
    print("Google Drive storage")
    print(f"  Identity          : {identity}")

    if quota["limit"] is None:
        print("  Total             : not reported (unlimited or service account)")
        print(f"  Used (all Google) : {format_bytes(quota['usage'])}")
    else:
        pct = (quota["usage"] / quota["limit"] * 100) if quota["limit"] else 0.0
        print(f"  Total             : {format_bytes(quota['limit'])}")
        print(f"  Used (all Google) : {format_bytes(quota['usage'])} ({pct:.1f}%)")
        print(f"  Free              : {format_bytes(quota['free'])} ({100 - pct:.1f}% available)")

    print(f"  Used in Drive     : {format_bytes(quota['usage_in_drive'])}")
    print(f"  Drive trash       : {format_bytes(quota['usage_in_drive_trash'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
