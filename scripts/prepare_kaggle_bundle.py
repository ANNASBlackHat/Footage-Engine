#!/usr/bin/env python3
"""Package the Remotion frontend, documentary components, and timeline into a portable bundle for Kaggle rendering."""

import os
import tarfile
from pathlib import Path

FRONTEND_DIR = Path("/Users/annasblackhat/Documents/Experiment/video-generation-frontend")
TIMELINE_SRC = Path("/Users/annasblackhat/Documents/Experiment/content-expert/projects/1.-The-Giant-Squid-That-San-a-Ship/timeline.json")
OUTPUT_DIR = Path("/Users/annasblackhat/Documents/Experiment/content-expert/projects/1.-The-Giant-Squid-That-San-a-Ship")
BUNDLE_FILE = OUTPUT_DIR / "kaggle_render_bundle.tar.gz"

INCLUDED_PATHS = [
    "src",
    "scripts",
    "package.json",
    "tsconfig.json",
    "tailwind.config.ts",
    "postcss.config.js",
]


def create_kaggle_bundle():
    print(f"Creating portable Kaggle render bundle: {BUNDLE_FILE}")

    # Copy latest timeline into frontend directory temporarily so it's packaged inside
    temp_timeline = FRONTEND_DIR / "timeline.json"
    with open(TIMELINE_SRC, "r", encoding="utf-8") as f_in, open(temp_timeline, "w", encoding="utf-8") as f_out:
        f_out.write(f_in.read())

    with tarfile.open(BUNDLE_FILE, "w:gz") as tar:
        for p in INCLUDED_PATHS:
            full_path = FRONTEND_DIR / p
            if full_path.exists():
                tar.add(full_path, arcname=p)
                print(f"  + Added: {p}")
        tar.add(temp_timeline, arcname="timeline.json")
        print("  + Added: timeline.json")

    # Clean up temp file
    if temp_timeline.exists():
        temp_timeline.unlink()

    size_mb = BUNDLE_FILE.stat().st_size / (1024 * 1024)
    print(f"\n✅ Bundle created successfully! Size: {size_mb:.2f} MB")
    print(f"File location: {BUNDLE_FILE}")


if __name__ == "__main__":
    create_kaggle_bundle()
