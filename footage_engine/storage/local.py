"""Local disk storage backend."""

import os
from pathlib import Path
from typing import BinaryIO


class LocalStorageBackend:
    """Stores raw footage files directly on the local filesystem."""

    def __init__(self, base_dir: str = "./data/storage", cache_dir: str | None = None):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, storage_path: str) -> Path:
        # Sanitize path to prevent directory traversal
        clean_path = storage_path.lstrip("/\\")
        return self.base_dir / clean_path

    def save_file(
        self,
        file_data: bytes | BinaryIO,
        filename: str,
        content_type: str | None = None,
    ) -> str:
        target_path = self._resolve_path(filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(file_data, bytes):
            with open(target_path, "wb") as f:
                f.write(file_data)
        else:
            with open(target_path, "wb") as f:
                f.write(file_data.read())

        # Return relative storage path
        return filename

    def get_file(self, storage_path: str) -> bytes:
        file_path = self._resolve_path(storage_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found in storage: {storage_path}")
        with open(file_path, "rb") as f:
            return f.read()

    def get_local_path(self, storage_path: str) -> str:
        import requests
        import time
        from footage_engine.sources.youtube import YouTubeAdapter, extract_youtube_video_id, is_youtube_url

        if is_youtube_url(storage_path):
            vid = extract_youtube_video_id(storage_path) or "yt"
            cached_file = self.cache_dir / f"youtube_{vid}.mp4"
            if not cached_file.exists() or cached_file.stat().st_size < 1000:
                adapter = YouTubeAdapter()
                adapter.download_to_path(storage_path, str(cached_file))
            return str(cached_file)

        if storage_path.startswith(("http://", "https://")):
            clean_name = storage_path.split("?")[0].split("/")[-1]
            if not clean_name.endswith((".mp4", ".webm", ".ogv", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".webp")):
                clean_name += ".mp4"
            cached_file = self.cache_dir / clean_name
            if not cached_file.exists() or cached_file.stat().st_size < 1000:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://commons.wikimedia.org/",
                    "Accept": "*/*"
                }
                tmp_file = cached_file.with_suffix(cached_file.suffix + ".tmp")
                success = False
                for attempt in range(5):
                    try:
                        resp = requests.get(storage_path, headers=headers, stream=True, timeout=60)
                        if resp.status_code == 429:
                            time.sleep(4 * (attempt + 1))
                            continue
                        resp.raise_for_status()
                        ctype = resp.headers.get("content-type", "").lower()
                        if "text/html" in ctype or "text/plain" in ctype:
                            time.sleep(3 * (attempt + 1))
                            continue
                        with open(tmp_file, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                        if tmp_file.stat().st_size > 1000:
                            tmp_file.replace(cached_file)
                            success = True
                            break
                    except Exception as e:
                        if attempt == 4:
                            if tmp_file.exists():
                                tmp_file.unlink()
                            raise e
                        time.sleep(3 * (attempt + 1))
                if not success and tmp_file.exists():
                    tmp_file.unlink()
            return str(cached_file)
        if storage_path.startswith("file://"):
            return storage_path[7:]
        file_path = self._resolve_path(storage_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found in storage: {storage_path}")
        return str(file_path)

    def get_url(self, storage_path: str) -> str:
        if storage_path.startswith(("http://", "https://", "file://")):
            return storage_path
        file_path = self._resolve_path(storage_path)
        return f"file://{file_path}"

    def exists(self, storage_path: str) -> bool:
        return self._resolve_path(storage_path).exists()

    def delete_file(self, storage_path: str) -> bool:
        file_path = self._resolve_path(storage_path)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
