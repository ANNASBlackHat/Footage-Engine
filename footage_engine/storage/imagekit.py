"""ImageKit managed cloud storage backend."""

import os
import tempfile
from pathlib import Path
from typing import BinaryIO
import requests

try:
    from imagekitio import ImageKit
except ImportError:
    ImageKit = None  # type: ignore


class ImageKitStorageBackend:
    """Stores raw footage files in ImageKit managed object storage."""

    def __init__(
        self,
        public_key: str,
        private_key: str,
        url_endpoint: str,
        cache_dir: str | None = None,
    ):
        if ImageKit is None:
            raise ImportError(
                "imagekitio package is required for ImageKitStorageBackend. "
                "Install with: pip install imagekitio"
            )
        self.public_key = public_key
        self.private_key = private_key
        self.url_endpoint = url_endpoint.rstrip("/")
        
        # Instantiate ImageKit client compatible with both v5 and legacy versions
        try:
            self.client = ImageKit(private_key=private_key, timeout=300.0)
        except TypeError:
            self.client = ImageKit(
                public_key=public_key,
                private_key=private_key,
                url_endpoint=url_endpoint,
            )

        self.cache_dir = Path(cache_dir or os.path.join(tempfile.gettempdir(), "footage_engine_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_file(
        self,
        file_data: bytes | BinaryIO,
        filename: str,
        content_type: str | None = None,
    ) -> str:
        folder = "/footage_engine/raw"
        
        # Check if v5 client.files.upload exists
        if hasattr(self.client, "files") and hasattr(self.client.files, "upload"):
            import io
            upload_payload = io.BytesIO(file_data) if isinstance(file_data, bytes) else file_data
            res = self.client.files.upload(
                file=upload_payload,
                file_name=filename,
                folder=folder,
                use_unique_file_name=False,
                overwrite_file=True,
            )
        else:
            # Legacy v3/v4 SDK
            res = self.client.upload_file(
                file=file_data,
                file_name=filename,
                options={
                    "folder": folder,
                    "use_unique_file_name": False,
                    "overwrite_file": True,
                },
            )

        file_path = getattr(res, "file_path", None)
        if not file_path and hasattr(res, "filePath"):
            file_path = res.filePath
        if not file_path and isinstance(res, dict):
            file_path = res.get("filePath") or res.get("file_path")
            
        if not file_path:
            file_path = f"{folder}/{filename}"
        return file_path

    def get_url(self, storage_path: str) -> str:
        if storage_path.startswith(("http://", "https://", "file://")):
            return storage_path
        clean_path = storage_path.lstrip("/")
        return f"{self.url_endpoint}/{clean_path}"

    def get_file(self, storage_path: str) -> bytes:
        url = self.get_url(storage_path)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

    def get_local_path(self, storage_path: str) -> str:
        from footage_engine.sources.youtube import YouTubeAdapter, extract_youtube_video_id, is_youtube_url

        if is_youtube_url(storage_path):
            vid = extract_youtube_video_id(storage_path) or "yt"
            cached_file = self.cache_dir / f"youtube_{vid}.mp4"
            if not cached_file.exists() or cached_file.stat().st_size < 1000:
                adapter = YouTubeAdapter()
                adapter.download_to_path(storage_path, str(cached_file))
            return str(cached_file)

        if storage_path.startswith(("http://", "https://")):
            import time
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
        clean_name = storage_path.replace("/", "_").lstrip("_")
        cached_file = self.cache_dir / clean_name
        if not cached_file.exists():
            data = self.get_file(storage_path)
            with open(cached_file, "wb") as f:
                f.write(data)
        return str(cached_file)

    def exists(self, storage_path: str) -> bool:
        url = self.get_url(storage_path)
        resp = requests.head(url, timeout=10)
        return resp.status_code == 200

    def delete_file(self, storage_path: str) -> bool:
        return True
