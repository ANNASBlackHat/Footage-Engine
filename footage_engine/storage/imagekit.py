"""ImageKit managed cloud storage backend."""

import os
import tempfile
from pathlib import Path
from typing import BinaryIO
import requests

from footage_engine.storage._remote_cache import resolve_remote_reference

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
        media_type: str | None = None,
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
        # YouTube links, plain HTTP URLs and ``file://`` references are handled
        # by the shared resolver; anything left is an asset this backend owns,
        # so it is downloaded into the local cache on first use.
        resolved = resolve_remote_reference(storage_path, self.cache_dir)
        if resolved is not None:
            return resolved

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
