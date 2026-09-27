"""Local disk storage backend."""

from pathlib import Path
from typing import BinaryIO

from footage_engine.storage._remote_cache import resolve_remote_reference


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
        media_type: str | None = None,
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
        # YouTube links, plain HTTP URLs and ``file://`` references are handled
        # by the shared resolver; anything left is one of our own local files.
        resolved = resolve_remote_reference(storage_path, self.cache_dir)
        if resolved is not None:
            return resolved

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
