"""Storage module factory."""

from footage_engine.config import Settings, get_settings
from footage_engine.storage.base import StorageBackend
from footage_engine.storage.local import LocalStorageBackend
from footage_engine.storage.imagekit import ImageKitStorageBackend

_storage_instance: StorageBackend | None = None


def get_storage_backend(settings: Settings | None = None) -> StorageBackend:
    global _storage_instance
    if _storage_instance is not None and settings is None:
        return _storage_instance

    cfg = settings or get_settings()
    if cfg.STORAGE_BACKEND == "imagekit":
        if not (cfg.IMAGEKIT_PUBLIC_KEY and cfg.IMAGEKIT_PRIVATE_KEY and cfg.IMAGEKIT_URL_ENDPOINT):
            raise ValueError("IMAGEKIT_PUBLIC_KEY, IMAGEKIT_PRIVATE_KEY, and IMAGEKIT_URL_ENDPOINT must be configured.")
        _storage_instance = ImageKitStorageBackend(
            public_key=cfg.IMAGEKIT_PUBLIC_KEY,
            private_key=cfg.IMAGEKIT_PRIVATE_KEY,
            url_endpoint=cfg.IMAGEKIT_URL_ENDPOINT,
        )
    else:
        _storage_instance = LocalStorageBackend(base_dir=cfg.LOCAL_STORAGE_DIR)

    return _storage_instance


__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
    "ImageKitStorageBackend",
    "get_storage_backend",
]
