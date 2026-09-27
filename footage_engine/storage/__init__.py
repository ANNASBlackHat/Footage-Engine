"""Storage module factory."""

from footage_engine.config import Settings, get_settings
from footage_engine.storage.base import StorageBackend
from footage_engine.storage.cloudinary import CloudinaryStorageBackend
from footage_engine.storage.gdrive import GoogleDriveStorageBackend, split_scopes
from footage_engine.storage.imagekit import ImageKitStorageBackend
from footage_engine.storage.local import LocalStorageBackend

_storage_instance: StorageBackend | None = None

SUPPORTED_BACKENDS = ("local", "imagekit", "cloudinary", "gdrive")


def get_storage_backend(settings: Settings | None = None) -> StorageBackend:
    """Builds the storage backend described by ``settings`` (default: app config).

    Only the app-config backend is memoised. Passing explicit ``settings``
    returns a fresh instance without overwriting the process-wide one, so a
    worker or test configured differently cannot poison other callers.
    """
    global _storage_instance
    if settings is None and _storage_instance is not None:
        return _storage_instance

    cfg = settings or get_settings()

    if cfg.STORAGE_BACKEND == "imagekit":
        if not (cfg.IMAGEKIT_PUBLIC_KEY and cfg.IMAGEKIT_PRIVATE_KEY and cfg.IMAGEKIT_URL_ENDPOINT):
            raise ValueError("IMAGEKIT_PUBLIC_KEY, IMAGEKIT_PRIVATE_KEY, and IMAGEKIT_URL_ENDPOINT must be configured.")
        backend: StorageBackend = ImageKitStorageBackend(
            public_key=cfg.IMAGEKIT_PUBLIC_KEY,
            private_key=cfg.IMAGEKIT_PRIVATE_KEY,
            url_endpoint=cfg.IMAGEKIT_URL_ENDPOINT,
        )
    elif cfg.STORAGE_BACKEND == "cloudinary":
        if not (cfg.CLOUDINARY_CLOUD_NAME and cfg.CLOUDINARY_API_KEY and cfg.CLOUDINARY_API_SECRET):
            raise ValueError(
                "CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET must be configured."
            )
        backend = CloudinaryStorageBackend(
            cloud_name=cfg.CLOUDINARY_CLOUD_NAME,
            api_key=cfg.CLOUDINARY_API_KEY,
            api_secret=cfg.CLOUDINARY_API_SECRET,
            folder=cfg.CLOUDINARY_FOLDER,
        )
    elif cfg.STORAGE_BACKEND == "gdrive":
        if not (cfg.GDRIVE_SERVICE_ACCOUNT_FILE or cfg.GDRIVE_SERVICE_ACCOUNT_JSON):
            raise ValueError(
                "GDRIVE_SERVICE_ACCOUNT_FILE or GDRIVE_SERVICE_ACCOUNT_JSON must be "
                "configured (exactly one)."
            )
        if not (cfg.GDRIVE_FOLDER_ID or cfg.GDRIVE_DRIVE_ID):
            raise ValueError(
                "GDRIVE_FOLDER_ID or GDRIVE_DRIVE_ID must be configured — a service "
                "account has no Drive of its own, so uploads need an explicit target "
                "it has been shared into."
            )
        backend = GoogleDriveStorageBackend(
            credentials_file=cfg.GDRIVE_SERVICE_ACCOUNT_FILE,
            credentials_json=cfg.GDRIVE_SERVICE_ACCOUNT_JSON,
            folder_id=cfg.GDRIVE_FOLDER_ID,
            drive_id=cfg.GDRIVE_DRIVE_ID,
            scopes=split_scopes(cfg.GDRIVE_SCOPES),
            url_template=cfg.GDRIVE_URL_TEMPLATE,
        )
    elif cfg.STORAGE_BACKEND == "local":
        backend = LocalStorageBackend(base_dir=cfg.LOCAL_STORAGE_DIR)
    else:
        # Explicit rather than a silent fallback to local disk: a mistyped or
        # unsupported backend must fail loudly instead of writing to the
        # developer's filesystem.
        raise ValueError(
            f"Unsupported STORAGE_BACKEND: {cfg.STORAGE_BACKEND!r}. "
            f"Expected one of: {', '.join(SUPPORTED_BACKENDS)}."
        )

    if settings is None:
        _storage_instance = backend
    return backend


__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
    "ImageKitStorageBackend",
    "CloudinaryStorageBackend",
    "GoogleDriveStorageBackend",
    "SUPPORTED_BACKENDS",
    "get_storage_backend",
]
