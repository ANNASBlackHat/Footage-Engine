"""Storage module factory."""

from footage_engine.config import Settings, get_settings
from footage_engine.storage.base import StorageBackend
from footage_engine.storage.cloudinary import CloudinaryStorageBackend
from footage_engine.storage.gdrive import GoogleDriveStorageBackend, split_scopes
from footage_engine.storage.imagekit import ImageKitStorageBackend
from footage_engine.storage.local import LocalStorageBackend

_storage_instance: StorageBackend | None = None

SUPPORTED_BACKENDS = ("local", "imagekit", "cloudinary", "gdrive")

# Reference shapes that are already absolute and must never be prefixed with an
# endpoint. Mirrors the passthrough each backend's ``get_url`` performs.
ABSOLUTE_REF_PREFIXES = ("http://", "https://", "file://")


KEY_BACKED_BACKENDS = (GoogleDriveStorageBackend, LocalStorageBackend)


def to_stored_reference(backend: StorageBackend, key: str) -> str:
    """Returns the most durable reference to store for a freshly-saved asset.

    ``save_file`` returns a backend-owned key that is only meaningful to the
    backend that issued it — and, for ImageKit and Cloudinary, to the *account*
    that issued it. Rotating to a free-tier account on a different endpoint
    makes every previously-stored key unreachable, with nothing in the row to
    record which account wrote it. Storing the absolute URL pins the asset to
    the endpoint that actually holds the bytes, so a later backend change can
    still resolve it.

    Two backends are deliberately left on their key:

    ``GoogleDriveStorageBackend``
        A Drive file id is already account-independent, and ``get_url`` returns
        an unauthenticated ``uc?export=download`` link. Persisting that would
        route every read through a plain HTTP download instead of the
        authenticated ``MediaIoBaseDownload`` path, which is the only way to
        read a file whose public grant was refused -- and it hits Drive's
        virus-scan confirmation page past roughly 100 MB.

    ``LocalStorageBackend``
        The key is a path relative to ``LOCAL_STORAGE_DIR``, which stays valid
        wherever the repo is checked out. An absolute ``file://`` would instead
        bake the writing machine's filesystem layout into a shared database.

    Reads of both shapes already work: ``resolve_remote_reference`` dispatches
    absolute URLs (including YouTube and ``file://``) and returns ``None`` for
    backend keys, and every backend calls it first in ``get_local_path``.
    """
    if not key or key.startswith(ABSOLUTE_REF_PREFIXES):
        return key
    if isinstance(backend, KEY_BACKED_BACKENDS):
        return key
    return backend.get_url(key)


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
        service_account_configured = bool(
            cfg.GDRIVE_SERVICE_ACCOUNT_FILE or cfg.GDRIVE_SERVICE_ACCOUNT_JSON
        )
        oauth_configured = bool(
            cfg.GDRIVE_OAUTH_CREDENTIALS_FILE or cfg.GDRIVE_OAUTH_CREDENTIALS_JSON
        )
        if service_account_configured == oauth_configured:
            # Equal covers both "neither" and "both": one identity is required.
            raise ValueError(
                "Configure exactly one Google Drive credential identity — either "
                "GDRIVE_SERVICE_ACCOUNT_FILE/JSON (service account, Shared Drives only) "
                "or GDRIVE_OAUTH_CREDENTIALS_FILE/JSON (user OAuth, owns the files and "
                f"uses that account's quota); {'both were set' if oauth_configured else 'neither was set'}."
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
            oauth_credentials_file=cfg.GDRIVE_OAUTH_CREDENTIALS_FILE,
            oauth_credentials_json=cfg.GDRIVE_OAUTH_CREDENTIALS_JSON,
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
    "ABSOLUTE_REF_PREFIXES",
    "KEY_BACKED_BACKENDS",
    "get_storage_backend",
    "to_stored_reference",
]
