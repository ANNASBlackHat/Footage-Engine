"""Cloudinary managed cloud storage backend.

Two things make Cloudinary different from the Local/ImageKit backends and
shape this module:

* **Every upload must declare a ``resource_type``** (``image`` or ``video``).
  It is inferred from the file extension (falling back to the optional
  ``media_type`` / ``content_type`` hints) and inferred the same way on read,
  so delivery URLs, existence checks and deletes stay symmetric with uploads.
* **``/upload`` rejects payloads over 100 MB with HTTP 413**, so anything
  larger is routed through the chunked ``upload_large`` method. Chunking does
  **not** raise the ceiling: Cloudinary also enforces a per-account maximum
  file size (100 MiB on free plans — confirmed live as "File size too large.
  Got 107454163. Maximum is 104857600."). Sources above that cap cannot be
  stored at all, whatever ``chunk_size`` is used.

Cloudinary's third resource type, ``raw``, is deliberately unused: a raw
asset's id carries its extension, so it cannot be re-derived from a stored
path. Every asset this engine stores is either video or image.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

import requests

from footage_engine.storage._remote_cache import (
    MIN_VALID_FILE_BYTES,
    resolve_remote_reference,
)

try:
    import cloudinary
    import cloudinary.api
    import cloudinary.uploader
except ImportError:  # pragma: no cover - the ImportError path is unit tested
    cloudinary = None  # type: ignore

# Cloudinary's non-chunked upload endpoint fails above 100 MB, so larger
# payloads go through upload_large. Kept below the documented limit for margin.
LARGE_UPLOAD_THRESHOLD_BYTES = 95 * 1024 * 1024
UPLOAD_LARGE_CHUNK_SIZE = 20 * 1024 * 1024

RESOURCE_TYPE_IMAGE = "image"
RESOURCE_TYPE_VIDEO = "video"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".ogv", ".mov", ".mkv", ".avi", ".m4v"})


def infer_resource_type(
    filename: str,
    content_type: str | None = None,
    media_type: str | None = None,
) -> str:
    """Maps an asset to the ``resource_type`` Cloudinary requires.

    A recognised extension always wins, which guarantees the read path
    recomputes the same value the write path used. The ``media_type`` and
    ``content_type`` hints only cover extensions we cannot classify.
    """
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return RESOURCE_TYPE_IMAGE
    if ext in VIDEO_EXTENSIONS:
        return RESOURCE_TYPE_VIDEO

    hint = (media_type or "").strip().lower()
    if hint in (RESOURCE_TYPE_IMAGE, RESOURCE_TYPE_VIDEO):
        return hint

    if content_type:
        major = content_type.split("/")[0].strip().lower()
        if major in (RESOURCE_TYPE_IMAGE, RESOURCE_TYPE_VIDEO):
            return major

    # Matches Settings.DEFAULT_MEDIA_TYPE: an unclassifiable asset is treated
    # as video rather than being rejected.
    return RESOURCE_TYPE_VIDEO


def split_asset_path(filename: str) -> tuple[str, str]:
    """Splits ``folder/name.ext`` into its public id stem and extension.

    The extension keeps its leading dot and is ``''`` when absent.
    """
    clean = filename.replace("\\", "/").strip("/")
    ext = Path(clean).suffix
    stem = clean[: -len(ext)] if ext else clean
    return stem, ext


def public_id_for(storage_path: str) -> str:
    """Cloudinary public id for a stored asset (its path without extension)."""
    clean = storage_path.replace("\\", "/").strip("/")
    stem, _ = split_asset_path(clean)
    return stem


def payload_size(file_data: bytes | BinaryIO | bytearray) -> int | None:
    """Best-effort size of an upload payload, leaving the stream position intact."""
    if isinstance(file_data, (bytes, bytearray)):
        return len(file_data)
    try:
        position = file_data.tell()
        file_data.seek(0, os.SEEK_END)
        size = file_data.tell()
        file_data.seek(position)
        return size
    except Exception:
        return None


def response_value(response: Any, key: str) -> Any:
    """Reads ``key`` from an SDK response that may be a dict or an object."""
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


class CloudinaryStorageBackend:
    """Stores raw footage files in Cloudinary managed object storage."""

    delivery_endpoint = "https://res.cloudinary.com"

    def __init__(
        self,
        cloud_name: str,
        api_key: str,
        api_secret: str,
        folder: str = "footage_engine/raw",
        cache_dir: str | None = None,
    ):
        if cloudinary is None:
            raise ImportError(
                "cloudinary package is required for CloudinaryStorageBackend. "
                "Install with: pip install cloudinary"
            )
        self.cloud_name = cloud_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.folder = folder.strip("/")

        # The Cloudinary SDK holds credentials in module-global config.
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,
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
        stem, ext = split_asset_path(filename)
        resource_type = infer_resource_type(filename, content_type, media_type)
        public_id = f"{self.folder}/{stem}" if self.folder else stem

        size = payload_size(file_data)
        if size is not None and size > LARGE_UPLOAD_THRESHOLD_BYTES:
            response = self._upload_large(file_data, public_id, resource_type, ext=ext)
        else:
            response = self._upload(file_data, public_id, resource_type)

        stored_id = response_value(response, "public_id") or public_id
        fmt = response_value(response, "format") or ext.lstrip(".")
        return f"{stored_id}.{fmt}" if fmt else stored_id

    def _upload(self, file_data: bytes | BinaryIO, public_id: str, resource_type: str) -> Any:
        """Single-request upload, used for payloads under the 100 MB limit."""
        payload = io.BytesIO(file_data) if isinstance(file_data, bytes) else file_data
        return cloudinary.uploader.upload(
            payload,
            public_id=public_id,
            resource_type=resource_type,
            overwrite=True,
            unique_filename=False,
            use_filename=False,
            invalidate=True,
        )

    def _upload_large(
        self,
        file_data: bytes | BinaryIO,
        public_id: str,
        resource_type: str,
        ext: str = "",
    ) -> Any:
        """Chunked upload for payloads over 100 MB.

        ``upload_large`` streams from a path on disk, so the payload is spooled
        to a temporary file which is always removed afterwards.
        """
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext or ".mp4", delete=False) as tmp:
                tmp_path = tmp.name
                if isinstance(file_data, (bytes, bytearray)):
                    tmp.write(file_data)
                else:
                    if hasattr(file_data, "seek"):
                        file_data.seek(0)
                    while True:
                        block = file_data.read(1024 * 1024)
                        if not block:
                            break
                        tmp.write(block)

            return cloudinary.uploader.upload_large(
                tmp_path,
                public_id=public_id,
                resource_type=resource_type,
                overwrite=True,
                unique_filename=False,
                use_filename=False,
                chunk_size=UPLOAD_LARGE_CHUNK_SIZE,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_url(self, storage_path: str) -> str:
        if storage_path.startswith(("http://", "https://", "file://")):
            return storage_path
        resource_type = infer_resource_type(storage_path)
        clean = storage_path.replace("\\", "/").strip("/")
        return f"{self.delivery_endpoint}/{self.cloud_name}/{resource_type}/upload/{clean}"

    def get_file(self, storage_path: str) -> bytes:
        resp = requests.get(self.get_url(storage_path), timeout=60)
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
        if not cached_file.exists() or cached_file.stat().st_size < MIN_VALID_FILE_BYTES:
            cached_file.write_bytes(self.get_file(storage_path))
        return str(cached_file)

    def exists(self, storage_path: str) -> bool:
        resource_type = infer_resource_type(storage_path)
        public_id = public_id_for(storage_path)
        try:
            cloudinary.api.resource(public_id, resource_type=resource_type)
            return True
        except Exception:
            # The admin API raises (rather than returning falsy) when an asset is
            # absent, so absence is reported as False either way.
            return False

    def delete_file(self, storage_path: str) -> bool:
        resource_type = infer_resource_type(storage_path)
        public_id = public_id_for(storage_path)
        response = cloudinary.uploader.destroy(
            public_id,
            resource_type=resource_type,
            invalidate=True,
        )
        return response_value(response, "result") == "ok"


__all__ = [
    "CloudinaryStorageBackend",
    "LARGE_UPLOAD_THRESHOLD_BYTES",
    "infer_resource_type",
    "public_id_for",
    "split_asset_path",
]


