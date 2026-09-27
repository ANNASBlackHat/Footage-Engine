"""Storage backend abstraction protocol."""

from typing import BinaryIO, Protocol


class StorageBackend(Protocol):
    """Interface for raw footage storage backends (Local, ImageKit, Cloudinary, S3, etc.)."""

    def save_file(
        self,
        file_data: bytes | BinaryIO,
        filename: str,
        content_type: str | None = None,
        media_type: str | None = None,
    ) -> str:
        """Saves file to storage and returns the unique storage path/key.

        The returned storage path is an *opaque key*: callers must only ever
        hand it back to this backend, never parse or interpret it.

        ``content_type`` and ``media_type`` are optional hints (e.g.
        ``"video/mp4"`` / ``"video"``) used by backends that must classify an
        asset at write time. Backends are free to ignore them and infer from
        ``filename`` instead.
        """
        ...

    def get_file(self, storage_path: str) -> bytes:
        """Retrieves raw file bytes by storage path."""
        ...


    def get_local_path(self, storage_path: str) -> str:
        """Returns a local file system path for the file (downloading/caching if remote)."""
        ...

    def get_url(self, storage_path: str) -> str:
        """Returns a public/accessible URL for the stored file."""
        ...

    def exists(self, storage_path: str) -> bool:
        """Checks whether a file exists at the given storage path."""
        ...

    def delete_file(self, storage_path: str) -> bool:
        """Deletes a file from storage.

        Returns True only if the asset is confirmed gone — never True as a
        placeholder for an unimplemented delete.
        """
        ...
