"""Storage backend abstraction protocol."""

from typing import BinaryIO, Protocol


class StorageBackend(Protocol):
    """Interface for raw footage storage backends (Local, ImageKit, S3, etc.)."""

    def save_file(
        self,
        file_data: bytes | BinaryIO,
        filename: str,
        content_type: str | None = None,
    ) -> str:
        """Saves file to storage and returns the unique storage path/key."""
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
        """Deletes a file from storage. Returns True if deleted."""
        ...
