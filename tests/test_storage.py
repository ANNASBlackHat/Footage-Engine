"""Tests for local storage backend."""

import pytest
from footage_engine.storage.local import LocalStorageBackend


def test_local_storage_crud(temp_dir):
    storage = LocalStorageBackend(base_dir=temp_dir)
    file_bytes = b"fake video raw bytes mp4"
    filename = "test_video.mp4"

    # Save
    path = storage.save_file(file_bytes, filename)
    assert path == filename
    assert storage.exists(filename)

    # Read
    read_bytes = storage.get_file(filename)
    assert read_bytes == file_bytes

    # Local path
    local_path = storage.get_local_path(filename)
    assert local_path.endswith("test_video.mp4")

    # URL
    url = storage.get_url(filename)
    assert url.startswith("file://")

    # Delete
    assert storage.delete_file(filename) is True
    assert not storage.exists(filename)

    # Read non-existent raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        storage.get_file("non_existent.mp4")
