"""Contract tests: every backend must look and behave like ``StorageBackend``.

Adding a backend is only a drop-in if callers can rely on the same surface, so
these assertions run against every implementation. Behavioural CRUD coverage
lives in ``test_storage.py`` (Local) and ``test_cloudinary_storage.py``.
"""

import inspect

import pytest

from footage_engine.storage.base import StorageBackend
from footage_engine.storage.cloudinary import CloudinaryStorageBackend
from footage_engine.storage.gdrive import GoogleDriveStorageBackend
from footage_engine.storage.imagekit import ImageKitStorageBackend
from footage_engine.storage.local import LocalStorageBackend

BACKENDS = (
    LocalStorageBackend,
    ImageKitStorageBackend,
    CloudinaryStorageBackend,
    GoogleDriveStorageBackend,
)

PROTOCOL_METHODS = ("save_file", "get_file", "get_local_path", "get_url", "exists", "delete_file")


def _backend_ids():
    return [cls.__name__ for cls in BACKENDS]


def test_protocol_declares_every_method():
    """Guard the interface itself: a method silently dropped from the protocol
    would leave callers depending on something that is no longer contracted."""
    for name in PROTOCOL_METHODS:
        assert callable(getattr(StorageBackend, name, None)), f"StorageBackend is missing {name}"


@pytest.mark.parametrize("backend_cls", BACKENDS, ids=_backend_ids())
def test_backend_implements_every_protocol_method(backend_cls):
    for name in PROTOCOL_METHODS:
        assert callable(getattr(backend_cls, name, None)), f"{backend_cls.__name__} is missing {name}"


@pytest.mark.parametrize("backend_cls", BACKENDS, ids=_backend_ids())
def test_backend_exposes_only_the_protocol_api(backend_cls):
    """No backend may leak extra public methods callers could grow to depend on."""
    public = {
        name
        for name, _ in inspect.getmembers(backend_cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert public == set(PROTOCOL_METHODS), (
        f"{backend_cls.__name__} adds {public - set(PROTOCOL_METHODS)} "
        f"and is missing {set(PROTOCOL_METHODS) - public}"
    )


@pytest.mark.parametrize("backend_cls", BACKENDS, ids=_backend_ids())
def test_save_file_signature_matches_the_protocol(backend_cls):
    expected = list(inspect.signature(StorageBackend.save_file).parameters)
    actual = list(inspect.signature(backend_cls.save_file).parameters)
    assert actual == expected, f"{backend_cls.__name__}.save_file must accept {expected}"


@pytest.mark.parametrize("backend_cls", BACKENDS, ids=_backend_ids())
def test_read_methods_take_a_single_storage_path(backend_cls):
    for name in ("get_file", "get_local_path", "get_url", "exists", "delete_file"):
        params = list(inspect.signature(getattr(backend_cls, name)).parameters)
        assert params == ["self", "storage_path"], f"{backend_cls.__name__}.{name}{params}"


def test_local_backend_satisfies_the_opaque_key_contract(temp_dir):
    """A storage path is an opaque key: only the backend ever interprets it."""
    storage = LocalStorageBackend(base_dir=temp_dir)

    stored_path = storage.save_file(b"raw-bytes" * 64, "contract_clip.mp4")

    assert isinstance(stored_path, str) and stored_path
    assert storage.exists(stored_path) is True
    assert storage.get_file(stored_path) == b"raw-bytes" * 64
    assert storage.get_local_path(stored_path).endswith("contract_clip.mp4")
    assert storage.get_url(stored_path).startswith("file://")
    assert storage.delete_file(stored_path) is True
    assert storage.exists(stored_path) is False
