"""Tests for the storage backend factory."""

import pytest

import footage_engine.storage as storage_module
from footage_engine.config import Settings
from footage_engine.storage import SUPPORTED_BACKENDS, get_storage_backend
from footage_engine.storage.local import LocalStorageBackend


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch):
    """The factory memoises the app-config backend; keep tests independent."""
    monkeypatch.setattr(storage_module, "_storage_instance", None)


def _settings(**overrides) -> Settings:
    base = {"DATABASE_URL": "sqlite:///:memory:", "STORAGE_BACKEND": "local"}
    base.update(overrides)
    return Settings(**base)


def test_local_backend_is_built(tmp_path):
    backend = get_storage_backend(_settings(LOCAL_STORAGE_DIR=str(tmp_path / "store")))
    assert isinstance(backend, LocalStorageBackend)


def test_imagekit_backend_requires_credentials():
    # Credentials are passed as None explicitly so the test does not depend on
    # whatever happens to be configured in the developer's .env.
    with pytest.raises(ValueError, match="IMAGEKIT_"):
        get_storage_backend(
            _settings(
                STORAGE_BACKEND="imagekit",
                IMAGEKIT_PUBLIC_KEY=None,
                IMAGEKIT_PRIVATE_KEY=None,
                IMAGEKIT_URL_ENDPOINT=None,
            )
        )


def test_cloudinary_backend_requires_credentials():
    with pytest.raises(ValueError, match="CLOUDINARY_"):
        get_storage_backend(
            _settings(
                STORAGE_BACKEND="cloudinary",
                CLOUDINARY_CLOUD_NAME=None,
                CLOUDINARY_API_KEY=None,
                CLOUDINARY_API_SECRET=None,
            )
        )


def test_cloudinary_backend_is_built_from_settings(monkeypatch):
    built = {}

    class StubCloudinaryBackend:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(storage_module, "CloudinaryStorageBackend", StubCloudinaryBackend)

    backend = get_storage_backend(
        _settings(
            STORAGE_BACKEND="cloudinary",
            CLOUDINARY_CLOUD_NAME="demo",
            CLOUDINARY_API_KEY="key",
            CLOUDINARY_API_SECRET="secret",
            CLOUDINARY_FOLDER="custom/folder",
        )
    )

    assert isinstance(backend, StubCloudinaryBackend)
    assert built == {
        "cloud_name": "demo",
        "api_key": "key",
        "api_secret": "secret",
        "folder": "custom/folder",
    }


def test_unsupported_backend_fails_instead_of_falling_back_to_disk(tmp_path):
    # Bypass Literal validation to prove the factory itself refuses to silently
    # write to the local filesystem.
    unsupported = Settings.model_construct(
        DATABASE_URL="sqlite:///:memory:",
        STORAGE_BACKEND="azure",
        LOCAL_STORAGE_DIR=str(tmp_path / "store"),
    )
    with pytest.raises(ValueError, match="Unsupported STORAGE_BACKEND"):
        get_storage_backend(unsupported)


def test_explicit_settings_do_not_poison_the_shared_instance(monkeypatch, tmp_path):
    monkeypatch.setattr(
        storage_module,
        "get_settings",
        lambda: _settings(LOCAL_STORAGE_DIR=str(tmp_path / "a")),
    )
    shared = get_storage_backend()
    assert get_storage_backend() is shared

    # An explicitly-configured caller gets its own instance and must not
    # replace the shared one.
    other = get_storage_backend(_settings(LOCAL_STORAGE_DIR=str(tmp_path / "b")))
    assert other is not shared
    assert get_storage_backend() is shared


def test_supported_backends_lists_every_implemented_backend():
    assert set(SUPPORTED_BACKENDS) == {"local", "imagekit", "cloudinary"}
