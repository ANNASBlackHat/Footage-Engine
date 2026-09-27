"""Tests for the Cloudinary storage backend.

The Cloudinary SDK is not a hard dependency, so it is replaced with an
in-memory fake that records every call. This keeps the tests offline and
deterministic while still asserting the wire-level contract (resource types,
public ids, chunked routing).
"""

import io
import os

import pytest

import footage_engine.storage.cloudinary as cloudinary_backend
from footage_engine.storage.cloudinary import (
    LARGE_UPLOAD_THRESHOLD_BYTES,
    CloudinaryStorageBackend,
    infer_resource_type,
    public_id_for,
    split_asset_path,
)


class FakeUploader:
    """Records upload/destroy calls instead of talking to Cloudinary."""

    destroy_result = "ok"
    # Cloudinary reports the stored format explicitly. Left unset by default so
    # tests exercise the filename-extension fallback; set it to assert that the
    # reported format wins.
    response_format = None

    def __init__(self):
        self.uploads = []
        self.large_uploads = []
        self.destroyed = []

    def _response(self, options):
        response = {"public_id": options["public_id"]}
        if self.response_format:
            response["format"] = self.response_format
        return response

    def upload(self, file, **options):
        body = file.read() if hasattr(file, "read") else file
        self.uploads.append({"body": body, "options": options})
        return self._response(options)

    def upload_large(self, file, **options):
        # upload_large must be handed a real path it can stream from.
        assert isinstance(file, str) and os.path.exists(file)
        with open(file, "rb") as fh:
            body = fh.read()
        self.large_uploads.append({"path": file, "body": body, "options": options})
        return self._response(options)

    def destroy(self, public_id, **options):
        self.destroyed.append({"public_id": public_id, "options": options})
        return {"result": self.destroy_result}


class FakeApi:
    """Admin API fake; ``missing`` ids raise the way the real SDK does."""

    def __init__(self):
        self.lookups = []
        self.missing = set()

    def resource(self, public_id, **options):
        self.lookups.append({"public_id": public_id, "options": options})
        if public_id in self.missing:
            raise Exception("Resource not found")
        return {"public_id": public_id}


class FakeCloudinary:
    def __init__(self):
        self.config_calls = []
        self.uploader = FakeUploader()
        self.api = FakeApi()

    def config(self, **options):
        self.config_calls.append(options)


@pytest.fixture
def fake_sdk(monkeypatch):
    """Installs the fake SDK in place of the optional ``cloudinary`` package."""
    fake = FakeCloudinary()
    monkeypatch.setattr(cloudinary_backend, "cloudinary", fake)
    return fake


@pytest.fixture
def storage(fake_sdk, temp_dir):
    return CloudinaryStorageBackend(
        cloud_name="demo",
        api_key="key",
        api_secret="secret",
        folder="footage_engine/raw",
        cache_dir=temp_dir,
    )


# --------------------------------------------------------------------------- #
# Classification helpers
# --------------------------------------------------------------------------- #

def test_infer_resource_type_from_extension():
    assert infer_resource_type("clip.mp4") == "video"
    assert infer_resource_type("clip.webm") == "video"
    assert infer_resource_type("photo.jpg") == "image"
    assert infer_resource_type("photo.png") == "image"


def test_infer_resource_type_hints_only_apply_to_unknown_extensions():
    # A recognised extension always wins, which is what keeps the read path and
    # the write path in agreement.
    assert infer_resource_type("clip.mp4", media_type="image") == "video"
    # Unknown extension: the explicit hint is used.
    assert infer_resource_type("blob.bin", media_type="image") == "image"
    assert infer_resource_type("blob.bin", content_type="image/png") == "image"
    # Nothing to go on: the engine default (video) applies.
    assert infer_resource_type("blob.bin") == "video"


def test_split_asset_path_and_public_id():
    assert split_asset_path("footage_engine/raw/chunks/abc.mp4") == ("footage_engine/raw/chunks/abc", ".mp4")
    assert split_asset_path("/leading/slash.mp4") == ("leading/slash", ".mp4")
    assert split_asset_path("no_extension") == ("no_extension", "")
    assert public_id_for("footage_engine/raw/a.mp4") == "footage_engine/raw/a"


def test_large_upload_threshold_stays_below_cloudinary_limit():
    # Cloudinary returns HTTP 413 above 100 MB, so the routing threshold must
    # stay clear of it.
    assert LARGE_UPLOAD_THRESHOLD_BYTES < 100 * 1000 * 1000


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def test_requires_cloudinary_package(monkeypatch):
    monkeypatch.setattr(cloudinary_backend, "cloudinary", None)
    with pytest.raises(ImportError) as exc:
        CloudinaryStorageBackend(cloud_name="demo", api_key="key", api_secret="secret")
    assert "cloudinary" in str(exc.value)


def test_configures_sdk_credentials(fake_sdk):
    CloudinaryStorageBackend(
        cloud_name="demo",
        api_key="key",
        api_secret="secret",
        folder="/footage_engine/raw/",
    )
    assert fake_sdk.config_calls == [
        {"cloud_name": "demo", "api_key": "key", "api_secret": "secret", "secure": True}
    ]


# --------------------------------------------------------------------------- #
# save_file
# --------------------------------------------------------------------------- #

def test_save_file_uploads_video_with_video_resource_type(storage, fake_sdk):
    stored_path = storage.save_file(b"video-bytes", "pexels_123_ab12.mp4")

    assert stored_path == "footage_engine/raw/pexels_123_ab12.mp4"
    assert len(fake_sdk.uploader.uploads) == 1
    call = fake_sdk.uploader.uploads[0]
    assert call["body"] == b"video-bytes"
    assert call["options"]["resource_type"] == "video"
    assert call["options"]["public_id"] == "footage_engine/raw/pexels_123_ab12"
    assert call["options"]["overwrite"] is True
    assert call["options"]["unique_filename"] is False


def test_save_file_uploads_image_with_image_resource_type(storage, fake_sdk):
    stored_path = storage.save_file(b"image-bytes", "photo.png")

    assert stored_path == "footage_engine/raw/photo.png"
    assert fake_sdk.uploader.uploads[0]["options"]["resource_type"] == "image"


def test_save_file_keeps_subfolders_in_the_public_id(storage, fake_sdk):
    # The chunk uploader passes names like "chunks/<id>.mp4".
    stored_path = storage.save_file(b"chunk-bytes", "chunks/item1_chunk2.mp4")

    assert stored_path == "footage_engine/raw/chunks/item1_chunk2.mp4"
    assert fake_sdk.uploader.uploads[0]["options"]["public_id"] == "footage_engine/raw/chunks/item1_chunk2"


def test_save_file_accepts_a_file_like_payload(storage, fake_sdk):
    stored_path = storage.save_file(io.BytesIO(b"streamed-bytes"), "clip.mov")

    assert stored_path == "footage_engine/raw/clip.mov"
    assert fake_sdk.uploader.uploads[0]["body"] == b"streamed-bytes"


def test_save_file_prefers_the_format_reported_by_cloudinary(storage, fake_sdk):
    # The SDK's response is authoritative: if Cloudinary normalised .mov to mp4,
    # the stored path must record mp4 so read-time inference agrees.
    fake_sdk.uploader.response_format = "mp4"

    assert storage.save_file(b"payload", "clip.mov") == "footage_engine/raw/clip.mp4"


def test_save_file_routes_large_payload_to_upload_large(storage, fake_sdk, monkeypatch):
    # Keep the test small instead of allocating a real 95 MB buffer.
    monkeypatch.setattr(cloudinary_backend, "LARGE_UPLOAD_THRESHOLD_BYTES", 1024)
    payload = b"v" * 4096

    stored_path = storage.save_file(payload, "huge_video.mp4")

    assert stored_path == "footage_engine/raw/huge_video.mp4"
    assert fake_sdk.uploader.uploads == []
    assert len(fake_sdk.uploader.large_uploads) == 1
    large = fake_sdk.uploader.large_uploads[0]
    assert large["body"] == payload
    assert large["options"]["resource_type"] == "video"
    assert large["options"]["chunk_size"] == cloudinary_backend.UPLOAD_LARGE_CHUNK_SIZE
    # The spooled temp file must not outlive the call.
    assert not os.path.exists(large["path"])


def test_save_file_uses_single_upload_at_the_threshold(storage, fake_sdk, monkeypatch):
    monkeypatch.setattr(cloudinary_backend, "LARGE_UPLOAD_THRESHOLD_BYTES", 1024)

    storage.save_file(b"v" * 1024, "exactly_limit.mp4")

    assert len(fake_sdk.uploader.uploads) == 1
    assert fake_sdk.uploader.large_uploads == []


# --------------------------------------------------------------------------- #
# get_url / get_file / get_local_path
# --------------------------------------------------------------------------- #

def test_get_url_builds_a_delivery_url(storage):
    assert storage.get_url("footage_engine/raw/clip.mp4") == (
        "https://res.cloudinary.com/demo/video/upload/footage_engine/raw/clip.mp4"
    )
    assert storage.get_url("footage_engine/raw/photo.jpg") == (
        "https://res.cloudinary.com/demo/image/upload/footage_engine/raw/photo.jpg"
    )
    # Chunk subfolders travel as part of the public id path.
    assert storage.get_url("footage_engine/raw/chunks/a_b.mp4") == (
        "https://res.cloudinary.com/demo/video/upload/footage_engine/raw/chunks/a_b.mp4"
    )


def test_get_url_passes_through_non_storage_references(storage):
    for reference in ("https://example.com/clip.mp4", "file:///tmp/clip.mp4"):
        assert storage.get_url(reference) == reference


def test_get_file_streams_the_delivery_url(storage, monkeypatch):
    captured = {}

    class FakeResponse:
        content = b"downloaded-bytes"

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None):
        captured["url"] = url
        return FakeResponse()

    monkeypatch.setattr(cloudinary_backend.requests, "get", fake_get)

    assert storage.get_file("footage_engine/raw/clip.mp4") == b"downloaded-bytes"
    assert captured["url"] == "https://res.cloudinary.com/demo/video/upload/footage_engine/raw/clip.mp4"


def test_get_local_path_resolves_file_urls_without_touching_cloudinary(storage):
    assert storage.get_local_path("file:///tmp/some_clip.mp4") == "/tmp/some_clip.mp4"


def test_get_local_path_downloads_an_owned_asset_into_the_cache(storage, monkeypatch):
    payload = b"x" * (cloudinary_backend.MIN_VALID_FILE_BYTES + 64)
    requests_seen = []

    class FakeResponse:
        content = payload

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None):
        requests_seen.append(url)
        return FakeResponse()

    monkeypatch.setattr(cloudinary_backend.requests, "get", fake_get)

    local_path = storage.get_local_path("footage_engine/raw/clips/a.mp4")

    assert local_path == os.path.join(storage.cache_dir, "footage_engine_raw_clips_a.mp4")
    with open(local_path, "rb") as fh:
        assert fh.read() == payload

    # A second call is served from the cache with no further HTTP traffic.
    storage.get_local_path("footage_engine/raw/clips/a.mp4")
    assert len(requests_seen) == 1


# --------------------------------------------------------------------------- #
# exists / delete_file
# --------------------------------------------------------------------------- #

def test_exists_uses_the_admin_api(storage, fake_sdk):
    assert storage.exists("footage_engine/raw/clip.mp4") is True
    assert fake_sdk.api.lookups[0]["public_id"] == "footage_engine/raw/clip"
    assert fake_sdk.api.lookups[0]["options"]["resource_type"] == "video"


def test_exists_returns_false_when_the_admin_api_raises(storage, fake_sdk):
    fake_sdk.api.missing.add("footage_engine/raw/clip")
    assert storage.exists("footage_engine/raw/clip.mp4") is False


def test_delete_file_reports_only_confirmed_deletions(storage, fake_sdk):
    assert storage.delete_file("footage_engine/raw/clip.mp4") is True
    assert fake_sdk.uploader.destroyed[0]["public_id"] == "footage_engine/raw/clip"
    assert fake_sdk.uploader.destroyed[0]["options"]["resource_type"] == "video"

    # A "not found" response is not a deletion.
    fake_sdk.uploader.destroy_result = "not found"
    assert storage.delete_file("footage_engine/raw/clip.mp4") is False


# --------------------------------------------------------------------------- #
# Round-trip symmetry
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [
        ("clip.mp4", "video"),
        ("clip.webm", "video"),
        ("photo.jpg", "image"),
        ("photo.webp", "image"),
    ],
)
def test_resource_type_is_symmetric_between_write_and_read(
    storage, fake_sdk, filename, expected_type
):
    stored_path = storage.save_file(b"payload", filename)

    # The type chosen at upload time must be recomputed identically from the
    # stored path, otherwise URLs and deletes would target the wrong asset.
    assert fake_sdk.uploader.uploads[-1]["options"]["resource_type"] == expected_type

    storage.exists(stored_path)
    assert fake_sdk.api.lookups[-1]["options"]["resource_type"] == expected_type

    storage.delete_file(stored_path)
    assert fake_sdk.uploader.destroyed[-1]["options"]["resource_type"] == expected_type

    assert f"/{expected_type}/upload/" in storage.get_url(stored_path)


def test_no_caller_changes_are_needed_for_the_protocol(storage):
    """The backend satisfies every method the StorageBackend protocol declares."""
    assert callable(storage.save_file)
    assert callable(storage.get_file)
    assert callable(storage.get_local_path)
    assert callable(storage.get_url)
    assert callable(storage.exists)
    assert callable(storage.delete_file)


