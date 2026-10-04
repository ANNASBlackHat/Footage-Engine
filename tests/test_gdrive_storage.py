"""Tests for the Google Drive storage backend.

The Google SDK is an optional extra and is *not* installed here, so every SDK
symbol the backend touches (``build``, ``service_account``, ``HttpError``,
``MediaIoBaseUpload``/``MediaIoBaseDownload``) is replaced with an in-memory
fake. That keeps the suite offline while still asserting the wire-level
contract: parents, scopes, the anyone-reader grant, the file-id key, the
404-based existence checks, and the backoff loop.
"""

import io
import json
from types import SimpleNamespace

import pytest

import footage_engine.storage.gdrive as gdrive_backend
from footage_engine.storage.gdrive import (
    MAX_BACKOFF_SEC,
    GoogleDriveStorageBackend,
    backoff_seconds,
    build_credentials,
    guess_content_type,
    parse_credentials,
    parse_oauth_credentials,
    safe_cache_name,
    split_scopes,
)

KEY_JSON = json.dumps(
    {
        "type": "service_account",
        "project_id": "demo",
        "private_key_id": "pk-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----\n",
        "client_email": "footage-engine@demo.iam.gserviceaccount.com",
        "client_id": "12345",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
)

# What the consent flow in scripts/gdrive_oauth_login.py writes: no private key,
# a refresh token standing in for one.
OAUTH_JSON = json.dumps(
    {
        "type": "authorized_user",
        "client_id": "client-id.apps.googleusercontent.com",
        "client_secret": "client-secret",
        "refresh_token": "1//refresh-token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": ["https://www.googleapis.com/auth/drive"],
    }
)


class FakeHttpError(Exception):
    """Stands in for ``googleapiclient.errors.HttpError``.

    The module's ``except HttpError`` clause resolves the name at runtime, so
    monkeypatching it keeps the exception shape identical whether or not the
    real SDK is installed.
    """

    def __init__(self, status: int, reason: str = ""):
        self.resp = SimpleNamespace(status=status)
        body = {"error": {"code": status, "errors": [{"reason": reason}]}}
        self.content = json.dumps(body).encode()
        super().__init__(f"HTTP {status} {reason}")


class FakeRequest:
    """A single API request: records the call, then answers (or fails)."""

    def __init__(self, result=None, error: Exception | None = None, errors=None):
        self.result = result
        self.error = error
        # Errors raised one-per-call before the request succeeds.
        self.errors = list(errors or [])
        self.execute_calls = 0
        self.chunk_calls = 0

    def execute(self):
        self.execute_calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.error is not None:
            raise self.error
        return self.result

    def next_chunk(self):
        self.chunk_calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.error is not None:
            raise self.error
        return (None, self.result)

class FakeUpload:
    """Captures the MediaIoBaseUpload arguments instead of buffering a video."""

    def __init__(self, payload, mimetype=None, chunksize=None, resumable=False):
        self.payload = payload
        self.mimetype = mimetype
        self.chunksize = chunksize
        self.resumable = resumable


class FakeDownloader:
    """Writes the fake file's bytes straight into the destination handle."""

    instances: list = []

    def __init__(self, destination, request, chunksize=None):
        self.destination = destination
        self.request = request
        self.chunksize = chunksize
        self.chunks = 0
        FakeDownloader.instances.append(self)

    def next_chunk(self):
        self.chunks += 1
        self.destination.write(self.request.body)
        return (None, True)


class FakeFiles:
    """The ``files`` resource: records create/get/delete and models 404s."""

    def __init__(self):
        self.creates = []
        self.gets = []
        self.deletes = []
        self.names = {}  # fileId -> remote name
        self.content = {}  # fileId -> bytes
        self.missing = set()  # ids that answer 404
        self.create_errors: list = []  # raised one-per-chunk before success
        self.create_request: FakeRequest | None = None
        self.public_grant_error: Exception | None = None

    def create(self, **kwargs):
        self.creates.append(kwargs)
        body = kwargs.get("body") or {}
        file_id = "file_abc123"
        self.names[file_id] = body.get("name", "asset")
        request = FakeRequest(result={"id": file_id, "name": body.get("name", "asset")})
        request.errors = list(self.create_errors)
        self.create_errors = []
        self.create_request = request
        return request

    def get(self, **kwargs):
        self.gets.append(kwargs)
        file_id = kwargs.get("fileId")
        if file_id in self.missing:
            return FakeRequest(error=FakeHttpError(404, "notFound"))
        if kwargs.get("alt") == "media":
            return SimpleNamespace(body=self.content.get(file_id, b""))
        return FakeRequest(result={"id": file_id, "name": self.names.get(file_id, "clip.mp4")})

    def get_media(self, **kwargs):
        """Mirrors the real client: this, not ``get(alt="media")``, fetches bytes.

        ``get(alt="media")`` silently degrades to ``alt=json`` in
        google-api-python-client 2.x, which streams metadata where content was
        expected. Modelling the two as different methods is what lets the test
        below assert the backend stopped relying on the broken spelling.
        """
        self.gets.append(kwargs)
        file_id = kwargs.get("fileId")
        if file_id in self.missing:
            return FakeRequest(error=FakeHttpError(404, "notFound"))
        return SimpleNamespace(body=self.content.get(file_id, b""))

    def delete(self, **kwargs):
        self.deletes.append(kwargs)
        file_id = kwargs.get("fileId")
        if file_id in self.missing:
            return FakeRequest(error=FakeHttpError(404, "notFound"))
        self.missing.add(file_id)
        return FakeRequest(result=None)


class FakePermissions:
    def __init__(self, files: FakeFiles):
        self.files = files
        self.grants = []

    def create(self, **kwargs):
        self.grants.append(kwargs)
        if self.files.public_grant_error is not None:
            return FakeRequest(error=self.files.public_grant_error)
        return FakeRequest(result={"id": "perm-1", "type": "anyone", "role": "reader"})


class FakeService:
    def __init__(self):
        self.files_resource = FakeFiles()
        self.permissions_resource = FakePermissions(self.files_resource)

    def files(self):
        return self.files_resource

    def permissions(self):
        return self.permissions_resource


@pytest.fixture
def sdk(monkeypatch):
    """Installs the fakes for every optional-SDK symbol the backend imports."""
    FakeDownloader.instances = []
    service = FakeService()
    build_calls = []

    def fake_build(api, version, credentials=None, cache_discovery=None):
        build_calls.append(
            {"api": api, "version": version, "credentials": credentials, "cache_discovery": cache_discovery}
        )
        return service

    class FakeCredentials:
        @staticmethod
        def from_service_account_info(info, scopes=None):
            return {"client_email": info["client_email"], "scopes": tuple(scopes or ())}

    class FakeUserCredentials:
        @staticmethod
        def from_authorized_user_info(info, scopes=None):
            return {
                "refresh_token": info["refresh_token"],
                "client_id": info["client_id"],
                "scopes": tuple(scopes or ()),
            }

    monkeypatch.setattr(gdrive_backend, "build", fake_build)
    monkeypatch.setattr(
        gdrive_backend, "service_account", SimpleNamespace(Credentials=FakeCredentials)
    )
    monkeypatch.setattr(
        gdrive_backend, "user_credentials", SimpleNamespace(Credentials=FakeUserCredentials)
    )
    monkeypatch.setattr(gdrive_backend, "HttpError", FakeHttpError)
    monkeypatch.setattr(gdrive_backend, "MediaIoBaseUpload", FakeUpload)
    monkeypatch.setattr(gdrive_backend, "MediaIoBaseDownload", FakeDownloader)
    return SimpleNamespace(service=service, build_calls=build_calls)


def make_backend(**overrides) -> GoogleDriveStorageBackend:
    """Builds a backend with an in-memory service account key."""
    options = {
        "credentials_json": KEY_JSON,
        "folder_id": "folder_1",
        "cache_dir": None,
    }
    options.update(overrides)
    return GoogleDriveStorageBackend(**options)


# -- credentials -----------------------------------------------------------------


def test_missing_credentials_names_both_env_vars():
    with pytest.raises(ValueError, match="GDRIVE_SERVICE_ACCOUNT_FILE"):
        parse_credentials(None, None)


def test_file_and_json_sources_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_credentials("/tmp/key.json", KEY_JSON)


def test_malformed_json_is_rejected_with_the_parse_error():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_credentials(None, "{not json")


def test_key_missing_required_fields_points_at_console_download():
    with pytest.raises(ValueError, match="private_key"):
        parse_credentials(None, json.dumps({"client_email": "a@b.iam.gserviceaccount.com"}))


def test_service_account_slot_redirects_an_oauth_token_to_the_oauth_variable():
    """The two families stay separate: no silent cross-acceptance."""
    with pytest.raises(ValueError, match="GDRIVE_OAUTH_CREDENTIALS_FILE"):
        parse_credentials(None, OAUTH_JSON)


def test_key_file_is_read_from_disk(tmp_path):
    key_path = tmp_path / "sa.json"
    key_path.write_text(KEY_JSON, encoding="utf-8")
    info = parse_credentials(str(key_path), None)
    assert info["client_email"] == "footage-engine@demo.iam.gserviceaccount.com"


def test_nonexistent_key_file_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        parse_credentials(str(tmp_path / "nope.json"), None)


# -- user oauth credentials ------------------------------------------------------


def test_missing_oauth_names_the_oauth_variables():
    with pytest.raises(ValueError, match="GDRIVE_OAUTH_CREDENTIALS_FILE"):
        parse_oauth_credentials(None, None)


def test_oauth_file_and_json_sources_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_oauth_credentials("/tmp/token.json", OAUTH_JSON)


def test_oauth_token_without_a_refresh_token_is_rejected():
    """A token missing refresh_token cannot refresh; fail at load, not mid-upload."""
    broken = json.dumps({"type": "authorized_user", "client_id": "c", "client_secret": "s"})
    with pytest.raises(ValueError, match="refresh_token"):
        parse_oauth_credentials(None, broken)


def test_oauth_slot_rejects_a_service_account_key():
    with pytest.raises(ValueError, match="GDRIVE_SERVICE_ACCOUNT_FILE"):
        parse_oauth_credentials(None, KEY_JSON)


def test_oauth_token_is_read_from_disk(tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text(OAUTH_JSON, encoding="utf-8")
    info = parse_oauth_credentials(str(token_path), None)
    assert info["refresh_token"] == "1//refresh-token"


def test_build_credentials_dispatches_on_type_not_on_the_configured_variable(sdk):
    """Both identities reach the SDK through one builder, keyed off ``type``."""
    from_info = parse_credentials(None, KEY_JSON)
    oauth_info = parse_oauth_credentials(None, OAUTH_JSON)

    sa_creds = build_credentials(from_info, ["https://www.googleapis.com/auth/drive"])
    oauth_creds = build_credentials(oauth_info, ["https://www.googleapis.com/auth/drive"])

    assert sa_creds["client_email"] == "footage-engine@demo.iam.gserviceaccount.com"
    assert oauth_creds["refresh_token"] == "1//refresh-token"
    assert oauth_creds["scopes"] == ("https://www.googleapis.com/auth/drive",)


def test_backend_accepts_a_user_oauth_token(sdk):
    backend = make_backend(credentials_json=None, oauth_credentials_json=OAUTH_JSON)
    assert backend.identity == "user_oauth"
    call = sdk.build_calls[0]
    assert call["api"] == "drive"
    assert call["credentials"]["refresh_token"] == "1//refresh-token"


def test_backend_records_a_service_account_identity(sdk):
    assert make_backend().identity == "service_account"


def test_backend_rejects_two_identities_at_once(sdk):
    with pytest.raises(ValueError, match="exactly one credential identity"):
        make_backend(oauth_credentials_json=OAUTH_JSON)


def test_backend_rejects_no_identity_at_all(sdk):
    with pytest.raises(ValueError, match="exactly one credential identity"):
        make_backend(credentials_json=None)


def test_upload_uses_the_same_wire_contract_under_user_oauth(sdk):
    """Identity changes who owns the file, not how the upload is shaped."""
    backend = make_backend(credentials_json=None, oauth_credentials_json=OAUTH_JSON)
    file_id = backend.save_file(b"bytes", "clip.mp4")
    created = sdk.service.files_resource.creates[-1]
    assert created["body"]["name"] == "clip.mp4"
    assert created["body"]["parents"] == ["folder_1"]
    assert created["supportsAllDrives"] is True
    assert sdk.service.permissions_resource.grants[-1]["fileId"] == file_id


def test_destination_is_required_even_with_credentials():
    with pytest.raises(ValueError, match="GDRIVE_FOLDER_ID"):
        make_backend(folder_id=None, drive_id=None)


def test_build_receives_service_account_credentials_and_drive_v3(sdk):
    make_backend(scopes="https://www.googleapis.com/auth/drive")
    call = sdk.build_calls[0]
    assert call["api"] == "drive"
    assert call["version"] == "v3"
    assert call["cache_discovery"] is False
    assert call["credentials"]["client_email"] == "footage-engine@demo.iam.gserviceaccount.com"
    assert call["credentials"]["scopes"] == ("https://www.googleapis.com/auth/drive",)


def test_scopes_and_content_type_helpers():
    assert split_scopes(None) == ["https://www.googleapis.com/auth/drive"]
    assert split_scopes("") == ["https://www.googleapis.com/auth/drive"]
    assert split_scopes(" https://www.googleapis.com/auth/drive.file , ") == [
        "https://www.googleapis.com/auth/drive.file"
    ]
    assert guess_content_type("clip.mp4") == "video/mp4"
    assert guess_content_type("x.xyz", media_type="video") == "video/mp4"
    assert guess_content_type("x.xyz", "image/webp", "image") == "image/webp"


def test_cache_name_recovers_an_extension_from_the_remote_name():
    assert safe_cache_name("1AbC9xyz", "raw/clip.mov") == "1AbC9xyz.mov"
    assert safe_cache_name("1AbC9xyz", "archive.tar.gz") == "1AbC9xyz.mp4"
    assert safe_cache_name("1AbC9xyz", "") == "1AbC9xyz.mp4"


# -- save_file -------------------------------------------------------------------


def test_save_file_returns_the_file_id_and_files_it_under_the_folder(sdk):
    backend = make_backend(cache_dir=None)
    key = backend.save_file(b"video-bytes", "clips/raw/master.mp4", media_type="video")

    assert key == "file_abc123"
    create = sdk.service.files_resource.creates[0]
    assert create["body"]["name"] == "clips/raw/master.mp4"
    assert create["body"]["parents"] == ["folder_1"]
    assert create["supportsAllDrives"] is True
    assert create["media_body"].resumable is True
    assert create["media_body"].mimetype == "video/mp4"


def test_save_file_grants_anyone_reader_so_the_url_is_playable(sdk):
    backend = make_backend(cache_dir=None)
    backend.save_file(b"bytes", "clip.mp4")

    grant = sdk.service.permissions_resource.grants[0]
    assert grant["fileId"] == "file_abc123"
    assert grant["body"] == {"role": "reader", "type": "anyone"}
    assert grant["supportsAllDrives"] is True


def test_save_file_fails_loudly_when_public_sharing_is_refused(sdk):
    sdk.service.files_resource.public_grant_error = FakeHttpError(400, "domainPolicy")
    backend = make_backend(cache_dir=None)

    with pytest.raises(PermissionError, match="Anyone with the link"):
        backend.save_file(b"bytes", "clip.mp4")


def test_save_file_backs_off_on_rate_limit_then_succeeds(sdk, monkeypatch):
    sleeps = []
    monkeypatch.setattr(gdrive_backend.time, "sleep", sleeps.append)

    # Two throttled chunks in a row, then success: proves the delay grows.
    sdk.service.files_resource.create_errors = [
        FakeHttpError(429, "userRateLimitExceeded"),
        FakeHttpError(403, "rateLimitExceeded"),
    ]

    backend = make_backend(cache_dir=None)
    backend.save_file(b"bytes", "clip.mp4")

    request = sdk.service.files_resource.create_request
    assert request.chunk_calls == 3, "two failed chunks plus the successful retry"
    assert len(sleeps) == 2
    assert all(0 < delay <= MAX_BACKOFF_SEC for delay in sleeps)
    assert sleeps[0] < sleeps[1], "backoff must grow exponentially"


def test_save_file_reraises_errors_that_backoff_cannot_fix(sdk, monkeypatch):
    monkeypatch.setattr(gdrive_backend.time, "sleep", lambda _: pytest.fail("must not back off"))
    backend = make_backend(cache_dir=None)

    with pytest.raises(FakeHttpError):
        # A 404 during upload is a real failure, not throttling.
        backend.service.files_resource.create = lambda **kw: FakeRequest(
            error=FakeHttpError(404, "notFound")
        )
        backend.save_file(b"bytes", "clip.mp4")


def test_save_file_accepts_a_file_object(sdk):
    backend = make_backend(cache_dir=None)
    key = backend.save_file(io.BytesIO(b"streamed"), "clip.mp4")
    assert key == "file_abc123"
    upload = sdk.service.files_resource.creates[0]["media_body"]
    assert upload.payload.read() == b"streamed"


# -- read path -------------------------------------------------------------------


def test_get_url_builds_the_public_download_link(sdk):
    backend = make_backend(cache_dir=None)
    assert backend.get_url("file_abc123") == (
        "https://drive.google.com/uc?export=download&id=file_abc123"
    )


def test_get_url_passes_foreign_references_through(sdk):
    backend = make_backend(cache_dir=None)
    for reference in (
        "https://www.youtube.com/watch?v=abc",
        "file:///tmp/clip.mp4",
        "http://example.com/clip.mp4",
    ):
        assert backend.get_url(reference) == reference


def test_get_url_honours_a_custom_template(sdk):
    backend = make_backend(
        cache_dir=None,
        url_template="https://lh3.googleusercontent.com/d/{file_id}",
    )
    assert backend.get_url("file_abc123") == "https://lh3.googleusercontent.com/d/file_abc123"


def test_get_file_streams_bytes_over_authenticated_media_download(sdk):
    sdk.service.files_resource.content["file_abc123"] = b"z" * 4096
    backend = make_backend(cache_dir=None)

    data = backend.get_file("file_abc123")

    assert data == b"z" * 4096
    media_request = sdk.service.files_resource.gets[0]
    assert media_request["supportsAllDrives"] is True


def test_download_uses_get_media_not_get_alt_media(sdk):
    """Regression: ``get(alt="media")`` degrades to ``alt=json`` in client 2.x.

    The discovery default wins over the ``alt`` keyword, so the old call
    returned the request URI as ``.../files/<id>?alt=json`` and streamed file
    *metadata* into the destination. The fake returns metadata for a bare
    ``get()``, so this fails if the backend regresses to the broken spelling.
    """
    sdk.service.files_resource.content["file_abc123"] = b"real-bytes"
    backend = make_backend(cache_dir=None)

    # A bare get() would hand back JSON metadata here, not these bytes.
    assert backend.get_file("file_abc123") == b"real-bytes"

    sdk.service.files_resource.gets.clear()
    backend.get_local_path("file_abc123")
    # get_media passes no `alt`; it sets the media parameter itself.
    assert all("alt" not in kwargs for kwargs in sdk.service.files_resource.gets)


def test_get_local_path_caches_under_a_name_with_the_metadata_extension(sdk, tmp_path):
    sdk.service.files_resource.names["file_abc123"] = "raw/season1/ep02.mov"
    sdk.service.files_resource.content["file_abc123"] = b"v" * 4096
    backend = make_backend(cache_dir=str(tmp_path))

    local_path = backend.get_local_path("file_abc123")

    assert local_path.endswith("file_abc123.mov")
    assert open(local_path, "rb").read() == b"v" * 4096
    assert not list(tmp_path.glob("*.tmp")), "partial downloads must not survive"

    # Second call must be a pure cache hit: no new metadata round trips.
    # `fields` marks the metadata-only probe; the media download sends none.
    metadata_calls = [g for g in sdk.service.files_resource.gets if "fields" in g]
    assert len(metadata_calls) == 1
    assert backend.get_local_path("file_abc123") == local_path


def test_get_local_path_defers_to_the_shared_remote_resolver(sdk, tmp_path):
    backend = make_backend(cache_dir=str(tmp_path))
    local = tmp_path / "existing.mp4"
    local.write_bytes(b"x" * 2048)

    # file:// references are the shared resolver's job and never hit Drive.
    assert backend.get_local_path(f"file://{local}") == str(local)
    assert sdk.service.files_resource.gets == []


def test_exists_is_true_for_live_files_and_false_only_on_404(sdk):
    backend = make_backend(cache_dir=None)
    assert backend.exists("file_abc123") is True

    sdk.service.files_resource.missing.add("gone_id")
    assert backend.exists("gone_id") is False
    assert backend.exists("") is False


def test_exists_propagates_permission_errors(sdk):
    backend = make_backend(cache_dir=None)
    sdk.service.files_resource.missing.add("file_abc123")
    # A 404 is "missing"; anything else must not be silently swallowed.
    sdk.service.files_resource.get = lambda **kw: FakeRequest(
        error=FakeHttpError(403, "forbidden")
    )
    with pytest.raises(FakeHttpError):
        backend.exists("locked_id")


def test_delete_file_confirms_the_file_is_actually_gone(sdk):
    backend = make_backend(cache_dir=None)

    assert backend.delete_file("file_abc123") is True
    assert sdk.service.files_resource.deletes[0] == {
        "fileId": "file_abc123",
        "supportsAllDrives": True,
    }

    # Deleting an already-absent file is a no-op, not a success.
    assert backend.delete_file("file_abc123") is False
    assert backend.delete_file("") is False


# -- backoff policy --------------------------------------------------------------


def test_backoff_applies_google_truncated_exponential_on_throttling():
    rate_limit = FakeHttpError(403, "rateLimitExceeded")
    assert backoff_seconds(rate_limit, 0) == 1
    assert backoff_seconds(rate_limit, 5) == 32
    assert backoff_seconds(rate_limit, 9) == MAX_BACKOFF_SEC
    assert backoff_seconds(FakeHttpError(429, "userRateLimitExceeded"), 2) == 4
    # Reason-less 403s are treated as throttling too.
    assert backoff_seconds(FakeHttpError(403, ""), 1) == 2


def test_backoff_declines_errors_that_will_not_heal():
    assert backoff_seconds(FakeHttpError(404, "notFound"), 0) is None
    assert backoff_seconds(FakeHttpError(403, "cannotDownloadFile"), 0) is None
    assert backoff_seconds(ValueError("boom"), 0) is None


def test_missing_sdk_raises_an_install_hint(monkeypatch):
    monkeypatch.setattr(gdrive_backend, "service_account", None)
    with pytest.raises(ImportError, match="footage-engine\\[gdrive\\]"):
        make_backend(cache_dir=None)

