"""Google Drive managed cloud storage backend.

Three things make Drive different from the Local/ImageKit/Cloudinary backends
and shape this module:

* **Drive has no CDN and no public bucket.** Every file is private until a
  permission says otherwise, so ``save_file`` grants an ``anyone`` reader
  permission on every upload — that is what makes ``get_url`` return a link a
  browser can actually play. Google Workspace admins can block that grant
  (domain-wide "Anyone with the link" sharing disabled); when that happens the
  upload fails loudly rather than storing a private file whose broken URL would
  only surface later in a search response.
* **Keys are opaque file ids, not paths.** ``save_file`` returns the Drive
  ``fileId``, which carries no extension — so the local cache name is derived
  from the file's *metadata* name instead, because OpenCV and ffprobe both need
  a real extension to sniff the container.
* **Reads and writes are authenticated.** Unlike Cloudinary/ImageKit, ``get_url``
  cannot be used to fetch bytes: content download goes through
  ``files.get(alt="media")`` with the service account's bearer token.

Auth is a **service account**, not an API key (an API key cannot authorize
writes) and not a browser OAuth consent flow (this is a headless worker). A
service account owns no Drive of its own, so ``GDRIVE_FOLDER_ID`` must name a
folder shared with the service account's ``client_email``, or ``GDRIVE_DRIVE_ID``
a Shared Drive it has been added to.

Known limitation: Google prescribes truncated exponential backoff for
``rateLimitExceeded`` / ``userRateLimitExceeded``; this backend applies it with
a 32 s ceiling, but quota *sizes* are per-project and not enforced here — a bulk
ingest can still exhaust the daily upload quota for the project.
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO, Sequence

from footage_engine.storage._remote_cache import (
    KNOWN_MEDIA_EXTENSIONS,
    MIN_VALID_FILE_BYTES,
    resolve_remote_reference,
)

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
except ImportError:  # pragma: no cover - the ImportError path is unit tested
    service_account = None  # type: ignore[assignment]
    build = None  # type: ignore[assignment]
    MediaIoBaseDownload = None  # type: ignore[assignment]
    MediaIoBaseUpload = None  # type: ignore[assignment]

    class HttpError(Exception):  # type: ignore[no-redef]
        """Stand-in so ``except HttpError`` still resolves without the SDK.

        The real class comes from ``googleapiclient.errors``; this only exists so
        the module imports (and its tests can inject a fake) when the optional
        ``gdrive`` extra is not installed.
        """

        resp: Any = None
        content: bytes = b""


# A single OAuth scope; comma-separated in Settings and split here.
DEFAULT_SCOPES = ("https://www.googleapis.com/auth/drive",)

# Drive requires upload chunk sizes to be a multiple of 256 KiB. 8 MiB keeps
# the resumable session moving without holding a whole video in memory.
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024

# Ceiling for the truncated exponential backoff Google's limits documentation
# prescribes for the per-user and per-project rate limit errors.
MAX_BACKOFF_SEC = 32.0

# Error reasons Drive reports for throttling. ``sharingRateLimitExceeded`` is
# specific to the permission grant used to make uploads publicly readable.
QUOTA_REASONS = frozenset(
    {
        "rateLimitExceeded",
        "userRateLimitExceeded",
        "sharingRateLimitExceeded",
        "teamDriveRateLimitExceeded",
        "downloadQuotaExceeded",
    }
)

# Fields every call asks for; keeps the metadata round trips off the wire.
_METADATA_FIELDS = "id,name,mimeType,webViewLink,webContentLink"


def parse_credentials(credentials_file: str | None, credentials_json: str | None) -> dict[str, Any]:
    """Loads the service account key from a path or from raw JSON contents.

    Exactly one source must be given. The returned dict is the key file as-is
    (``client_email``, ``private_key``, ``token_uri``, ...), validated only for
    the fields ``from_service_account_info`` actually needs so a truncated or
    pasted-with-quotes key fails here with an actionable message.
    """
    if credentials_file and credentials_json:
        raise ValueError(
            "GDRIVE_SERVICE_ACCOUNT_FILE and GDRIVE_SERVICE_ACCOUNT_JSON are mutually "
            "exclusive; configure exactly one."
        )
    if not credentials_file and not credentials_json:
        raise ValueError(
            "Google Drive needs a service account key: set GDRIVE_SERVICE_ACCOUNT_FILE "
            "to the downloaded JSON path, or GDRIVE_SERVICE_ACCOUNT_JSON to its raw contents."
        )

    if credentials_file:
        path = Path(os.path.expanduser(credentials_file))
        if not path.is_file():
            raise ValueError(f"GDRIVE_SERVICE_ACCOUNT_FILE does not exist: {path}")
        raw = path.read_text(encoding="utf-8")
    else:
        assert credentials_json is not None
        raw = credentials_json.strip()
        # Tolerate a path pasted into the raw-JSON variable by mistake.
        if raw.startswith("/") and raw.endswith(".json") and Path(raw).is_file():
            raw = Path(raw).read_text(encoding="utf-8")

    try:
        info = json.loads(raw)
    except ValueError as err:
        raise ValueError(f"Service account key is not valid JSON: {err}") from err
    if not isinstance(info, dict):
        raise ValueError("Service account key must be a JSON object, not a list or string.")

    key_type = info.get("type")
    if key_type and key_type != "service_account":
        raise ValueError(
            f"GDRIVE_SERVICE_ACCOUNT_JSON must be a service_account key (got type={key_type!r}). "
            "OAuth 'authorized_user' client secrets are not supported."
        )
    missing = [field for field in ("client_email", "private_key", "token_uri") if not info.get(field)]
    if missing:
        raise ValueError(
            "Service account key is missing required field(s): "
            + ", ".join(missing)
            + ". Download a fresh JSON key from Google Cloud Console > IAM & Admin > "
            "Service Accounts > Keys."
        )
    return info


def http_status(err: Exception) -> int:
    """HTTP status of an ``HttpError`` (0 when it carries no response).

    Reads the attribute first (``httplib2.Response``) and falls back to mapping
    access, because ``HttpError.resp`` is a dict subclass in some code paths.
    """
    resp = getattr(err, "resp", None)
    if resp is None:
        return 0
    status = getattr(resp, "status", None)
    if status is None and hasattr(resp, "get"):
        status = resp.get("status")
    try:
        return int(status or 0)
    except (TypeError, ValueError):
        return 0

def error_reason(err: Exception) -> str:
    """First ``error.errors[].reason`` from an ``HttpError`` body ('' if absent)."""
    content = getattr(err, "content", b"") or b""
    if isinstance(content, bytes):
        content = content.decode("utf-8", "replace")
    if not content:
        return ""
    try:
        payload = json.loads(content)
    except ValueError:
        return ""
    errors = ((payload or {}).get("error") or {}).get("errors") or []
    for entry in errors:
        reason = (entry or {}).get("reason")
        if reason:
            return str(reason)
    return ""


def backoff_seconds(err: Exception, attempt: int) -> float | None:
    """Google's truncated exponential backoff, or ``None`` when we must re-raise.

    Only throttling gets retried: a 404 or a permission-denied 403 will never
    become a success, so raising immediately keeps failures visible.
    """
    status = http_status(err)
    reason = error_reason(err)
    if status == 429:
        return min(MAX_BACKOFF_SEC, float(2**attempt))
    if status != 403:
        return None
    if reason and reason not in QUOTA_REASONS:
        return None
    return min(MAX_BACKOFF_SEC, float(2**attempt))


def split_scopes(raw: str | Sequence[str] | None) -> list[str]:
    """Normalises ``GDRIVE_SCOPES`` (comma-separated string or list) to a list."""
    if raw is None:
        return list(DEFAULT_SCOPES)
    if isinstance(raw, str):
        scopes = [part.strip() for part in raw.split(",")]
        return [scope for scope in scopes if scope] or list(DEFAULT_SCOPES)
    scopes = [str(part).strip() for part in raw]
    return [scope for scope in scopes if scope] or list(DEFAULT_SCOPES)


def guess_content_type(
    filename: str,
    content_type: str | None = None,
    media_type: str | None = None,
) -> str:
    """Best-effort mimetype for the upload metadata.

    Drive stores this as metadata only (it does not gate delivery), so the
    precedence is: explicit ``content_type``, then the caller's ``media_type``
    category when ``mimetypes`` cannot confirm it (``.xyz`` is a real chemistry
    extension, not a video one), then the filename guess, then octet-stream.
    """
    if content_type:
        return content_type
    guessed, _ = mimetypes.guess_type(filename)
    hint = (media_type or "").strip().lower()
    if hint == "video" and not (guessed or "").startswith("video/"):
        return "video/mp4"
    if hint == "image" and not (guessed or "").startswith("image/"):
        return "image/jpeg"
    return guessed or "application/octet-stream"


def safe_cache_name(file_id: str, remote_name: str) -> str:
    """Builds a local cache filename carrying both the file id and an extension.

    The file id alone is extension-less (``1AbC...``), which OpenCV and ffprobe
    cannot sniff, so the extension is taken from Drive's metadata name and
    anything unrecognised is normalised to ``.mp4`` — the same rule the shared
    ``_remote_cache`` helper applies to HTTP URLs.
    """
    stem = remote_name.replace("\\", "/").split("/")[-1].strip() or file_id
    suffix = Path(stem).suffix.lower()
    if suffix not in KNOWN_MEDIA_EXTENSIONS:
        suffix = ".mp4"
    return f"{file_id}{suffix}"


class GoogleDriveStorageBackend:
    """Stores raw footage files in Google Drive under a service account.

    Every mutating/read call passes ``supportsAllDrives=True`` so the same
    backend works against a personal folder, a Shared Drive, or a folder inside
    one — without it, Shared Drive files are invisible to all four CRUD methods.
    """

    def __init__(
        self,
        credentials_file: str | None = None,
        credentials_json: str | None = None,
        folder_id: str | None = None,
        drive_id: str | None = None,
        scopes: str | Sequence[str] | None = None,
        url_template: str | None = None,
        cache_dir: str | None = None,
    ):
        info = parse_credentials(credentials_file, credentials_json)
        if not (folder_id or drive_id):
            raise ValueError(
                "Google Drive needs a destination: set GDRIVE_FOLDER_ID to a folder shared "
                "with the service account's client_email, or GDRIVE_DRIVE_ID to a Shared Drive "
                "the service account has been added to."
            )
        if service_account is None or build is None:
            raise ImportError(
                "google-api-python-client and google-auth are required for "
                "GoogleDriveStorageBackend. Install with: pip install \"footage-engine[gdrive]\""
            )

        self.folder_id = folder_id
        self.drive_id = drive_id
        self.scopes = split_scopes(scopes)
        self.url_template = url_template or "https://drive.google.com/uc?export=download&id={file_id}"

        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=list(self.scopes)
        )
        self.service = build(
            "drive", "v3", credentials=credentials, cache_discovery=False
        )

        # Memoises metadata names so a cache miss costs one round trip, not one
        # per chunk probe.
        self._name_cache: dict[str, str] = {}

        self.cache_dir = Path(cache_dir or os.path.join(tempfile.gettempdir(), "footage_engine_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- internal helpers (underscore-prefixed: the contract test asserts the
    #    class exposes exactly the six protocol methods) ----------------------

    def _run(self, call: Callable[[], Any]) -> Any:
        """Runs ``call``, applying truncated exponential backoff on throttling.

        Google's own limits documentation prescribes catching
        ``rateLimitExceeded``/``userRateLimitExceeded`` and backing off
        exponentially with a ~32 s ceiling; every other ``HttpError`` propagates
        immediately so real failures stay visible.
        """
        attempt = 0
        while True:
            try:
                return call()
            except HttpError as err:
                delay = backoff_seconds(err, attempt)
                if delay is None:
                    raise
                print(
                    f"    → Google Drive throttled (status={http_status(err)}, "
                    f"reason={error_reason(err) or 'n/a'}), backing off {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
                attempt += 1

    def _make_public(self, file_id: str) -> None:
        """Grants ``anyone`` reader access so ``get_url`` is playable.

        Fails loudly on refusal: a silently-private file would produce a broken
        ``storage_url`` that only surfaces when someone tries to play it.
        """
        request = self.service.permissions().create(
            fileId=file_id,
            body={"role": "reader", "type": "anyone"},
            supportsAllDrives=True,
            fields="id,type,role",
        )
        try:
            self._run(request.execute)
        except HttpError as err:
            status = http_status(err)
            raise PermissionError(
                f"Google Drive refused to make file {file_id} publicly readable "
                f"(HTTP {status}: {error_reason(err) or 'unknown'}). This backend relies on "
                "anyone-with-the-link access for playback. On Google Workspace an admin may "
                "have disabled 'Anyone with the link' sharing for the domain — enable it for "
                "the target folder/drive, or switch STORAGE_BACKEND back to local/imagekit/"
                "cloudinary."
            ) from err


    def save_file(
        self,
        file_data: bytes | BinaryIO,
        filename: str,
        content_type: str | None = None,
        media_type: str | None = None,
    ) -> str:
        """Uploads via a resumable session and returns the Drive ``fileId``.

        Resumable rather than multipart: footage files run to hundreds of MB and
        multipart uploads the whole payload in one request with no retry. The
        returned id is the opaque key every other method accepts — Drive ids
        carry no extension, so ``get_local_path`` recovers one from metadata.
        """
        if MediaIoBaseUpload is None:  # pragma: no cover - guarded in __init__
            raise ImportError("google-api-python-client is required for uploads.")

        name = filename.replace("\\", "/").strip("/") or "asset"
        body: dict[str, Any] = {"name": name}
        if self.folder_id:
            body["parents"] = [self.folder_id]

        payload = io.BytesIO(file_data) if isinstance(file_data, (bytes, bytearray)) else file_data
        if hasattr(payload, "seek"):
            payload.seek(0)

        media = MediaIoBaseUpload(
            payload,
            mimetype=guess_content_type(name, content_type, media_type),
            chunksize=UPLOAD_CHUNK_SIZE,
            resumable=True,
        )
        request = self.service.files().create(
            body=body,
            media_body=media,
            supportsAllDrives=True,
            fields=_METADATA_FIELDS,
            **({"driveId": self.drive_id} if self.drive_id and not self.folder_id else {}),
        )

        # ``next_chunk`` (not ``execute``) is the resumable protocol: each call
        # pushes one chunk and returns ``body=None`` until the last one lands.
        # Retrying a chunk through ``_run`` is safe — googleapiclient tracks the
        # committed offset and issues the matching ``Range`` header.
        response = None
        while response is None:
            _, response = self._run(request.next_chunk)

        file_id = (response or {}).get("id")
        if not file_id:
            raise RuntimeError(f"Google Drive upload returned no file id for {name!r}: {response!r}")

        # Public by decision: the URL handed to search results must be playable
        # without credentials, so every upload carries an anyone-reader grant.
        self._make_public(file_id)
        return file_id


    def get_url(self, storage_path: str) -> str:
        """Returns a directly fetchable link for the file.

        Passes through references this backend does not own (YouTube, plain
        HTTP, ``file://``) so a mixed ``media_items`` table resolves either way.
        For an owned file the URL is built from ``GDRIVE_URL_TEMPLATE`` — the
        documented ``uc?export=download`` link by default. It serves small files
        directly; Drive may insert a virus-scan confirmation page past roughly
        100 MB, so set the template to
        ``https://lh3.googleusercontent.com/d/{file_id}``` if large assets do not
        stream in your tenant.
        """
        if storage_path.startswith(("http://", "https://", "file://")):
            return storage_path
        return self.url_template.format(file_id=storage_path)

    def _file_name(self, file_id: str) -> str:
        """Fetches (and memoises) the remote name used to extend the cache file."""
        cached = self._name_cache.get(file_id)
        if cached is not None:
            return cached
        response = self._run(
            lambda: self.service.files().get(
                fileId=file_id, fields="name", supportsAllDrives=True
            ).execute()
        )
        name = str((response or {}).get("name") or "")
        self._name_cache[file_id] = name
        return name

    def _download(self, file_id: str, destination: BinaryIO) -> None:
        """Streams file content into ``destination`` using the bearer token.

        Authenticated on purpose: ``get_url`` points at an unauthenticated
        download endpoint, which is fine for browsers but not something to hang
        byte-exact reads on (and it is the only way to read a file the public
        grant was refused for).
        """
        if MediaIoBaseDownload is None:  # pragma: no cover - guarded in __init__
            raise ImportError("google-api-python-client is required for downloads.")
        request = self.service.files().get(
            fileId=file_id, alt="media", supportsAllDrives=True
        )
        downloader = MediaIoBaseDownload(destination, request, chunksize=DOWNLOAD_CHUNK_SIZE)
        done = False
        while not done:
            _, done = self._run(downloader.next_chunk)

    def get_file(self, storage_path: str) -> bytes:
        """Returns the whole file as bytes.

        Materialises the asset in memory — fine for clips and images, but prefer
        ``get_local_path`` for multi-hundred-MB masters.
        """
        buffer = io.BytesIO()
        self._download(storage_path, buffer)
        return buffer.getvalue()


    def _cache_path_for(self, file_id: str) -> Path:
        """Locates an existing cached file, or computes the name for a new one.

        The glob comes first so a warm cache never spends an API call on
        metadata; only a miss fetches ``name`` to recover the extension.
        """
        for candidate in sorted(self.cache_dir.glob(f"{file_id}.*")):
            if candidate.suffix == ".tmp":
                continue
            if candidate.stat().st_size >= MIN_VALID_FILE_BYTES:
                return candidate
        return self.cache_dir / safe_cache_name(file_id, self._file_name(file_id))

    def get_local_path(self, storage_path: str) -> str:
        """Downloads into the shared cache and returns the local path.

        YouTube links, plain HTTP URLs and ``file://`` references are handled by
        the shared resolver first; anything left is a Drive file id owned by this
        backend. The cached name carries the extension Drive's metadata reports,
        which is what lets OpenCV and ffprobe sniff the container.
        """
        resolved = resolve_remote_reference(storage_path, self.cache_dir)
        if resolved is not None:
            return resolved

        cached_file = self._cache_path_for(storage_path)
        if not cached_file.exists() or cached_file.stat().st_size < MIN_VALID_FILE_BYTES:
            tmp_file = cached_file.with_suffix(cached_file.suffix + ".tmp")
            try:
                with open(tmp_file, "wb") as handle:
                    self._download(storage_path, handle)
                tmp_file.replace(cached_file)
            finally:
                if tmp_file.exists() and not cached_file.exists():
                    tmp_file.unlink()
        return str(cached_file)

    def exists(self, storage_path: str) -> bool:
        """True when Drive resolves the id; 404 is the only tolerated failure."""
        if not storage_path:
            return False
        try:
            self._run(
                lambda: self.service.files().get(
                    fileId=storage_path, fields="id", supportsAllDrives=True
                ).execute()
            )
            return True
        except HttpError as err:
            if http_status(err) == 404:
                return False
            raise

    def delete_file(self, storage_path: str) -> bool:
        """Trashes nothing: deletes outright, and confirms the file is gone.

        Returning True only after a follow-up 404 honours the contract's "never
        True as a placeholder" rule — Drive's delete answers 204 for success and
        404 for an already-absent file, and the confirmation read is what makes
        a shared-drive permission problem distinguishable from a no-op.
        """
        if not storage_path:
            return False
        try:
            self._run(
                lambda: self.service.files().delete(
                    fileId=storage_path, supportsAllDrives=True
                ).execute()
            )
        except HttpError as err:
            if http_status(err) == 404:
                return False
            raise
        self._name_cache.pop(storage_path, None)
        return not self.exists(storage_path)


__all__ = [
    "DEFAULT_SCOPES",
    "DOWNLOAD_CHUNK_SIZE",
    "MAX_BACKOFF_SEC",
    "QUOTA_REASONS",
    "UPLOAD_CHUNK_SIZE",
    "GoogleDriveStorageBackend",
    "backoff_seconds",
    "error_reason",
    "guess_content_type",
    "http_status",
    "parse_credentials",
    "safe_cache_name",
    "split_scopes",
]

