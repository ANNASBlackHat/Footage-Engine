"""Tests for ``to_stored_reference`` -- what gets persisted as ``storage_path``.

``storage_path`` is polymorphic: it holds either a backend-owned key or an
absolute URL, and rows written under different backends share the table. The
rule under test is that a saved asset is recorded as durably as possible
(an absolute URL pinned to the endpoint that holds the bytes) without breaking
the read path of the backends whose key is already portable or whose reads must
stay authenticated.

Reads of either shape are covered by the existing contract/backend suites; this
module covers only the write-side decision.
"""

from unittest.mock import MagicMock

import pytest

from footage_engine.storage import to_stored_reference
from footage_engine.storage.base import StorageBackend

KEY = "footage_engine/raw/chunks_abc12345_def67890.mp4"
FOREIGN_URL = "https://ik.imagekit.io/some_other_account/raw/clip.mp4"


def _stub_backend(url_for_key: str):
    """A backend stand-in whose ``get_url`` mirrors the real prefixing rule."""
    backend = MagicMock(spec=StorageBackend)
    backend.get_url.side_effect = lambda key: (
        key
        if key.startswith(("http://", "https://", "file://"))
        else f"{url_for_key}/{key.lstrip('/')}"
    )
    return backend


def test_absolute_url_is_stored_for_a_prefixing_backend():
    """The whole point: a new row must pin the endpoint holding the bytes."""
    backend = _stub_backend("https://ik.imagekit.io/account_a")

    stored = to_stored_reference(backend, KEY)

    assert stored == f"https://ik.imagekit.io/account_a/{KEY}"
    assert stored.startswith("https://")


def test_stored_url_round_trips_through_get_url():
    """get_url must not re-prefix an already-absolute reference."""
    backend = _stub_backend("https://ik.imagekit.io/account_a")
    stored = to_stored_reference(backend, KEY)

    assert backend.get_url(stored) == stored


def test_a_foreign_accounts_url_is_kept_as_is():
    """A row written under another account must resolve unchanged."""
    backend = _stub_backend("https://ik.imagekit.io/account_b")

    assert to_stored_reference(backend, FOREIGN_URL) == FOREIGN_URL
    backend.get_url.assert_not_called()


def test_file_url_is_kept_as_is():
    backend = _stub_backend("https://ik.imagekit.io/account_a")

    assert to_stored_reference(backend, "file:///tmp/clip.mp4") == "file:///tmp/clip.mp4"


@pytest.mark.parametrize("value", ["", None])
def test_empty_reference_is_returned_unchanged(value):
    """A falsy key (failed upload) must not become the string 'None'."""
    backend = _stub_backend("https://ik.imagekit.io/account_a")

    assert to_stored_reference(backend, value) == value


# -- the deliberate exception that needs a real instance -------------------------
# The Drive case lives in test_gdrive_storage.py, where the in-memory SDK fakes
# already exist; special-casing is by class, so a stub would miss the branch.


def test_local_backend_keeps_its_relative_path(tmp_path):
    """An absolute file:// would bake the writer's filesystem into a shared DB."""
    from footage_engine.storage.local import LocalStorageBackend

    backend = LocalStorageBackend(base_dir=str(tmp_path))

    stored = to_stored_reference(backend, KEY)

    assert stored == KEY
    assert backend.get_url(stored).startswith("file://")