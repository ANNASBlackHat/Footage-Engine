"""Pytest configuration and fixtures."""

import os
import shutil
import tempfile
import pytest

from footage_engine.config import Settings
from footage_engine.models.db import init_db
from footage_engine.storage.local import LocalStorageBackend
from footage_engine.orchestrator import Orchestrator


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_settings(temp_dir):
    db_path = os.path.join(temp_dir, "test.db")
    storage_path = os.path.join(temp_dir, "storage")
    return Settings(
        DATABASE_URL=f"sqlite:///{db_path}",
        STORAGE_BACKEND="local",
        LOCAL_STORAGE_DIR=storage_path,
        PIXABAY_API_KEY="test_pixabay_key",
        PEXELS_API_KEY="test_pexels_key",
    )


@pytest.fixture
def test_storage(test_settings):
    return LocalStorageBackend(base_dir=test_settings.LOCAL_STORAGE_DIR)


@pytest.fixture
def test_orchestrator(test_settings, test_storage):
    init_db(test_settings.DATABASE_URL)
    return Orchestrator(
        settings=test_settings,
        storage=test_storage,
        database_url=test_settings.DATABASE_URL,
    )
