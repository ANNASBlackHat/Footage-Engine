"""Tests for Qwen captioning (parser, chunk selection, DB writes, config)."""

import importlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# The Qwen caption module lives at the repo root; mirror the script's bootstrap.
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from footage_engine.captions.qwen_captioner import parse_model_response
from footage_engine.config import Settings
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaType


# ---------------------------------------------------------------------------
# parse_model_response
# ---------------------------------------------------------------------------

def test_parse_response_with_headers():
    raw = (
        "CAPTION: A white shark glides through murky open water, camera tracking "
        "laterally, dim diffuse light from above.\n"
        "TAGS: shark, underwater, murky, open ocean"
    )
    caption, tags = parse_model_response(raw)
    assert "shark" in caption
    assert caption == "A white shark glides through murky open water, camera tracking laterally, dim diffuse light from above."
    assert tags == ["shark", "underwater", "murky", "open ocean"]


def test_parse_response_falls_back_to_raw_when_no_headers():
    raw = "A quiet coastal scene with waves lapping the shore at dawn."
    caption, tags = parse_model_response(raw)
    assert caption == raw
    # Tags derived from the caption's longest words.
    assert "coastal" in tags or "waves" in tags


def test_parse_response_case_insensitive_headers():
    raw = "caption: A submarine descends into a trench\ntags: submarine, trench, deep sea"
    caption, tags = parse_model_response(raw)
    assert caption == "A submarine descends into a trench"
    assert tags == ["submarine", "trench", "deep sea"]


def test_parse_response_empty_input():
    caption, tags = parse_model_response("")
    assert caption == ""
    assert tags == []


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_caption_defaults():
    cfg = Settings(DATABASE_URL="sqlite:///:memory:", STORAGE_BACKEND="local")
    assert cfg.QWEN_CAPTION_MODEL == "Qwen/Qwen3-VL-2B-Instruct"
    assert cfg.QWEN_CAPTION_FRAMES == 3
    assert cfg.QWEN_CAPTION_DEVICE == "auto"


def test_config_caption_overridable():
    cfg = Settings(
        DATABASE_URL="sqlite:///:memory:",
        STORAGE_BACKEND="local",
        QWEN_CAPTION_MODEL="Qwen/Qwen3-VL-4B-Instruct",
        QWEN_CAPTION_FRAMES=5,
        QWEN_CAPTION_DEVICE="cuda",
    )
    assert cfg.QWEN_CAPTION_MODEL == "Qwen/Qwen3-VL-4B-Instruct"
    assert cfg.QWEN_CAPTION_FRAMES == 5
    assert cfg.QWEN_CAPTION_DEVICE == "cuda"


# ---------------------------------------------------------------------------
# QwenCaptioner lazy import
# ---------------------------------------------------------------------------

def test_captioner_module_imports_without_torch_or_transformers():
    """Importing the caption module must not fail just because the optional
    qwen extra is not installed — the model loads lazily inside the ctor."""
    module = importlib.reload(sys.modules["footage_engine.captions.qwen_captioner"])
    assert hasattr(module, "QwenCaptioner")


# ---------------------------------------------------------------------------
# scripts.caption_chunks chunk selection on a real SQLite DB
# ---------------------------------------------------------------------------

@pytest.fixture
def caption_db():
    """A throwaway SQLite DB with two media items and overlapping chunks."""
    tmp = tempfile.mkdtemp()
    db_url = f"sqlite:///{os.path.join(tmp, 'test.db')}"
    init_db(db_url)
    engine = create_engine(db_url)
    factory = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
    session = factory()

    item_a = MediaItem(id="item-a", provider="youtube", source_url="https://y/a", storage_path="a.mp4")
    item_b = MediaItem(id="item-b", provider="youtube", source_url="https://y/b", storage_path="b.mp4")
    item_img = MediaItem(
        id="item-img",
        provider="unsplash",
        source_url="https://u/x",
        storage_path="x.jpg",
        media_type=MediaType.IMAGE,
    )
    session.add_all([item_a, item_b, item_img])

    # item_a: one captioned, one uncaptioned
    session.add_all(
        [
            Chunk(media_item_id="item-a", start_ts=0.0, end_ts=10.0, caption="old", tags=["x"]),
            Chunk(media_item_id="item-a", start_ts=10.0, end_ts=20.0, caption=None, tags=None),
        ]
    )
    # item_b: two uncaptioned
    session.add_all(
        [
            Chunk(media_item_id="item-b", start_ts=0.0, end_ts=10.0, caption=None, tags=None),
            Chunk(media_item_id="item-b", start_ts=10.0, end_ts=20.0, caption=None, tags=None),
        ]
    )
    # item_img: a single still image chunk with no interval
    session.add(
        Chunk(
            media_item_id="item-img",
            start_ts=0.0,
            end_ts=None,
            media_type=MediaType.IMAGE,
            caption=None,
            tags=None,
        )
    )
    session.commit()
    session.close()
    yield db_url
    engine.dispose()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def test_find_next_uncaptioned_picks_most(caption_db):
    from scripts.caption_chunks import find_next_uncaptioned

    assert find_next_uncaptioned(caption_db) == "item-b"


def test_find_next_uncaptioned_respects_max_start_ts_and_limit(caption_db):
    from scripts.caption_chunks import fetch_chunk_intervals

    with get_db_session(caption_db) as session:
        rows = fetch_chunk_intervals(session, "item-b", max_start_ts=15.0, limit=1)
        assert len(rows) == 1
        assert rows[0].start_ts == 0.0


def test_find_next_uncaptioned_overwrite_includes_captured(caption_db):
    from scripts.caption_chunks import fetch_chunk_intervals

    with get_db_session(caption_db) as session:
        without_overwrite = fetch_chunk_intervals(session, "item-a")
        with_overwrite = fetch_chunk_intervals(session, "item-a", overwrite=True)
        # item-a has 1 captioned + 1 uncaptioned
        assert len(without_overwrite) == 1
        assert len(with_overwrite) == 2


def test_find_next_uncaptioned_dedups_overlapping_intervals(caption_db):
    """Two chunk rows with the same (start_ts, end_ts) collapse to one."""
    from scripts.caption_chunks import fetch_chunk_intervals

    with get_db_session(caption_db) as session:
        # Add a duplicate interval to item-b (same 0-10s, different embedding model)
        session.add(
            Chunk(
                media_item_id="item-b",
                start_ts=0.0,
                end_ts=10.0,
                embedding_model="qwen",
                caption=None,
            )
        )
        rows = fetch_chunk_intervals(session, "item-b")
        # 0-10s appears twice but dedups to one; 10-20s once
        assert len(rows) == 2
        starts = sorted(r.start_ts for r in rows)
        assert starts == [0.0, 10.0]


def test_process_item_writes_captions_to_db(caption_db):
    from scripts.caption_chunks import process_item

    # Stub out the heavy captioner + storage: capture what gets written.
    fake_captioner = MagicMock()
    fake_captioner.caption_chunk.return_value = ("a shark swims", ["shark", "ocean"])
    fake_storage = MagicMock()
    fake_storage.get_local_path.return_value = "/tmp/dummy.mp4"

    with (
        patch("scripts.caption_chunks.QwenCaptioner", return_value=fake_captioner),
        patch("scripts.caption_chunks.get_storage_backend", return_value=fake_storage),
    ):
        n = process_item(
            database_url=caption_db,
            media_item_id="item-a",
            model_id="fake",
            num_frames=3,
        )

    assert n == 1  # only the one uncaptioned interval
    with get_db_session(caption_db) as session:
        chunk = session.execute(
            select(Chunk).where(
                Chunk.media_item_id == "item-a",
                Chunk.start_ts == 10.0,
            )
        ).scalar_one()
        assert chunk.caption == "a shark swims"
        assert chunk.tags == ["shark", "ocean"]


def test_process_item_dry_run_writes_nothing(caption_db):
    from scripts.caption_chunks import process_item

    n = process_item(
        database_url=caption_db,
        media_item_id="item-a",
        model_id="fake",
        num_frames=3,
        dry_run=True,
    )
    assert n == 1  # reports how many *would* be done
    with get_db_session(caption_db) as session:
        chunk = session.execute(
            select(Chunk).where(
                Chunk.media_item_id == "item-a",
                Chunk.start_ts == 10.0,
            )
        ).scalar_one()
        assert chunk.caption is None


def test_fetch_chunk_intervals_includes_image_chunks(caption_db):
    from scripts.caption_chunks import fetch_chunk_intervals

    with get_db_session(caption_db) as session:
        rows = fetch_chunk_intervals(session, "item-img")
        assert len(rows) == 1
        assert rows[0].media_type == MediaType.IMAGE
        assert rows[0].end_ts is None


def test_process_item_captions_image_via_caption_image(caption_db):
    from scripts.caption_chunks import process_item

    fake_captioner = MagicMock()
    fake_captioner.caption_image.return_value = ("an octopus on coral", ["octopus"])
    fake_captioner.caption_chunk.side_effect = AssertionError("video path used")
    fake_storage = MagicMock()
    fake_storage.get_local_path.return_value = "/tmp/dummy.jpg"

    with (
        patch("scripts.caption_chunks.QwenCaptioner", return_value=fake_captioner),
        patch("scripts.caption_chunks.get_storage_backend", return_value=fake_storage),
    ):
        n = process_item(
            database_url=caption_db,
            media_item_id="item-img",
            model_id="fake",
            num_frames=3,
        )

    assert n == 1
    fake_captioner.caption_image.assert_called_once_with("/tmp/dummy.jpg")
    fake_captioner.caption_chunk.assert_not_called()
    with get_db_session(caption_db) as session:
        chunk = session.execute(
            select(Chunk).where(Chunk.media_item_id == "item-img")
        ).scalar_one()
        assert chunk.caption == "an octopus on coral"
        assert chunk.tags == ["octopus"]


def test_caption_image_loads_and_downscales_a_single_frame(tmp_path):
    from PIL import Image

    from footage_engine.captions.qwen_captioner import IMAGE_PROMPT_TEMPLATE, QwenCaptioner

    path = tmp_path / "still.png"
    Image.new("RGB", (1000, 500), (10, 20, 30)).save(path)

    captioner = object.__new__(QwenCaptioner)
    captioner.image_prompt_template = IMAGE_PROMPT_TEMPLATE
    seen = {}

    def fake_caption(frames, prompt):
        seen["frames"] = frames
        seen["prompt"] = prompt
        return "a still caption", ["still"]

    captioner._caption_frames = fake_caption

    assert captioner.caption_image(str(path)) == ("a still caption", ["still"])
    assert len(seen["frames"]) == 1
    assert seen["frames"][0].size == (448, 224)
    assert "still image" in seen["prompt"]
