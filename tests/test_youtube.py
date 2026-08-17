"""Tests for YouTube source adapter and ingestion."""

import os
from unittest.mock import MagicMock, patch
import pytest

from footage_engine.models.db import init_db, get_db_session
from footage_engine.models.media import MediaItem, MediaType
from footage_engine.orchestrator import Orchestrator
from footage_engine.sources.direct import DirectURLAdapter
from footage_engine.sources.youtube import (
    YouTubeAdapter,
    extract_youtube_video_id,
    is_youtube_url,
    normalize_youtube_url,
)
from footage_engine.storage.local import LocalStorageBackend


def test_youtube_url_detection():
    # Valid YouTube URLs
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=10s")
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert is_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
    assert is_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ")
    assert is_youtube_url("http://youtube.com/v/dQw4w9WgXcQ")

    # Invalid / non-YouTube URLs
    assert not is_youtube_url("https://vimeo.com/12345678")
    assert not is_youtube_url("https://example.com/video.mp4")
    assert not is_youtube_url("file:///path/to/video.mp4")
    assert not is_youtube_url("")
    assert not is_youtube_url(None)


def test_youtube_video_id_extraction():
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("https://www.youtube.com/shorts/abcdef12345") == "abcdef12345"
    assert extract_youtube_video_id("https://example.com/not_youtube") is None


def test_youtube_url_normalization():
    assert normalize_youtube_url("https://youtu.be/dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert normalize_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert normalize_youtube_url("https://example.com/video.mp4") == "https://example.com/video.mp4"


def test_youtube_adapter_url_to_candidate():
    mock_info = {
        "id": "dQw4w9WgXcQ",
        "title": "Sample YouTube Video",
        "duration": 212.0,
        "width": 1920,
        "height": 1080,
        "uploader": "Test Creator",
        "channel_id": "UC12345",
        "view_count": 1000000,
        "like_count": 50000,
        "description": "A demo video description.",
        "tags": ["demo", "video"],
        "license": "Creative Commons Attribution license (reuse allowed)",
        "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
    }

    adapter = YouTubeAdapter()
    with patch.object(adapter, "extract_info", return_value=mock_info):
        cand = adapter.url_to_candidate("https://youtu.be/dQw4w9WgXcQ")
        assert cand.provider == "youtube"
        assert cand.source_id == "dQw4w9WgXcQ"
        assert cand.source_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert cand.duration_sec == 212.0
        assert cand.resolution == "1920x1080"
        assert cand.license_type == "creative_commons"
        assert cand.metadata["title"] == "Sample YouTube Video"
        assert cand.metadata["uploader"] == "Test Creator"


def test_youtube_adapter_search():
    mock_search_results = {
        "entries": [
            {
                "id": "vid1",
                "title": "Ocean Sunset",
                "duration": 60.0,
                "width": 1280,
                "height": 720,
                "uploader": "NatureChannel",
                "view_count": 1000,
                "description": "Waves at sunset",
                "thumbnail": "https://example.com/thumb1.jpg",
            },
            {
                "id": "vid2",
                "title": "Ship in Storm",
                "duration": 45.0,
                "width": 1920,
                "height": 1080,
                "uploader": "SeaVids",
                "view_count": 5000,
                "description": "Rough seas",
                "thumbnail": "https://example.com/thumb2.jpg",
            },
        ]
    }

    adapter = YouTubeAdapter()
    with patch("yt_dlp.YoutubeDL") as MockYDL:
        instance = MockYDL.return_value.__enter__.return_value
        instance.extract_info.return_value = mock_search_results

        candidates = adapter.search("ocean", max_results=2)
        assert len(candidates) == 2
        assert candidates[0].source_id == "vid1"
        assert candidates[0].source_url == "https://www.youtube.com/watch?v=vid1"
        assert candidates[1].source_id == "vid2"


def test_direct_url_adapter_with_youtube():
    mock_info = {
        "id": "dQw4w9WgXcQ",
        "title": "Sample YouTube Video",
        "duration": 120.0,
        "width": 1920,
        "height": 1080,
        "uploader": "Direct Test",
    }
    with patch.object(YouTubeAdapter, "extract_info", return_value=mock_info):
        candidate = DirectURLAdapter.url_to_candidate("https://youtu.be/dQw4w9WgXcQ")
        assert candidate.provider == "youtube"
        assert candidate.source_id == "dQw4w9WgXcQ"
        assert candidate.source_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert candidate.media_type == "video"


def test_orchestrator_ingest_youtube(test_orchestrator):
    mock_info = {
        "id": "dQw4w9WgXcQ",
        "title": "Ingested YouTube Video",
        "duration": 180.0,
        "width": 1920,
        "height": 1080,
    }
    with patch.object(YouTubeAdapter, "extract_info", return_value=mock_info):
        item = test_orchestrator.ingest(
            source_url="https://youtu.be/dQw4w9WgXcQ",
            provider="manual",  # should auto-route to youtube
        )
        assert item.provider == "youtube"
        assert item.source_id == "dQw4w9WgXcQ"
        assert item.source_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert item.media_type == MediaType.VIDEO

        # Test pre-spend deduplication on identical YouTube URL
        duplicate = test_orchestrator.ingest(
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            provider="youtube",
        )
        assert duplicate.id == item.id


def test_local_storage_youtube_caching(temp_dir):
    storage = LocalStorageBackend(base_dir=temp_dir)
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def fake_download(url, target_path):
        with open(target_path, "wb") as f:
            f.write(b"fake_youtube_video_content_bytes_mp4" * 100)
        return target_path

    with patch.object(YouTubeAdapter, "download_to_path", side_effect=fake_download):
        cached_path = storage.get_local_path(yt_url)
        assert os.path.exists(cached_path)
        assert cached_path.endswith("youtube_dQw4w9WgXcQ.mp4")
        with open(cached_path, "rb") as f:
            content = f.read()
        assert b"fake_youtube_video_content" in content


def test_youtube_cookie_resolution_file(temp_dir):
    from footage_engine.sources.youtube import resolve_cookie_source
    cookie_path = os.path.join(temp_dir, "my_cookies.txt")
    with open(cookie_path, "w") as f:
        f.write("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t1750000000\tSID\t12345\n")

    resolved = resolve_cookie_source(cookie_path)
    assert resolved == cookie_path
    assert os.path.isfile(resolved)


def test_youtube_cookie_resolution_url(temp_dir):
    from footage_engine.sources.youtube import resolve_cookie_source
    mock_resp = MagicMock()
    mock_resp.content = b"# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t1750000000\tHSID\t67890\n"

    with patch("requests.get", return_value=mock_resp):
        resolved = resolve_cookie_source("https://example.com/cookies.txt", cache_dir=temp_dir)
        assert resolved is not None
        assert os.path.isfile(resolved)
        with open(resolved, "rb") as f:
            data = f.read()
        assert b"HSID" in data


def test_youtube_cookie_resolution_raw_and_base64():
    import base64
    from footage_engine.sources.youtube import resolve_cookie_source

    raw_cookies = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t1750000000\tSSID\txyz999\n"
    resolved_raw = resolve_cookie_source(raw_cookies)
    assert resolved_raw is not None
    assert os.path.isfile(resolved_raw)
    with open(resolved_raw, "r") as f:
        assert "SSID" in f.read()

    b64_cookies = base64.b64encode(raw_cookies.encode("utf-8")).decode("utf-8")
    resolved_b64 = resolve_cookie_source(b64_cookies)
    assert resolved_b64 is not None
    assert os.path.isfile(resolved_b64)
    with open(resolved_b64, "r") as f:
        assert "SSID" in f.read()


def test_youtube_adapter_with_browser_cookies():
    adapter = YouTubeAdapter(cookies_from_browser="chrome")
    opts = adapter._get_ydl_opts()
    assert opts.get("cookiesfrombrowser") == ("chrome",)
