"""Tests for Ingestion sources and pre-spend deduplication."""

from unittest.mock import MagicMock, patch
import pytest
from footage_engine.models.media import MediaStatus
from footage_engine.sources.base import Candidate
from footage_engine.sources.protocol import IterableURLProvider


def test_ingest_new_and_duplicate_prespend(test_orchestrator, test_settings, monkeypatch):
    test_settings.UPLOAD_RAW_TO_STORAGE = True
    test_url = "https://cdn.example.com/sample_video.mp4"
    mock_bytes = b"\x00\x00\x00\x18ftypmp42"

    download_count = 0

    def mock_get(url, **kwargs):
        nonlocal download_count
        download_count += 1
        response = MagicMock()
        response.status_code = 200
        response.content = mock_bytes
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr("requests.get", mock_get)

    # 1. First ingest - should download and insert new row
    item1 = test_orchestrator.ingest(
        source_url=test_url,
        provider="manual",
        source_id="manual_001",
        duration_sec=12.4,
        resolution="1280x720",
    )

    assert item1 is not None
    assert item1.status == MediaStatus.PENDING
    assert item1.source_url == test_url
    assert download_count == 1
    assert test_orchestrator.storage.exists(item1.storage_path)

    # 2. Second ingest with SAME url - must return existing item immediately WITHOUT downloading
    item2 = test_orchestrator.ingest(
        source_url=test_url,
        provider="manual",
        source_id="manual_001",
    )

    assert item2.id == item1.id
    # download_count MUST still be 1 (zero network request performed)
    assert download_count == 1

    # 3. Third ingest with SAME provider + source_id but slightly different URL - must also dedup
    item3 = test_orchestrator.ingest(
        source_url=test_url + "?query=param",
        provider="manual",
        source_id="manual_001",
    )

    assert item3.id == item1.id
    assert download_count == 1


def test_search_and_ingest(test_orchestrator, monkeypatch):
    mock_bytes = b"fake-video-payload"

    def mock_get(url, **kwargs):
        response = MagicMock()
        response.status_code = 200
        response.content = mock_bytes
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr("requests.get", mock_get)

    mock_candidates = [
        Candidate(
            provider="pixabay",
            source_id="p101",
            source_url="https://pixabay.com/v101.mp4",
            duration_sec=20.0,
            resolution="1920x1080",
            license_type="pixabay_free",
        ),
        Candidate(
            provider="pixabay",
            source_id="p102",
            source_url="https://pixabay.com/v102.mp4",
            duration_sec=35.0,
            resolution="1920x1080",
            license_type="pixabay_free",
        ),
    ]

    mock_adapter = MagicMock()
    mock_adapter.search.return_value = mock_candidates
    test_orchestrator.adapters["pixabay"] = mock_adapter

    results = test_orchestrator.search_and_ingest(keyword="ocean sunset", provider="pixabay", max_results=2)

    assert len(results) == 2
    assert results[0].source_id == "p101"
    assert results[1].source_id == "p102"
    assert results[0].status == MediaStatus.PENDING


def test_ingest_url_list(test_orchestrator, monkeypatch):
    mock_bytes = b"video-data"

    def mock_get(url, **kwargs):
        response = MagicMock()
        response.status_code = 200
        response.content = mock_bytes
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr("requests.get", mock_get)

    urls = [
        "https://cdn.example.com/clip1.mp4",
        "https://cdn.example.com/clip2.mp4",
    ]
    items = test_orchestrator.ingest_url_list(urls, provider="direct_test")
    assert len(items) == 2
    assert items[0].provider == "direct_test"
    assert items[1].provider == "direct_test"


def test_ingest_from_provider_protocol(test_orchestrator, monkeypatch):
    mock_bytes = b"external-db-payload"

    def mock_get(url, **kwargs):
        response = MagicMock()
        response.status_code = 200
        response.content = mock_bytes
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr("requests.get", mock_get)

    provider_data = [
        ("https://ext-db.com/video_a.mp4", {"id": "ext_001", "category": "nature"}),
        ("https://ext-db.com/video_b.mp4", {"id": "ext_002", "category": "tech"}),
    ]
    url_provider = IterableURLProvider(provider_data)

    items = test_orchestrator.ingest_from_provider(url_provider, provider_name="stock_insight_ai")
    assert len(items) == 2
    assert items[0].source_id == "ext_001"
    assert items[0].provider == "stock_insight_ai"
    assert items[0].item_metadata.get("category") == "nature"
    assert items[1].source_id == "ext_002"


def test_provider_orientation_and_media_type_handling(monkeypatch):
    from footage_engine.sources.pixabay import PixabayAdapter
    from footage_engine.sources.pexels import PexelsAdapter
    from footage_engine.sources.coverr import CoverrAdapter

    # 1. Pixabay video + orientation
    pixabay = PixabayAdapter(api_key="fake_key")
    captured_pixabay_params = {}

    def mock_pixabay_get(url, params=None, **kwargs):
        nonlocal captured_pixabay_params
        captured_pixabay_params = params
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"hits": []}
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr("requests.get", mock_pixabay_get)
    pixabay.search(keyword="sea", max_results=5, orientation="landscape", media_type="video")
    assert captured_pixabay_params["orientation"] == "horizontal"
    assert captured_pixabay_params["video_type"] == "all"

    # 2. Pexels video + portrait orientation
    pexels = PexelsAdapter(api_key="fake_key")
    captured_pexels_params = {}

    def mock_pexels_get(url, params=None, **kwargs):
        nonlocal captured_pexels_params
        captured_pexels_params = params
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"videos": []}
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr("requests.get", mock_pexels_get)
    pexels.search(keyword="sea", max_results=5, orientation="vertical", media_type="video")
    assert captured_pexels_params["orientation"] == "portrait"

    # 3. Coverr client-side orientation filtering
    coverr = CoverrAdapter(api_key="fake_key")
    mock_coverr_response = {
        "hits": [
            {
                "id": "cov_horiz",
                "is_vertical": False,
                "aspect_ratio": "16:9",
                "urls": {"mp4": "https://cdn.coverr.co/horiz.mp4"},
                "duration": 10.0,
                "max_width": 1920,
                "max_height": 1080,
            },
            {
                "id": "cov_vert",
                "is_vertical": True,
                "aspect_ratio": "9:16",
                "urls": {"mp4": "https://cdn.coverr.co/vert.mp4"},
                "duration": 15.0,
                "max_width": 1080,
                "max_height": 1920,
            },
        ]
    }

    def mock_coverr_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = mock_coverr_response
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr("requests.get", mock_coverr_get)
    # Search landscape only
    cands_horiz = coverr.search(keyword="nature", max_results=5, orientation="landscape")
    assert len(cands_horiz) == 1
    assert cands_horiz[0].source_id == "cov_horiz"

    # Search portrait only
    cands_vert = coverr.search(keyword="nature", max_results=5, orientation="portrait")
    assert len(cands_vert) == 1
    assert cands_vert[0].source_id == "cov_vert"
