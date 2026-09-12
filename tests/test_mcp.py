"""Tests for Footage Retrieval Engine MCP Server."""

import json
import os
import pytest
from sqlalchemy import select

from footage_engine.config import Settings
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.mcp.server import create_mcp_server
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaStatus, MediaType
from footage_engine.pipeline.processor import BatchProcessor
from footage_engine.storage.local import LocalStorageBackend
from footage_engine.vector.in_memory import InMemoryVectorStore


def _parse_tool_result(res):
    """Helper to extract parsed Python objects from MCP tool responses."""
    if res.content and hasattr(res.content[0], "text"):
        try:
            return json.loads(res.content[0].text)
        except json.JSONDecodeError:
            return res.content[0].text
    if res.structured_content is not None:
        if "result" in res.structured_content:
            return res.structured_content["result"]
        return res.structured_content
    raise ValueError(f"Unable to parse tool response: {res}")


@pytest.fixture
def mcp_env(temp_dir):
    """Fixture providing an isolated environment and configured MCP server."""
    db_path = os.path.join(temp_dir, "mcp_test.db")
    storage_path = os.path.join(temp_dir, "storage")
    os.makedirs(storage_path, exist_ok=True)

    settings = Settings(
        DATABASE_URL=f"sqlite:///{db_path}",
        STORAGE_BACKEND="local",
        LOCAL_STORAGE_DIR=storage_path,
        VECTOR_STORE="in_memory",
        PIXABAY_API_KEY="test_key",
        PEXELS_API_KEY="test_key",
    )
    init_db(settings.DATABASE_URL)

    storage = LocalStorageBackend(base_dir=settings.LOCAL_STORAGE_DIR)
    embedder = MockEmbedder(dimension=512)
    vector_store = InMemoryVectorStore()

    server = create_mcp_server(
        use_mock=True,
        settings=settings,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=settings.DATABASE_URL,
    )

    return {
        "server": server,
        "settings": settings,
        "storage": storage,
        "embedder": embedder,
        "vector_store": vector_store,
        "temp_dir": temp_dir,
    }


@pytest.mark.anyio
async def test_mcp_tool_registration(mcp_env):
    """Verify all expected MCP tools, resources, and prompts are properly registered."""
    server = mcp_env["server"]

    tools = await server.list_tools()
    tool_names = {t.name for t in tools}
    expected_tools = {
        "search_footage",
        "fine_localize_clip",
        "get_clip_details",
        "get_media_item_details",
        "ingest_url",
        "ingest_keywords",
        "process_pending_queue",
        "get_library_stats",
    }
    assert expected_tools.issubset(tool_names)

    resources = await server.list_resources()
    resource_uris = {r.uri for r in resources}
    assert "footage://stats" in resource_uris

    prompts = await server.list_prompts()
    prompt_names = {p.name for p in prompts}
    assert "broll-match-beat" in prompt_names


@pytest.mark.anyio
async def test_mcp_search_footage_and_clip_details(mcp_env):
    """Test search_footage and get_clip_details tools end-to-end."""
    server = mcp_env["server"]
    storage = mcp_env["storage"]
    settings = mcp_env["settings"]
    embedder = mcp_env["embedder"]
    vector_store = mcp_env["vector_store"]

    # Ingest test video
    storage.save_file(b"test video payload", "ocean.mp4")
    with get_db_session(settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="pexels",
            source_url="https://example.com/ocean.mp4",
            storage_path="ocean.mp4",
            duration_sec=20.0,
            resolution="1920x1080",
            license_type="pexels_free",
            status=MediaStatus.PENDING,
        )
        session.add(item)
        session.flush()
        media_id = item.id

    processor = BatchProcessor(
        settings=settings,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=settings.DATABASE_URL,
    )
    processor.process_all_pending()

    # Call search_footage tool
    res = await server.call_tool("search_footage", {"query": "ocean waves", "top_k": 5})
    search_data = _parse_tool_result(res)
    assert search_data["count"] == 1
    results = search_data["results"]
    assert len(results) == 1
    hit = results[0]
    assert hit["media_item_id"] == media_id
    assert hit["provider"] == "pexels"
    assert hit["resolution"] == "1920x1080"
    assert hit["storage_url"].startswith("file://")

    # Call get_clip_details tool
    chunk_id = hit["chunk_id"]
    clip_res = await server.call_tool("get_clip_details", {"chunk_id": chunk_id})
    assert not clip_res.is_error
    clip_data = _parse_tool_result(clip_res)
    assert clip_data["chunk_id"] == chunk_id
    assert clip_data["media_item_id"] == media_id
    assert clip_data["start_ts"] == 0.0

    # Call get_media_item_details tool
    item_res = await server.call_tool("get_media_item_details", {"media_item_id": media_id})
    assert not item_res.is_error
    item_data = _parse_tool_result(item_res)
    assert item_data["id"] == media_id
    assert item_data["status"] == "done"
    assert item_data["chunks_count"] >= 1


@pytest.mark.anyio
async def test_mcp_search_with_advanced_filters(mcp_env):
    """Test search_footage filtering by media_type, orientation, duration, and provider."""
    server = mcp_env["server"]
    storage = mcp_env["storage"]
    settings = mcp_env["settings"]
    embedder = mcp_env["embedder"]
    vector_store = mcp_env["vector_store"]

    storage.save_file(b"landscape vid", "horiz.mp4")
    storage.save_file(b"vertical vid", "vert.mp4")
    storage.save_file(b"landscape img", "photo.jpg")

    with get_db_session(settings.DATABASE_URL) as session:
        # 1. Landscape video, 15s, pexels
        v_horiz = MediaItem(
            provider="pexels",
            source_url="https://example.com/horiz.mp4",
            storage_path="horiz.mp4",
            media_type=MediaType.VIDEO,
            duration_sec=15.0,
            resolution="1920x1080",
            status=MediaStatus.PENDING,
        )
        # 2. Vertical video (Reels/Shorts format), 35s, pixabay
        v_vert = MediaItem(
            provider="pixabay",
            source_url="https://example.com/vert.mp4",
            storage_path="vert.mp4",
            media_type=MediaType.VIDEO,
            duration_sec=35.0,
            resolution="1080x1920",
            status=MediaStatus.PENDING,
        )
        # 3. Landscape image, pexels
        img_horiz = MediaItem(
            provider="pexels",
            source_url="https://example.com/photo.jpg",
            storage_path="photo.jpg",
            media_type=MediaType.IMAGE,
            resolution="1920x1080",
            status=MediaStatus.PENDING,
        )
        session.add_all([v_horiz, v_vert, img_horiz])

    processor = BatchProcessor(
        settings=settings,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=settings.DATABASE_URL,
    )
    processor.process_all_pending()

    # 1. Filter: orientation='vertical'
    res_vert = await server.call_tool(
        "search_footage",
        {"query": "action clip", "orientation": "vertical"},
    )
    data_vert = _parse_tool_result(res_vert)
    assert data_vert["count"] == 1
    assert data_vert["results"][0]["orientation"] == "vertical"
    assert data_vert["results"][0]["aspect_ratio"] == "9:16"
    assert data_vert["results"][0]["provider"] == "pixabay"

    # 2. Filter: orientation='landscape' (should return both landscape video and image)
    res_land = await server.call_tool(
        "search_footage",
        {"query": "action clip", "orientation": "landscape"},
    )
    data_land = _parse_tool_result(res_land)
    assert data_land["count"] == 2
    assert all(r["orientation"] == "horizontal" for r in data_land["results"])

    # 3. Filter: media_type='video' AND orientation='landscape'
    res_vid_land = await server.call_tool(
        "search_footage",
        {"query": "action clip", "media_type": "video", "orientation": "landscape"},
    )
    data_vid_land = _parse_tool_result(res_vid_land)
    assert data_vid_land["count"] == 1
    assert data_vid_land["results"][0]["resolution"] == "1920x1080"
    assert data_vid_land["results"][0]["duration_sec"] == 15.0

    # 4. Filter: min_duration=20.0 (only vertical video is 35s)
    res_min_dur = await server.call_tool(
        "search_footage",
        {"query": "action clip", "min_duration": 20.0},
    )
    data_min_dur = _parse_tool_result(res_min_dur)
    assert data_min_dur["count"] == 1
    assert data_min_dur["results"][0]["duration_sec"] == 35.0

    # 5. Filter: max_duration=20.0 (only landscape video is 15s)
    res_max_dur = await server.call_tool(
        "search_footage",
        {"query": "action clip", "media_type": "video", "max_duration": 20.0},
    )
    data_max_dur = _parse_tool_result(res_max_dur)
    assert data_max_dur["count"] == 1
    assert data_max_dur["results"][0]["duration_sec"] == 15.0

    # 6. Filter: provider='pixabay'
    res_prov = await server.call_tool(
        "search_footage",
        {"query": "action clip", "provider": "pixabay"},
    )
    data_prov = _parse_tool_result(res_prov)
    assert data_prov["count"] == 1
    assert data_prov["results"][0]["provider"] == "pixabay"


@pytest.mark.anyio
async def test_mcp_library_stats_and_resource(mcp_env):
    """Test get_library_stats tool and footage://stats resource."""
    server = mcp_env["server"]
    storage = mcp_env["storage"]
    settings = mcp_env["settings"]

    # Ingest test image
    storage.save_file(b"image payload", "mountain.jpg")
    with get_db_session(settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="pixabay",
            source_url="https://example.com/mountain.jpg",
            storage_path="mountain.jpg",
            media_type=MediaType.IMAGE,
            status=MediaStatus.DONE,
        )
        session.add(item)

    # Call get_library_stats tool
    stats_res = await server.call_tool("get_library_stats", {})
    assert not stats_res.is_error
    stats_data = _parse_tool_result(stats_res)
    assert stats_data["total_media_items"] == 1
    assert "pixabay" in stats_data["providers"]
    assert stats_data["vector_store"] == "in_memory"

    # Read footage://stats resource
    resource_contents = await server.read_resource("footage://stats")
    assert len(resource_contents) >= 1
    raw_json = resource_contents[0].content
    parsed = json.loads(raw_json)
    assert parsed["total_media_items"] == 1


@pytest.mark.anyio
async def test_mcp_ingest_url_and_deduplication(mcp_env):
    """Test ingest_url tool with pre-spend duplicate detection."""
    server = mcp_env["server"]
    temp_dir = mcp_env["temp_dir"]

    # Create dummy local file to simulate direct URL ingestion
    local_file = os.path.join(temp_dir, "sample.mp4")
    with open(local_file, "wb") as f:
        f.write(b"sample mp4 video data for ingest test")
    file_url = f"file://{os.path.abspath(local_file)}"

    # Ingest URL
    ingest_res1 = await server.call_tool(
        "ingest_url",
        {"url": file_url, "provider": "manual", "source_id": "vid_101", "auto_process": False},
    )
    assert not ingest_res1.is_error
    data1 = _parse_tool_result(ingest_res1)
    assert data1["is_duplicate"] is False
    assert data1["provider"] == "manual"
    first_id = data1["media_item_id"]

    # Ingest same URL again (must be detected as duplicate)
    ingest_res2 = await server.call_tool(
        "ingest_url",
        {"url": file_url, "provider": "manual", "source_id": "vid_101", "auto_process": False},
    )
    assert not ingest_res2.is_error
    data2 = _parse_tool_result(ingest_res2)
    assert data2["is_duplicate"] is True
    assert data2["media_item_id"] == first_id

    # Test automatic image extension detection (.webp -> image)
    img_file = os.path.join(temp_dir, "ocean_photo.webp")
    with open(img_file, "wb") as f:
        f.write(b"webp image fake bytes")
    img_url = f"file://{os.path.abspath(img_file)}"

    img_res = await server.call_tool(
        "ingest_url",
        {"url": img_url, "provider": "manual", "auto_process": False},
    )
    assert not img_res.is_error
    img_data = _parse_tool_result(img_res)
    assert img_data["media_type"] == "image"
    assert img_data["is_duplicate"] is False


@pytest.mark.anyio
async def test_mcp_fine_localize_clip(mcp_env):
    """Test fine_localize_clip tool with synthetic frames."""
    import cv2
    import numpy as np

    server = mcp_env["server"]
    storage = mcp_env["storage"]
    settings = mcp_env["settings"]
    embedder = mcp_env["embedder"]
    vector_store = mcp_env["vector_store"]

    # Create synthetic video
    video_filename = "synth_localize.mp4"
    local_path = os.path.join(storage.base_dir, video_filename)
    fps = 24
    width, height = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(local_path, fourcc, fps, (width, height))
    for i in range(120):
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        out.write(frame)
    out.release()

    with get_db_session(settings.DATABASE_URL) as session:
        item = MediaItem(
            provider="manual",
            source_url="https://example.com/synth_localize.mp4",
            storage_path=video_filename,
            duration_sec=5.0,
            status=MediaStatus.PENDING,
        )
        session.add(item)
        session.flush()
        media_id = item.id

    processor = BatchProcessor(
        settings=settings,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=settings.DATABASE_URL,
    )
    processor.process_all_pending()

    with get_db_session(settings.DATABASE_URL) as session:
        chunk = session.execute(select(Chunk).where(Chunk.media_item_id == media_id)).scalars().first()
        chunk_id = chunk.id

    res = await server.call_tool("fine_localize_clip", {"chunk_id": chunk_id, "query": "daylight"})
    assert not res.is_error
    loc_data = _parse_tool_result(res)
    assert loc_data["chunk_id"] == chunk_id
    assert loc_data["refined_start_ts"] >= 0.0
    assert loc_data["refined_end_ts"] <= 5.0
    assert loc_data["refined_start_ts"] < loc_data["refined_end_ts"]


@pytest.mark.anyio
async def test_mcp_prompt_broll_match_beat(mcp_env):
    """Test rendering the broll-match-beat prompt template."""
    server = mcp_env["server"]

    prompt_res = await server.get_prompt(
        "broll-match-beat",
        {"narration_text": "The rocket cleared the launch tower.", "target_duration_sec": 4.0},
    )
    assert prompt_res is not None
    messages = prompt_res.messages
    assert len(messages) >= 1
    content_text = messages[0].content.text
    assert "rocket cleared the launch tower" in content_text
    assert "4.0s" in content_text
