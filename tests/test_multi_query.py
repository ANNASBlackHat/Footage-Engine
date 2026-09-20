"""Tests for Multi-Query Expansion, LLM Judge reranking, and search_beat."""

from unittest.mock import MagicMock, patch
import pytest

from footage_engine.config import Settings
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import MediaItem, MediaStatus, MediaType
from footage_engine.pipeline.processor import BatchProcessor
from footage_engine.retrieval.api import RetrievalAPI
from footage_engine.retrieval.llm import ExpandedBeat, LLMClient, LLMJudge, QueryExpander
from footage_engine.retrieval.models import ChunkResult
from footage_engine.vector.in_memory import InMemoryVectorStore


# ---------------------------------------------------------------------------
# 1. LLMClient Tests
# ---------------------------------------------------------------------------

def test_llm_client_missing_key():
    settings = Settings(LLM_API_KEY="", GEMINI_API_KEY="", OPENAI_API_KEY="")
    client = LLMClient(settings=settings)
    assert not client.is_available()
    assert client.complete_json(prompt="hi") is None


@patch("requests.post")
def test_llm_client_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"content": '{"queries": ["ships sailing"], "detected_entity": null}'}}
        ]
    }
    mock_post.return_value = mock_resp

    settings = Settings(LLM_API_KEY="test-key", LLM_BASE_URL="https://api.example.com/v1")
    client = LLMClient(settings=settings)
    assert client.is_available()

    res = client.complete_json(prompt="expand beat")
    assert res is not None
    assert "queries" in res
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert call_kwargs["json"]["messages"][1]["content"] == "expand beat"


@patch("requests.post")
def test_llm_client_api_error(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = Exception("Internal Server Error")
    mock_post.return_value = mock_resp

    settings = Settings(LLM_API_KEY="test-key")
    client = LLMClient(settings=settings)
    res = client.complete_json(prompt="hi")
    assert res is None


# ---------------------------------------------------------------------------
# 2. QueryExpander Tests
# ---------------------------------------------------------------------------

def test_query_expander_unconfigured():
    client = MagicMock(spec=LLMClient)
    client.is_available.return_value = False
    expander = QueryExpander(client=client)

    beat = "As dusk fell over the harbor, the fleet prepared to depart."
    expanded = expander.expand_beat(beat)
    assert isinstance(expanded, ExpandedBeat)
    assert expanded.queries == [beat]
    assert expanded.detected_entity is None


def test_query_expander_success():
    client = MagicMock(spec=LLMClient)
    client.is_available.return_value = True
    client.complete_json.return_value = {
        "queries": ["harbor sunset ships", "fleet preparing to depart dock", "dusk twilight ocean port"],
        "detected_entity": None,
    }

    expander = QueryExpander(client=client)
    beat = "As dusk fell over the harbor, the fleet prepared to depart."
    expanded = expander.expand_beat(beat)

    assert len(expanded.queries) == 3
    assert "harbor sunset ships" in expanded.queries
    assert expanded.detected_entity is None


def test_query_expander_with_detected_entity():
    client = MagicMock(spec=LLMClient)
    client.is_available.return_value = True
    client.complete_json.return_value = {
        "queries": ["USS Cyclops sailing", "naval collier ship ocean"],
        "detected_entity": "USS Cyclops",
    }

    expander = QueryExpander(client=client)
    expanded = expander.expand_beat("The USS Cyclops vanished without a trace.")

    assert expanded.detected_entity == "USS Cyclops"
    assert len(expanded.queries) == 2


def test_query_expander_fallback_on_parse_error():
    client = MagicMock(spec=LLMClient)
    client.is_available.return_value = True
    client.complete_json.return_value = None

    expander = QueryExpander(client=client)
    beat = "A lonely astronaut floats in the cosmic void."
    expanded = expander.expand_beat(beat)
    assert expanded.queries == [beat]
    assert expanded.detected_entity is None


# ---------------------------------------------------------------------------
# 3. LLMJudge Tests
# ---------------------------------------------------------------------------

def test_llm_judge_unconfigured():
    client = MagicMock(spec=LLMClient)
    client.is_available.return_value = False
    judge = LLMJudge(client=client)

    candidates = [
        ChunkResult(
            chunk_id="c1", media_item_id="m1", score=0.8, start_ts=0.0, end_ts=5.0,
            duration_sec=5.0, media_type="video", provider="pexels",
            source_url="https://ex.com/1", storage_path="c1.mp4", storage_url="file:///c1.mp4",
        ),
        ChunkResult(
            chunk_id="c2", media_item_id="m2", score=0.7, start_ts=0.0, end_ts=5.0,
            duration_sec=5.0, media_type="video", provider="pexels",
            source_url="https://ex.com/2", storage_path="c2.mp4", storage_url="file:///c2.mp4",
        ),
    ]
    reranked = judge.judge_and_rerank("A ship in the ocean", candidates)
    assert [r.chunk_id for r in reranked] == ["c1", "c2"]


def test_llm_judge_rerank_success():
    client = MagicMock(spec=LLMClient)
    client.is_available.return_value = True
    # Invert the order: c2 preferred over c1
    client.complete_json.return_value = {
        "ranked_chunk_ids": ["c2", "c1"],
        "confidence_scores": {"c2": 0.95, "c1": 0.60},
    }

    judge = LLMJudge(client=client)
    candidates = [
        ChunkResult(
            chunk_id="c1", media_item_id="m1", score=0.8, start_ts=0.0, end_ts=5.0,
            duration_sec=5.0, media_type="video", provider="pexels",
            source_url="https://ex.com/1", storage_path="c1.mp4", storage_url="file:///c1.mp4",
        ),
        ChunkResult(
            chunk_id="c2", media_item_id="m2", score=0.7, start_ts=0.0, end_ts=5.0,
            duration_sec=5.0, media_type="video", provider="pexels",
            source_url="https://ex.com/2", storage_path="c2.mp4", storage_url="file:///c2.mp4",
        ),
    ]

    reranked = judge.judge_and_rerank("A massive storm in the ocean", candidates)
    assert [r.chunk_id for r in reranked] == ["c2", "c1"]
    assert reranked[0].score == 0.95
    assert reranked[1].score == 0.60


# ---------------------------------------------------------------------------
# 4. RetrievalAPI.search_beat Integration Test
# ---------------------------------------------------------------------------

def test_search_beat_integration(test_settings, test_storage):
    init_db(test_settings.DATABASE_URL)
    vector_store = InMemoryVectorStore()
    embedder = MockEmbedder(dimension=512)

    # Ingest 3 clips
    test_storage.save_file(b"clip1", "harbor.mp4")
    test_storage.save_file(b"clip2", "ocean.mp4")
    test_storage.save_file(b"clip3", "city.mp4")

    with get_db_session(test_settings.DATABASE_URL) as session:
        m1 = MediaItem(provider="pexels", source_url="https://ex.com/1", storage_path="harbor.mp4", duration_sec=10, status=MediaStatus.PENDING)
        m2 = MediaItem(provider="pexels", source_url="https://ex.com/2", storage_path="ocean.mp4", duration_sec=12, status=MediaStatus.PENDING)
        m3 = MediaItem(provider="pexels", source_url="https://ex.com/3", storage_path="city.mp4", duration_sec=8, status=MediaStatus.PENDING)
        session.add_all([m1, m2, m3])

    processor = BatchProcessor(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )
    processor.process_all_pending()

    api = RetrievalAPI(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )

    # Mock expander to return 2 distinct queries
    mock_expander = MagicMock(spec=QueryExpander)
    mock_expander.expand_beat.return_value = ExpandedBeat(
        raw_beat="The ships anchored peacefully under the sunset",
        queries=["calm harbor sunset", "deep ocean waves"],
        detected_entity=None,
    )

    with patch.object(api, "query_expander", mock_expander):
        results = api.search_beat("The ships anchored peacefully under the sunset", top_k=2, rerank=False)
        assert len(results) <= 2
        assert all(isinstance(r, ChunkResult) for r in results)
        mock_expander.expand_beat.assert_called_once()

    # Test with confidence_floor
    with patch.object(api, "query_expander", mock_expander):
        # Setting a ridiculously high confidence floor should filter out mock results
        high_floor_results = api.search_beat("The ships anchored peacefully", top_k=5, confidence_floor=2.0)
        assert len(high_floor_results) == 0


def test_mcp_search_script_beat(test_settings, test_storage):
    from footage_engine.mcp.server import create_mcp_server

    init_db(test_settings.DATABASE_URL)
    vector_store = InMemoryVectorStore()
    embedder = MockEmbedder(dimension=512)

    test_storage.save_file(b"clip1", "storm.mp4")
    with get_db_session(test_settings.DATABASE_URL) as session:
        m1 = MediaItem(provider="pexels", source_url="https://ex.com/s", storage_path="storm.mp4", duration_sec=10, status=MediaStatus.PENDING)
        session.add(m1)

    processor = BatchProcessor(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )
    processor.process_all_pending()

    server = create_mcp_server(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )

    # Verify tool is registered on server
    tools = getattr(server, "_tools", {}) or getattr(server, "tools", {})
    assert "search_script_beat" in tools or any("search_script_beat" in str(t) for t in getattr(server, "_tool_manager", []).list_tools() if hasattr(server, "_tool_manager"))
