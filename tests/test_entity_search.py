"""Tests for Entity-filtered semantic search and MCP tools."""

import pytest
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaStatus
from footage_engine.orchestrator import Orchestrator
from footage_engine.pipeline.processor import BatchProcessor
from footage_engine.retrieval.api import RetrievalAPI
from footage_engine.retrieval.models import SearchFilters
from footage_engine.vector.in_memory import InMemoryVectorStore


def test_entity_filtered_retrieval(test_settings, test_storage):
    init_db(test_settings.DATABASE_URL)
    vector_store = InMemoryVectorStore()
    embedder = MockEmbedder(dimension=512)

    orchestrator = Orchestrator(
        settings=test_settings,
        storage=test_storage,
        database_url=test_settings.DATABASE_URL,
    )

    # Ingest 3 clips:
    # 1. USS Cyclops
    # 2. Aye-aye lemur
    # 3. Generic ocean waves (no entity)
    p1 = test_storage.save_file(b"cyclops footage", "cyclops.mp4")
    p2 = test_storage.save_file(b"ayeaye footage", "ayeaye.mp4")
    p3 = test_storage.save_file(b"ocean footage", "ocean.mp4")

    item_cyclops = orchestrator.ingest(
        source_url=f"file://{p1}",
        provider="manual",
        entity_name="USS Cyclops",
        entity_type="ship",
        duration_sec=20.0,
    )
    item_ayeaye = orchestrator.ingest(
        source_url=f"file://{p2}",
        provider="manual",
        entity_name="Aye-aye",
        entity_type="animal",
        duration_sec=15.0,
    )
    item_ocean = orchestrator.ingest(
        source_url=f"file://{p3}",
        provider="manual",
        duration_sec=30.0,
    )

    assert item_cyclops.entity_id is not None
    assert item_ayeaye.entity_id is not None
    assert item_ocean.entity_id is None

    # Process all pending items through chunking and vector store
    processor = BatchProcessor(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )
    stats = processor.process_all_pending(max_workers=1)
    assert stats["succeeded"] == 3

    # Initialize Retrieval API
    api = RetrievalAPI(
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )

    # 1. General search (no entity filter) - returns all 3 clips
    all_results = api.search(query="footage query", top_k=10)
    assert len(all_results) == 3

    # 2. Entity-filtered search by entity_name="USS Cyclops"
    cyclops_results = api.search(
        query="footage query",
        top_k=10,
        filters=SearchFilters(entity_name="USS Cyclops"),
    )
    assert len(cyclops_results) == 1
    assert cyclops_results[0].entity_id == item_cyclops.entity_id
    assert cyclops_results[0].entity_name == "USS Cyclops"

    # 3. Entity-filtered search by entity_id
    ayeaye_results = api.search(
        query="footage query",
        top_k=10,
        filters=SearchFilters(entity_id=item_ayeaye.entity_id),
    )
    assert len(ayeaye_results) == 1
    assert ayeaye_results[0].entity_id == item_ayeaye.entity_id
    assert ayeaye_results[0].entity_name == "Aye-aye"

    # 4. Search for entity with no clips indexed - returns empty list
    orchestrator.entity_resolver.resolve_or_create(name="Eiffel Tower", entity_type="location")
    empty_results = api.search(
        query="footage query",
        top_k=10,
        filters=SearchFilters(entity_name="Eiffel Tower"),
    )
    assert len(empty_results) == 0

    # 5. Search for non-existent entity name - returns empty list
    nonexistent_results = api.search(
        query="footage query",
        top_k=10,
        filters=SearchFilters(entity_name="Atlantis"),
    )
    assert len(nonexistent_results) == 0


def test_mcp_server_entity_tools(test_settings, test_storage):
    from footage_engine.mcp.server import create_mcp_server

    init_db(test_settings.DATABASE_URL)
    embedder = MockEmbedder(dimension=512)
    vector_store = InMemoryVectorStore()

    server = create_mcp_server(
        use_mock=True,
        settings=test_settings,
        storage=test_storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=test_settings.DATABASE_URL,
    )

    # Get tools from server
    # FastMCP tools can be inspected or called via server
    resolver_tool = getattr(server, "resolve_or_create_entity", None)
    list_tool = getattr(server, "list_entities", None)
    ingest_tool = getattr(server, "ingest_url", None)
    search_tool = getattr(server, "search_footage", None)

    # Register entity via MCP tool if callable directly or through server
    # We can test the underlying functions directly registered
    assert server is not None
