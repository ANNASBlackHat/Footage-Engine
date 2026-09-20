"""Model Context Protocol (MCP) server for Footage Retrieval Engine."""

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional
from sqlalchemy import func, select

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore
    except ImportError as err:
        raise ImportError(
            "The 'mcp' package is required to run the MCP server. "
            "Install it via 'uv pip install mcp' or 'pip install -e \".[mcp]\"'."
        ) from err

from footage_engine.config import Settings, get_settings
from footage_engine.embeddings import get_embedder
from footage_engine.embeddings.base import EmbeddingBackend
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Chunk, MediaItem, MediaStatus, MediaType
from footage_engine.orchestrator import Orchestrator
from footage_engine.pipeline.processor import BatchProcessor
from footage_engine.retrieval.api import RetrievalAPI
from footage_engine.retrieval.models import SearchFilters
from footage_engine.storage import get_storage_backend
from footage_engine.storage.base import StorageBackend
from footage_engine.vector import get_vector_store
from footage_engine.vector.base import VectorStore

logger = logging.getLogger(__name__)


def create_mcp_server(
    use_mock: bool = False,
    settings: Optional[Settings] = None,
    storage: Optional[StorageBackend] = None,
    embedder: Optional[EmbeddingBackend] = None,
    vector_store: Optional[VectorStore] = None,
    database_url: Optional[str] = None,
    backend: Optional[str] = None,
) -> MCPServer:
    """Create and configure an MCPServer instance with all tools, resources, and prompts."""
    settings = settings or get_settings()
    if backend is not None:
        settings.EMBEDDING_BACKEND = backend  # type: ignore[assignment]
    db_url = database_url or settings.DATABASE_URL
    init_db(db_url)

    storage = storage or get_storage_backend(settings)

    is_mock = (
        use_mock
        or os.environ.get("FOOTAGE_MCP_USE_MOCK") == "1"
        or os.environ.get("USE_MOCK_EMBEDDER") == "1"
    )
    if embedder is None:
        if is_mock:
            embedder = MockEmbedder(dimension=settings.EMBEDDING_DIMENSION)
        else:
            embedder = get_embedder(settings)

    vector_store = vector_store or get_vector_store(settings, backend=settings.EMBEDDING_BACKEND)

    retrieval_api = RetrievalAPI(
        settings=settings,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=db_url,
    )
    orchestrator = Orchestrator(
        settings=settings,
        storage=storage,
        database_url=db_url,
    )
    batch_processor = BatchProcessor(
        settings=settings,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=db_url,
    )

    server = MCPServer(
        name="Footage Retrieval Engine",
        instructions=(
            "Footage Retrieval Engine MCP Server. "
            "Use this server to semantically search video/image footage, "
            "refine cut boundaries with frame-level fine localization, "
            "ingest stock or web footage, and inspect media metadata."
        ),
    )

    # -------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------

    @server.tool(
        name="search_footage",
        description=(
            "Semantically search for video or image footage using natural language queries. "
            "Supports filtering by canonical entity (entity_name or entity_id), media type ('video', 'image'), "
            "orientation ('landscape'/'horizontal' or 'vertical'/'portrait'), "
            "duration bounds (min_duration, max_duration in seconds), and provider ('pexels', 'pixabay', 'coverr', 'youtube', 'manual'). "
            "Returns ranked video chunks with scores, exact start/end timestamps, orientation, aspect ratio, "
            "entity associations, and media streaming/storage URLs."
        ),
    )
    def search_footage(
        query: str,
        top_k: int = 10,
        media_type: Optional[str] = None,
        orientation: Optional[str] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        provider: Optional[str] = None,
        entity_name: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> dict[str, Any]:
        filters = None
        if any(v is not None for v in (media_type, orientation, min_duration, max_duration, provider, entity_name, entity_id)):
            filters = SearchFilters(
                media_type=media_type,
                orientation=orientation,
                min_duration_sec=min_duration,
                max_duration_sec=max_duration,
                provider=provider,
                entity_name=entity_name,
                entity_id=entity_id,
            )

        results = retrieval_api.search(query=query, top_k=top_k, filters=filters)
        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "media_item_id": r.media_item_id,
                    "score": round(float(r.score), 4),
                    "start_ts": round(float(r.start_ts), 2),
                    "end_ts": round(float(r.end_ts), 2) if r.end_ts is not None else None,
                    "duration_sec": round(float(r.duration_sec), 2) if r.duration_sec is not None else None,
                    "media_type": r.media_type,
                    "orientation": r.orientation,
                    "aspect_ratio": r.aspect_ratio,
                    "provider": r.provider,
                    "entity_id": r.entity_id,
                    "entity_name": r.entity_name,
                    "storage_url": r.storage_url,
                    "source_url": r.source_url,
                    "resolution": r.resolution,
                    "license_type": r.license_type,
                }
                for r in results
            ],
        }

    @server.tool(
        name="fine_localize_clip",
        description=(
            "Refine video cut points within a chunk by running frame-by-frame (~1fps) "
            "multimodal scoring to locate the exact sub-window of action matching the query."
        ),
    )
    def fine_localize_clip(chunk_id: str, query: str) -> dict[str, Any]:
        chunk_res = retrieval_api.get_chunk(chunk_id)
        orig_start = chunk_res.start_ts
        orig_end = chunk_res.end_ts

        sub_start, sub_end = retrieval_api.fine_localize(chunk_id, query)
        return {
            "chunk_id": chunk_id,
            "query": query,
            "original_start_ts": round(float(orig_start), 2),
            "original_end_ts": round(float(orig_end), 2) if orig_end is not None else None,
            "refined_start_ts": round(float(sub_start), 2),
            "refined_end_ts": round(float(sub_end), 2),
            "refined_duration_sec": round(float(sub_end - sub_start), 2),
        }

    @server.tool(
        name="get_clip_details",
        description=(
            "Get complete metadata, timestamps, resolution, parent media item info, "
            "and streaming/storage URLs for a specific chunk."
        ),
    )
    def get_clip_details(chunk_id: str) -> dict[str, Any]:
        res = retrieval_api.get_chunk(chunk_id)
        return {
            "chunk_id": res.chunk_id,
            "media_item_id": res.media_item_id,
            "start_ts": res.start_ts,
            "end_ts": res.end_ts,
            "duration_sec": res.duration_sec,
            "media_type": res.media_type,
            "orientation": res.orientation,
            "aspect_ratio": res.aspect_ratio,
            "provider": res.provider,
            "storage_url": res.storage_url,
            "source_url": res.source_url,
            "resolution": res.resolution,
            "license_type": res.license_type,
        }

    @server.tool(
        name="get_media_item_details",
        description=(
            "Get full details for a raw media item, including status, download source, "
            "duration, and all partitioned chunks."
        ),
    )
    def get_media_item_details(media_item_id: str) -> dict[str, Any]:
        with get_db_session(db_url) as session:
            item = session.get(MediaItem, media_item_id)
            if not item:
                raise ValueError(f"MediaItem with id '{media_item_id}' not found.")
            storage_url = storage.get_url(item.storage_path) if item.storage_path else ""
            chunks_data = [
                {
                    "chunk_id": c.id,
                    "start_ts": round(float(c.start_ts), 2),
                    "end_ts": round(float(c.end_ts), 2) if c.end_ts is not None else None,
                    "storage_path": c.storage_path,
                    "usage_count": c.usage_count,
                }
                for c in item.chunks
            ]
            return {
                "id": item.id,
                "provider": item.provider,
                "source_id": item.source_id,
                "source_url": item.source_url,
                "media_type": item.media_type.value,
                "status": item.status.value,
                "duration_sec": item.duration_sec,
                "resolution": item.resolution,
                "storage_path": item.storage_path,
                "storage_url": storage_url,
                "chunks_count": len(chunks_data),
                "chunks": chunks_data,
            }

    @server.tool(
        name="ingest_url",
        description=(
            "Ingest a media item by direct file URL or YouTube URL. "
            "Applies pre-spend deduplication before network I/O. "
            "Supports optional entity association (entity_name or entity_id) for entity-aware footage tagging. "
            "Supports optional media_type ('image', 'video', or None for automatic extension detection like .webp/.png/.jpg/.mp4). "
            "If auto_process=True, chunks and indexes the clip immediately."
        ),
    )
    def ingest_url(
        url: str,
        provider: str = "manual",
        media_type: Optional[str] = None,
        source_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        entity_id: Optional[str] = None,
        auto_process: bool = True,
    ) -> dict[str, Any]:
        existing = orchestrator.find_existing(url, provider=provider, source_id=source_id)
        item = orchestrator.ingest(
            source_url=url,
            provider=provider,
            source_id=source_id,
            media_type=media_type,
            entity_name=entity_name,
            entity_id=entity_id,
        )
        was_duplicate = existing is not None
        processed = False

        if auto_process and not was_duplicate and item.status != MediaStatus.DONE:
            processed = batch_processor.process_item(item.id)

        return {
            "media_item_id": item.id,
            "provider": item.provider,
            "source_url": item.source_url,
            "media_type": item.media_type.value,
            "entity_id": item.entity_id,
            "is_duplicate": was_duplicate,
            "status": item.status.value,
            "auto_processed": processed,
        }

    @server.tool(
        name="ingest_keywords",
        description=(
            "Search stock footage providers (pexels, pixabay, coverr) by keyword, "
            "deduplicate candidates before downloading, and optionally process them into the vector store."
        ),
    )
    def ingest_keywords(
        keyword: str,
        provider: str = "pexels",
        max_results: int = 5,
        auto_process: bool = True,
    ) -> dict[str, Any]:
        items = orchestrator.search_and_ingest(
            keyword=keyword,
            provider=provider,  # type: ignore
            max_results=max_results,
        )
        processed_count = 0
        if auto_process:
            for item in items:
                if item.status != MediaStatus.DONE:
                    if batch_processor.process_item(item.id):
                        processed_count += 1

        return {
            "keyword": keyword,
            "provider": provider,
            "ingested_count": len(items),
            "media_item_ids": [it.id for it in items],
            "processed_count": processed_count,
        }

    @server.tool(
        name="process_pending_queue",
        description=(
            "Process pending or interrupted media items through scene chunking, "
            "multimodal embedding, and vector indexing."
        ),
    )
    def process_pending_queue(max_items: int = 20) -> dict[str, Any]:
        stats = batch_processor.process_all_pending(limit=max_items)
        return {
            "processed": stats["processed"],
            "failed": stats["failed"],
            "total": stats["total"],
        }

    @server.tool(
        name="get_library_stats",
        description="Get global statistics of indexed media items, chunks, and engine configuration.",
    )
    def get_library_stats() -> dict[str, Any]:
        with get_db_session(db_url) as session:
            total_media = session.query(MediaItem).count()
            total_chunks = session.query(Chunk).count()

            provider_rows = (
                session.query(MediaItem.provider, func.count(MediaItem.id))
                .group_by(MediaItem.provider)
                .all()
            )
            by_provider = {p: count for p, count in provider_rows}

            status_rows = (
                session.query(MediaItem.status, func.count(MediaItem.id))
                .group_by(MediaItem.status)
                .all()
            )
            status_map = {
                (s.value if hasattr(s, "value") else str(s)): count
                for s, count in status_rows
            }

        return {
            "total_media_items": total_media,
            "total_chunks": total_chunks,
            "providers": by_provider,
            "status_distribution": status_map,
            "vector_store": settings.VECTOR_STORE,
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.dimension,
        }

    @server.tool(
        name="list_entities",
        description="List registered canonical entities with their IDs, aliases, and entity types.",
    )
    def list_entities(entity_type: Optional[str] = None) -> dict[str, Any]:
        entities = orchestrator.entity_resolver.list_entities(entity_type=entity_type)
        return {
            "count": len(entities),
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "aliases": e.aliases or [],
                    "notes": e.notes,
                }
                for e in entities
            ],
        }

    @server.tool(
        name="resolve_or_create_entity",
        description="Resolve an existing entity by name or alias, or register a new canonical entity.",
    )
    def resolve_or_create_entity(
        name: str,
        entity_type: str = "other",
        aliases: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        ent = orchestrator.entity_resolver.resolve_or_create(
            name=name,
            entity_type=entity_type,
            aliases=aliases,
            notes=notes,
        )
        return {
            "id": ent.id,
            "name": ent.name,
            "entity_type": ent.entity_type,
            "aliases": ent.aliases or [],
            "notes": ent.notes,
        }

    # -------------------------------------------------------------------------
    # Resources
    # -------------------------------------------------------------------------

    @server.resource("footage://chunks/{chunk_id}")
    def get_chunk_resource(chunk_id: str) -> str:
        """Resource providing JSON metadata for a specific footage chunk."""
        res = retrieval_api.get_chunk(chunk_id)
        payload = {
            "chunk_id": res.chunk_id,
            "media_item_id": res.media_item_id,
            "start_ts": res.start_ts,
            "end_ts": res.end_ts,
            "duration_sec": res.duration_sec,
            "media_type": res.media_type,
            "provider": res.provider,
            "storage_url": res.storage_url,
            "source_url": res.source_url,
            "resolution": res.resolution,
            "license_type": res.license_type,
        }
        return json.dumps(payload, indent=2)

    @server.resource("footage://stats")
    def get_stats_resource() -> str:
        """Resource providing engine statistics."""
        stats = get_library_stats()
        return json.dumps(stats, indent=2)

    # -------------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------------

    @server.prompt("broll-match-beat")
    def broll_match_beat(
        narration_text: str,
        target_duration_sec: float = 5.0,
    ) -> str:
        """Prompt template to help models formulate optimal visual search queries for script beats."""
        return (
            f"You are a cinematic video director matching B-roll footage to narration.\n\n"
            f"Narration Line: \"{narration_text}\"\n"
            f"Target Clip Duration: {target_duration_sec}s\n\n"
            f"Formulate 2-3 visual search queries to find appropriate b-roll using search_footage. "
            f"Focus on concrete subjects, actions, lighting, and camera motion rather than abstract concepts."
        )

    return server


_default_server: Optional[MCPServer] = None


def get_default_server() -> MCPServer:
    """Get or create singleton default MCP server instance."""
    global _default_server
    if _default_server is None:
        _default_server = create_mcp_server()
    return _default_server


def __getattr__(name: str) -> Any:
    if name == "mcp":
        return get_default_server()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def main():
    """CLI entrypoint for running the Footage Retrieval Engine MCP server."""
    parser = argparse.ArgumentParser(description="Footage Retrieval Engine MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address when using network transports (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number when using network transports (default: 8000)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use lightweight MockEmbedder for instant testing without neural network models",
    )
    parser.add_argument(
        "--backend",
        choices=["xclip", "qwen"],
        default=None,
        help="Embedding backend (default: EMBEDDING_BACKEND env or 'xclip')",
    )

    args = parser.parse_args()

    # Configure logging for stdio transport
    log_level = logging.WARNING if args.transport == "stdio" else logging.INFO
    logging.basicConfig(level=log_level, stream=sys.stderr)

    server = create_mcp_server(use_mock=args.mock, backend=args.backend)

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
