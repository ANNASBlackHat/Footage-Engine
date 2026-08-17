"""Footage Retrieval Engine — Standalone library for footage ingestion, chunking, embedding, and semantic search."""

from footage_engine.config import Settings, get_settings
from footage_engine.models.media import Chunk, MediaItem, MediaStatus, MediaType
from footage_engine.orchestrator import (
    Orchestrator,
    get_orchestrator,
    ingest,
    ingest_from_provider,
    ingest_url_list,
    search_and_ingest,
)
from footage_engine.pipeline.processor import BatchProcessor
from footage_engine.retrieval.api import (
    RetrievalAPI,
    fine_localize,
    get_chunk,
    get_media_item,
    get_retrieval_api,
    search,
)
from footage_engine.retrieval.models import ChunkResult, SearchFilters

__all__ = [
    # Config & Settings
    "Settings",
    "get_settings",
    # Data Models
    "MediaItem",
    "Chunk",
    "MediaStatus",
    "MediaType",
    "ChunkResult",
    "SearchFilters",
    # Ingestion API
    "ingest",
    "search_and_ingest",
    "ingest_url_list",
    "ingest_from_provider",
    "Orchestrator",
    "get_orchestrator",
    # Batch Processing Pipeline
    "BatchProcessor",
    # Retrieval API
    "search",
    "fine_localize",
    "get_chunk",
    "get_media_item",
    "RetrievalAPI",
    "get_retrieval_api",
]

__version__ = "0.1.0"
