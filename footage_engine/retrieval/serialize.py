"""Shared JSON serializers for retrieval responses.

The MCP server and the asynchronous job worker both build their responses with
these helpers, so a queued job returns exactly the same payload as the
equivalent synchronous MCP tool call.
"""

from typing import Any, Optional, Sequence

from footage_engine.models.media import MediaItem
from footage_engine.retrieval.models import ChunkResult
from footage_engine.storage.base import StorageBackend


def _chunk_entry(r: ChunkResult) -> dict[str, Any]:
    """One ranked result entry, shared by search_footage and search_script_beat."""
    return {
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


def chunk_results_to_list(results: Sequence[ChunkResult]) -> list[dict[str, Any]]:
    """Serialize a ranked list of ChunkResult dataclasses into plain JSON dicts."""
    return [_chunk_entry(r) for r in results]


def search_footage_response(query: str, results: Sequence[ChunkResult]) -> dict[str, Any]:
    """Response payload for the ``search_footage`` operation."""
    return {
        "query": query,
        "count": len(results),
        "results": chunk_results_to_list(results),
    }


def search_script_beat_response(
    beat_text: str,
    results: Sequence[ChunkResult],
    reranked: bool,
) -> dict[str, Any]:
    """Response payload for the ``search_script_beat`` operation."""
    return {
        "beat_text": beat_text,
        "reranked": reranked,
        "count": len(results),
        "results": chunk_results_to_list(results),
    }


def clip_details_response(r: ChunkResult) -> dict[str, Any]:
    """Response payload for the ``get_clip_details`` operation."""
    return {
        "chunk_id": r.chunk_id,
        "media_item_id": r.media_item_id,
        "start_ts": r.start_ts,
        "end_ts": r.end_ts,
        "duration_sec": r.duration_sec,
        "media_type": r.media_type,
        "orientation": r.orientation,
        "aspect_ratio": r.aspect_ratio,
        "provider": r.provider,
        "storage_url": r.storage_url,
        "source_url": r.source_url,
        "resolution": r.resolution,
        "license_type": r.license_type,
    }


def fine_localize_response(
    chunk_id: str,
    query: str,
    original_start_ts: float,
    original_end_ts: Optional[float],
    refined_start_ts: float,
    refined_end_ts: float,
) -> dict[str, Any]:
    """Response payload for the ``fine_localize_clip`` operation."""
    return {
        "chunk_id": chunk_id,
        "query": query,
        "original_start_ts": round(float(original_start_ts), 2),
        "original_end_ts": round(float(original_end_ts), 2) if original_end_ts is not None else None,
        "refined_start_ts": round(float(refined_start_ts), 2),
        "refined_end_ts": round(float(refined_end_ts), 2),
        "refined_duration_sec": round(float(refined_end_ts - refined_start_ts), 2),
    }


def media_item_details_response(item: MediaItem, storage: StorageBackend) -> dict[str, Any]:
    """Response payload for the ``get_media_item_details`` operation."""
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


__all__ = [
    "chunk_results_to_list",
    "search_footage_response",
    "search_script_beat_response",
    "clip_details_response",
    "fine_localize_response",
    "media_item_details_response",
]
