"""Task handlers for the asynchronous worker.

Task names and payload keys mirror the MCP tool surface one-for-one, so a caller
that already speaks the MCP contract can enqueue the identical request and
receive the identical response shape (see ``footage_engine.retrieval.serialize``).
"""

import logging
from typing import Any, Optional

from footage_engine.models.db import get_db_session
from footage_engine.models.media import MediaItem
from footage_engine.retrieval.models import SearchFilters
from footage_engine.retrieval.serialize import (
    clip_details_response,
    fine_localize_response,
    media_item_details_response,
    search_footage_response,
    search_script_beat_response,
)
from footage_engine.worker.base import TaskContext, TaskHandler

logger = logging.getLogger(__name__)

# Payload keys accepted for filtering -- identical to the MCP tool arguments.
_FILTER_KEYS = (
    "media_type",
    "orientation",
    "min_duration",
    "max_duration",
    "provider",
    "entity_name",
    "entity_id",
)


def _require(payload: dict[str, Any], key: str, task: str) -> Any:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"Task '{task}' requires the '{key}' payload field.")
    return value


def _filters_from_payload(payload: dict[str, Any]) -> Optional[SearchFilters]:
    """Build SearchFilters from the same keys the MCP tools accept."""
    if not any(payload.get(key) is not None for key in _FILTER_KEYS):
        return None
    return SearchFilters(
        media_type=payload.get("media_type"),
        orientation=payload.get("orientation"),
        min_duration_sec=payload.get("min_duration"),
        max_duration_sec=payload.get("max_duration"),
        provider=payload.get("provider"),
        entity_name=payload.get("entity_name"),
        entity_id=payload.get("entity_id"),
    )


def handle_search_footage(ctx: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Semantic footage search. Mirrors the ``search_footage`` MCP tool."""
    query = _require(payload, "query", "search_footage")
    results = ctx.retrieval_api.search(
        query=query,
        top_k=int(payload.get("top_k", 10)),
        filters=_filters_from_payload(payload),
    )
    return search_footage_response(query, results)


def handle_search_script_beat(ctx: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Multi-Query script-beat search. Mirrors the ``search_script_beat`` MCP tool."""
    beat_text = payload.get("beat_text") or payload.get("query") or payload.get("beat")
    if not beat_text:
        raise ValueError("Task 'search_script_beat' requires the 'beat_text' payload field.")
    rerank = bool(payload.get("rerank", False))
    results = ctx.retrieval_api.search_beat(
        beat_text=beat_text,
        top_k=int(payload.get("top_k", 10)),
        filters=_filters_from_payload(payload),
        rerank=rerank,
        confidence_floor=payload.get("confidence_floor"),
    )
    return search_script_beat_response(beat_text, results, rerank)


def handle_fine_localize_clip(ctx: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Frame-level cut refinement. Mirrors the ``fine_localize_clip`` MCP tool.

    This is the most GPU-intensive operation in the engine: it embeds the query
    once and then every sampled frame of the target chunk.
    """
    chunk_id = _require(payload, "chunk_id", "fine_localize_clip")
    query = _require(payload, "query", "fine_localize_clip")

    chunk_res = ctx.retrieval_api.get_chunk(chunk_id)
    if chunk_res is None:
        raise ValueError(f"Chunk '{chunk_id}' not found.")

    refined_start, refined_end = ctx.retrieval_api.fine_localize(chunk_id, query)
    return fine_localize_response(
        chunk_id=chunk_id,
        query=query,
        original_start_ts=chunk_res.start_ts,
        original_end_ts=chunk_res.end_ts,
        refined_start_ts=refined_start,
        refined_end_ts=refined_end,
    )


def handle_get_clip_details(ctx: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Chunk metadata lookup. Mirrors the ``get_clip_details`` MCP tool."""
    chunk_id = _require(payload, "chunk_id", "get_clip_details")
    chunk_res = ctx.retrieval_api.get_chunk(chunk_id)
    if chunk_res is None:
        raise ValueError(f"Chunk '{chunk_id}' not found.")
    return clip_details_response(chunk_res)


def handle_get_media_item_details(ctx: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Raw media item lookup. Mirrors the ``get_media_item_details`` MCP tool."""
    media_item_id = _require(payload, "media_item_id", "get_media_item_details")
    with get_db_session(ctx.database_url) as session:
        item = session.get(MediaItem, media_item_id)
        if not item:
            raise ValueError(f"MediaItem with id '{media_item_id}' not found.")
        return media_item_details_response(item, ctx.storage)


# Task name -> handler. Names are intentionally identical to the MCP tool names.
TASK_HANDLERS: dict[str, TaskHandler] = {
    "search_footage": handle_search_footage,
    "search_script_beat": handle_search_script_beat,
    "fine_localize_clip": handle_fine_localize_clip,
    "get_clip_details": handle_get_clip_details,
    "get_media_item_details": handle_get_media_item_details,
}

#: Tasks a search worker serves when no allowlist is configured.
SEARCH_TASKS: tuple[str, ...] = tuple(TASK_HANDLERS)


def available_tasks() -> list[str]:
    """Return the sorted names of every registered task."""
    return sorted(TASK_HANDLERS)


def run_task(ctx: TaskContext, task: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Dispatch one job to its handler, raising ValueError for unknown tasks."""
    handler = TASK_HANDLERS.get(task)
    if handler is None:
        raise ValueError(
            f"Unknown task '{task}'. Available tasks: {', '.join(available_tasks())}"
        )
    return handler(ctx, payload or {})


__all__ = [
    "TASK_HANDLERS",
    "SEARCH_TASKS",
    "available_tasks",
    "run_task",
    "handle_search_footage",
    "handle_search_script_beat",
    "handle_fine_localize_clip",
    "handle_get_clip_details",
    "handle_get_media_item_details",
]
