"""Producer-side helpers and CLI for submitting footage-search jobs.

Any caller that can reach the database can enqueue work: from Python use these
helpers, and from another language (JS, Go, ...) simply INSERT a row into
``jobs`` with ``task`` and a JSON ``payload``.
"""

import argparse
import json
import sys
from typing import Any, Optional

from footage_engine.config import get_settings
from footage_engine.models.db import init_db
from footage_engine.models.jobs import JobStatus
from footage_engine.worker.queue import (
    count_jobs,
    count_live_workers,
    enqueue_job,
    get_job,
    list_workers,
    wait_for_job,
)
from footage_engine.worker.tasks import available_tasks

#: Filters accepted by the search tasks (mirrors the MCP tool arguments).
_FILTER_KEYS = (
    "media_type",
    "orientation",
    "min_duration",
    "max_duration",
    "provider",
    "entity_name",
    "entity_id",
)


def submit_job(
    task: str,
    payload: dict[str, Any],
    *,
    database_url: Optional[str] = None,
    backend: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Enqueue one job and return its id."""
    return enqueue_job(
        database_url,
        task=task,
        payload=payload,
        backend=backend,
        idempotency_key=idempotency_key,
    )


def _with_filters(payload: dict[str, Any], filters: Optional[dict[str, Any]]) -> dict[str, Any]:
    for key, value in (filters or {}).items():
        if key in _FILTER_KEYS and value is not None:
            payload[key] = value
    return payload


def submit_search_footage(
    query: str,
    *,
    top_k: int = 10,
    filters: Optional[dict[str, Any]] = None,
    database_url: Optional[str] = None,
    backend: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Enqueue a semantic footage search (mirrors the ``search_footage`` tool)."""
    payload = _with_filters({"query": query, "top_k": top_k}, filters)
    return submit_job(
        "search_footage",
        payload,
        database_url=database_url,
        backend=backend,
        idempotency_key=idempotency_key,
    )


def submit_search_script_beat(
    beat_text: str,
    *,
    top_k: int = 10,
    rerank: bool = False,
    confidence_floor: Optional[float] = None,
    filters: Optional[dict[str, Any]] = None,
    database_url: Optional[str] = None,
    backend: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Enqueue a multi-query script-beat search (mirrors ``search_script_beat``)."""
    payload = _with_filters(
        {
            "beat_text": beat_text,
            "top_k": top_k,
            "rerank": rerank,
            "confidence_floor": confidence_floor,
        },
        filters,
    )
    return submit_job(
        "search_script_beat",
        payload,
        database_url=database_url,
        backend=backend,
        idempotency_key=idempotency_key,
    )


def submit_fine_localize_clip(
    chunk_id: str,
    query: str,
    *,
    database_url: Optional[str] = None,
    backend: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Enqueue frame-level cut refinement (mirrors ``fine_localize_clip``)."""
    return submit_job(
        "fine_localize_clip",
        {"chunk_id": chunk_id, "query": query},
        database_url=database_url,
        backend=backend,
        idempotency_key=idempotency_key,
    )


def _build_payload(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Turn CLI flags into a (task, payload) pair."""
    task = args.task
    payload: dict[str, Any] = {}

    if args.payload:
        try:
            payload.update(json.loads(args.payload))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--payload must be valid JSON: {exc}") from exc

    if task == "search_footage":
        if not args.query:
            raise SystemExit("search_footage requires --query")
        payload.setdefault("query", args.query)
        payload.setdefault("top_k", args.top_k)
    elif task == "search_script_beat":
        beat = args.beat or args.query
        if not beat:
            raise SystemExit("search_script_beat requires --beat (or --query)")
        payload.setdefault("beat_text", beat)
        payload.setdefault("top_k", args.top_k)
        payload.setdefault("rerank", bool(args.rerank))
        if args.confidence_floor is not None:
            payload.setdefault("confidence_floor", args.confidence_floor)
    elif task == "fine_localize_clip":
        if not args.chunk_id or not args.query:
            raise SystemExit("fine_localize_clip requires --chunk-id and --query")
        payload.setdefault("chunk_id", args.chunk_id)
        payload.setdefault("query", args.query)
    elif task == "get_clip_details":
        if not args.chunk_id:
            raise SystemExit("get_clip_details requires --chunk-id")
        payload.setdefault("chunk_id", args.chunk_id)
    elif task == "get_media_item_details":
        if not args.media_item_id:
            raise SystemExit("get_media_item_details requires --media-item-id")
        payload.setdefault("media_item_id", args.media_item_id)

    filters = {
        "media_type": args.media_type,
        "orientation": args.orientation,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "provider": args.provider,
        "entity_name": args.entity_name,
        "entity_id": args.entity_id,
    }
    _with_filters(payload, filters)
    return task, payload


def main() -> int:
    """CLI entrypoint for submitting (and optionally awaiting) a job."""
    parser = argparse.ArgumentParser(
        description="Submit a footage-search job to the Footage Engine queue."
    )
    parser.add_argument(
        "--task",
        default="search_footage",
        choices=sorted(available_tasks()),
        help="Job type to enqueue (default: search_footage)",
    )
    parser.add_argument("--query", default=None, help="Natural language query")
    parser.add_argument("--beat", default=None, help="Narration/script beat for search_script_beat")
    parser.add_argument("--chunk-id", default=None, help="Target chunk id")
    parser.add_argument("--media-item-id", default=None, help="Target media item id")
    parser.add_argument("--top-k", type=int, default=10, help="Max results to return (default: 10)")
    parser.add_argument("--rerank", action="store_true", help="Apply the LLM judge reranker (search_script_beat)")
    parser.add_argument("--confidence-floor", type=float, default=None, help="Minimum score filter")
    parser.add_argument("--media-type", default=None, choices=["video", "image"])
    parser.add_argument("--orientation", default=None, help="landscape/horizontal or portrait/vertical")
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--provider", default=None, help="pexels, pixabay, coverr, youtube, manual")
    parser.add_argument("--entity-name", default=None)
    parser.add_argument("--entity-id", default=None)
    parser.add_argument(
        "--backend",
        default=None,
        help="Restrict the job to a worker backend ('qwen' or 'xclip'); default any",
    )
    parser.add_argument("--idempotency-key", default=None, help="Deduplicate repeated submissions")
    parser.add_argument("--payload", default=None, help="Raw JSON payload, merged over the flags")
    parser.add_argument("--wait", action="store_true", help="Block until the job finishes, then print the result")
    parser.add_argument("--timeout", type=float, default=300.0, help="--wait timeout in seconds (default: 300)")
    parser.add_argument("--stats", action="store_true", help="Print queue depth and live workers, then exit")
    args = parser.parse_args()

    settings = get_settings()
    database_url = settings.DATABASE_URL
    init_db(database_url)

    if args.stats:
        status = JobStatus.PENDING
        print(
            json.dumps(
                {
                    "pending_jobs": count_jobs(database_url, status=status),
                    "processing_jobs": count_jobs(database_url, status=JobStatus.PROCESSING),
                    "done_jobs": count_jobs(database_url, status=JobStatus.DONE),
                    "failed_jobs": count_jobs(database_url, status=JobStatus.FAILED),
                    "live_workers": count_live_workers(database_url),
                    "workers": list_workers(database_url),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    task, payload = _build_payload(args)
    job_id = enqueue_job(
        database_url,
        task=task,
        payload=payload,
        backend=args.backend,
        idempotency_key=args.idempotency_key,
    )
    print(json.dumps({"job_id": job_id, "task": task, "payload": payload}, indent=2, default=str))

    if not args.wait:
        return 0

    try:
        job = wait_for_job(database_url, job_id, timeout_sec=args.timeout, poll_interval_sec=1.0)
    except TimeoutError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    if job["status"] != JobStatus.DONE.value:
        print(f"\nERROR: job {job_id} ended as '{job['status']}': {job.get('error')}", file=sys.stderr)
        return 1

    print(json.dumps(job.get("result"), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
