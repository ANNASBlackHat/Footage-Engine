"""Tests for the asynchronous job worker: queue, leases, tasks, and MCP parity.

Everything runs offline with MockEmbedder + InMemoryVectorStore: no model
downloads, no GPU, no network.
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select, update

from footage_engine.config import Settings
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.mcp.server import create_mcp_server
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.jobs import Job, JobStatus, Worker, WorkerStatus
from footage_engine.models.media import MediaItem, MediaStatus
from footage_engine.models.media import utc_now
from footage_engine.pipeline.processor import BatchProcessor
from footage_engine.storage.local import LocalStorageBackend
from footage_engine.vector.in_memory import InMemoryVectorStore
from footage_engine.worker.client import submit_search_footage
from footage_engine.worker.queue import (
    claim_job,
    complete_job,
    count_jobs,
    count_live_workers,
    default_stale_sec,
    enqueue_job,
    fail_job,
    get_job,
    heartbeat_worker,
    list_workers,
    register_worker,
    release_jobs,
)
from footage_engine.worker.runner import JobWorker
from footage_engine.worker.tasks import available_tasks, run_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_tool_result(res):
    """Extract parsed Python objects from MCP tool responses."""
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
def worker_env(temp_dir):
    """Isolated database, storage, and in-memory vector store plus shared handles."""
    db_path = os.path.join(temp_dir, "worker_test.db")
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

    return {
        "settings": settings,
        "database_url": settings.DATABASE_URL,
        "storage": LocalStorageBackend(base_dir=settings.LOCAL_STORAGE_DIR),
        "embedder": MockEmbedder(dimension=512),
        "vector_store": InMemoryVectorStore(),
        "temp_dir": temp_dir,
    }


def _make_worker(env, **kwargs):
    kwargs.setdefault("poll_interval_sec", 0.1)
    kwargs.setdefault("worker_id", "test-worker")
    return JobWorker(
        settings=env["settings"],
        storage=env["storage"],
        embedder=env["embedder"],
        vector_store=env["vector_store"],
        database_url=env["database_url"],
        use_mock=True,
        **kwargs,
    )


def _seed_media(env, filename="ocean.mp4", provider="pexels", duration=20.0, resolution="1920x1080"):
    """Create a pending MediaItem and index it so searches have something to find."""
    env["storage"].save_file(b"test video payload", filename)
    with get_db_session(env["database_url"]) as session:
        item = MediaItem(
            provider=provider,
            source_url=f"https://example.com/{filename}",
            storage_path=filename,
            duration_sec=duration,
            resolution=resolution,
            license_type="pexels_free",
            status=MediaStatus.PENDING,
        )
        session.add(item)
        session.flush()
        media_id = item.id

    processor = BatchProcessor(
        settings=env["settings"],
        storage=env["storage"],
        embedder=env["embedder"],
        vector_store=env["vector_store"],
        database_url=env["database_url"],
    )
    processor.process_all_pending()
    return media_id


def _force_lease_into_past(env, job_id, seconds=30):
    """Simulate a worker that died mid-job (its lease lapsed)."""
    with get_db_session(env["database_url"]) as session:
        session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(lease_expires_at=utc_now() - timedelta(seconds=seconds))
            .execution_options(synchronize_session=False)
        )


# ---------------------------------------------------------------------------
# Queue primitives
# ---------------------------------------------------------------------------


def test_enqueue_creates_pending_job(worker_env):
    job_id = enqueue_job(worker_env["database_url"], "search_footage", {"query": "ocean"})
    job = get_job(worker_env["database_url"], job_id)
    assert job["status"] == "pending"
    assert job["task"] == "search_footage"
    assert job["payload"] == {"query": "ocean"}
    assert job["attempts"] == 0
    assert job["result"] is None


def test_enqueue_rejects_empty_or_bad_input(worker_env):
    url = worker_env["database_url"]
    with pytest.raises(ValueError):
        enqueue_job(url, "", {"query": "x"})
    with pytest.raises(TypeError):
        enqueue_job(url, "search_footage", ["not", "a", "dict"])


def test_claim_job_is_exclusive(worker_env):
    """Two workers racing for one job: exactly one wins."""
    url = worker_env["database_url"]
    job_id = enqueue_job(url, "search_footage", {"query": "ocean"})

    first = claim_job(url, "worker-a")
    assert first is not None
    assert first["id"] == job_id
    assert first["status"] == "processing"
    assert first["picked_by"] == "worker-a"
    assert first["attempts"] == 1

    assert claim_job(url, "worker-b") is None
    assert count_jobs(url, status=JobStatus.PENDING) == 0


def test_concurrent_claim_of_single_job_has_one_winner(worker_env):
    """Barrier-synchronised racers on one job: exactly one may own it.

    Every thread reads the same (pending) candidate list before any of them
    updates, so this fails unless the compare-and-swap update is doing the
    mutual exclusion.
    """
    url = worker_env["database_url"]
    job_id = enqueue_job(url, "search_footage", {"query": "ocean"})

    racers = 5
    barrier = threading.Barrier(racers)
    winners: list[str] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def _race(index: int) -> None:
        try:
            barrier.wait(timeout=15)
            claimed = claim_job(url, f"racer-{index}")
            if claimed is not None:
                with lock:
                    winners.append(claimed["id"])
        except Exception as exc:  # pragma: no cover - surfaced via assert
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_race, args=(i,)) for i in range(racers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert winners == [job_id]
    assert get_job(url, job_id)["attempts"] == 1


def test_lease_expiry_allows_reclaim(worker_env):
    """A job whose worker vanished becomes claimable again by someone else."""
    url = worker_env["database_url"]
    job_id = enqueue_job(url, "search_footage", {"query": "ocean"})
    claim_job(url, "worker-a", lease_sec=60)
    assert claim_job(url, "worker-b") is None

    _force_lease_into_past(worker_env, job_id)

    reclaimed = claim_job(url, "worker-b")
    assert reclaimed is not None
    assert reclaimed["id"] == job_id
    assert reclaimed["picked_by"] == "worker-b"
    assert reclaimed["attempts"] == 2


def test_heartbeat_renews_active_lease(worker_env):
    """The heartbeat extends the lease of every job this worker owns."""
    url = worker_env["database_url"]
    job_id = enqueue_job(url, "search_footage", {"query": "ocean"})
    register_worker(url, "worker-a", backend="xclip", concurrency=1)
    claim_job(url, "worker-a", lease_sec=60)

    _force_lease_into_past(worker_env, job_id)

    renewed = heartbeat_worker(url, "worker-a", lease_sec=60)
    assert renewed == 1

    job = get_job(url, job_id)
    assert job["status"] == "processing"
    # A renewed lease must be in the future, so nobody else can steal the job.
    assert claim_job(url, "worker-b") is None


def test_complete_job_stores_result(worker_env):
    url = worker_env["database_url"]
    job_id = enqueue_job(url, "search_footage", {"query": "ocean"})
    claim_job(url, "worker-a")

    assert complete_job(url, job_id, {"count": 2, "results": []}, worker_id="worker-a") is True
    job = get_job(url, job_id)
    assert job["status"] == "done"
    assert job["result"] == {"count": 2, "results": []}
    assert job["error"] is None
    assert job["completed_at"] is not None



def test_complete_job_requires_ownership(worker_env):
    """A worker that lost its lease cannot overwrite the result."""
    url = worker_env["database_url"]
    job_id = enqueue_job(url, "search_footage", {"query": "ocean"})
    claim_job(url, "worker-a")
    assert complete_job(url, job_id, {"count": 1}, worker_id="someone-else") is False
    assert get_job(url, job_id)["status"] == "processing"


def test_fail_job_records_error(worker_env):
    url = worker_env["database_url"]
    job_id = enqueue_job(url, "search_footage", {"query": "ocean"})
    claim_job(url, "worker-a")
    assert fail_job(url, job_id, "ValueError: boom", worker_id="worker-a") is True
    job = get_job(url, job_id)
    assert job["status"] == "failed"
    assert "boom" in job["error"]


def test_release_jobs_returns_work_to_queue(worker_env):
    url = worker_env["database_url"]
    first = enqueue_job(url, "search_footage", {"query": "a"})
    second = enqueue_job(url, "search_footage", {"query": "b"})
    claim_job(url, "worker-a")
    claim_job(url, "worker-a")

    assert release_jobs(url, "worker-a") == 2
    assert count_jobs(url, status=JobStatus.PENDING) == 2
    reclaimed = claim_job(url, "worker-b")
    assert reclaimed["id"] in {first, second}


def test_idempotency_key_dedupes_enqueue(worker_env):
    url = worker_env["database_url"]
    first = enqueue_job(url, "search_footage", {"query": "ocean"}, idempotency_key="req-123")
    second = enqueue_job(url, "search_footage", {"query": "ocean"}, idempotency_key="req-123")
    assert first == second
    assert count_jobs(url) == 1


def test_claim_respects_backend_scope(worker_env):
    """A Qwen worker must not steal an X-CLIP job (separate vector collections)."""
    url = worker_env["database_url"]
    enqueue_job(url, "search_footage", {"query": "ocean"}, backend="xclip")
    assert claim_job(url, "qwen-worker", backend="qwen") is None
    claimed = claim_job(url, "xclip-worker", backend="xclip")
    assert claimed is not None


def test_claim_respects_task_allowlist(worker_env):
    url = worker_env["database_url"]
    enqueue_job(url, "fine_localize_clip", {"chunk_id": "c1", "query": "q"})
    assert claim_job(url, "w", tasks=["search_footage"]) is None
    assert claim_job(url, "w", tasks=["fine_localize_clip"]) is not None


def test_claim_oldest_first(worker_env):
    url = worker_env["database_url"]
    first = enqueue_job(url, "search_footage", {"query": "first"})
    enqueue_job(url, "search_footage", {"query": "second"})
    assert claim_job(url, "w")["id"] == first


def test_claim_excludes_in_flight_ids(worker_env):
    url = worker_env["database_url"]
    first = enqueue_job(url, "search_footage", {"query": "first"})
    second = enqueue_job(url, "search_footage", {"query": "second"})
    assert claim_job(url, "w", exclude_ids=[first])["id"] == second


def test_worker_registry_and_liveness(worker_env):
    url = worker_env["database_url"]
    assert count_live_workers(url) == 0

    worker = _make_worker(worker_env, worker_id="gpu-1")
    info = worker.register()
    assert info["id"] == "gpu-1"
    assert info["backend"] == "xclip"
    assert info["status"] == "idle"

    assert count_live_workers(url) == 1
    assert count_live_workers(url, backend="qwen") == 0

    renewed = worker._heartbeat_once()
    assert renewed == 0  # nothing in flight yet
    assert list_workers(url)[0]["id"] == "gpu-1"


def test_stale_worker_is_not_live(worker_env):
    url = worker_env["database_url"]
    register_worker(url, "old-worker", backend="qwen")
    with get_db_session(url) as session:
        session.execute(
            update(Worker)
            .where(Worker.id == "old-worker")
            .values(last_heartbeat=utc_now() - timedelta(seconds=120))
            .execution_options(synchronize_session=False)
        )
    assert count_live_workers(url) == 0
    assert count_live_workers(url, within_sec=600) == 1


def test_raw_sql_insert_from_another_language_is_executed(worker_env):
    """The documented integration contract for non-Python callers.

    An external service only supplies (id, task, payload); status, attempts and
    created_at come from server defaults, so a JS/Go/curl client needs no
    knowledge of the ORM. Stored statuses are the enum NAMES, uppercase.
    """
    url = worker_env["database_url"]
    db_path = url.replace("sqlite:///", "")
    _seed_media(worker_env)

    job_id = str(uuid.uuid4())
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO jobs (id, task, payload) VALUES (?, ?, ?)",
            (job_id, "search_footage", json.dumps({"query": "ocean", "top_k": 1})),
        )
        conn.commit()
        status, attempts = conn.execute(
            "SELECT status, attempts FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()

    assert status == "PENDING"  # enum name, uppercase
    assert attempts == 0

    stats = _make_worker(worker_env).run(max_jobs=1, idle_exit_sec=5)
    assert stats["processed"] == 1

    with sqlite3.connect(db_path) as conn:
        status, attempts, result, error = conn.execute(
            "SELECT status, attempts, result, error FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()

    assert status == "DONE"
    assert attempts == 1
    assert error is None
    assert json.loads(result)["count"] == 1



# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


def test_live_window_exceeds_the_heartbeat_interval(worker_env):
    """Regression: a healthy worker must not look dead between heartbeats.

    The default liveness window used to be a 10s constant while the heartbeat
    interval defaults to 30s, so a healthy worker would be reported dead for most
    of every cycle - exactly what `submit_job.py --stats` shows the user.
    """
    url = worker_env["database_url"]
    window = default_stale_sec()
    assert window > worker_env["settings"].WORKER_HEARTBEAT_SEC

    register_worker(url, "hb-worker", backend="qwen")
    # Backdated by nearly a full window: still live under the derived window, yet
    # invisible under the old hardcoded 10s one.
    with get_db_session(url) as session:
        session.execute(
            update(Worker)
            .where(Worker.id == "hb-worker")
            .values(last_heartbeat=utc_now() - timedelta(seconds=window - 5))
            .execution_options(synchronize_session=False)
        )

    assert count_live_workers(url) == 1


def test_available_tasks_match_mcp_tool_names():
    """Worker task names must be a subset of the MCP tool surface."""
    mcp_tools = {
        "search_footage",
        "search_script_beat",
        "fine_localize_clip",
        "get_clip_details",
        "get_media_item_details",
        "ingest_url",
        "ingest_keywords",
        "process_pending_queue",
        "get_library_stats",
        "list_entities",
        "resolve_or_create_entity",
    }
    assert set(available_tasks()) <= mcp_tools
    assert "search_footage" in available_tasks()


def test_run_task_rejects_unknown_task(worker_env):
    worker = _make_worker(worker_env)
    with pytest.raises(ValueError) as excinfo:
        run_task(worker.task_ctx, "no_such_task", {})
    assert "Unknown task" in str(excinfo.value)


def test_search_task_requires_query(worker_env):
    worker = _make_worker(worker_env)
    with pytest.raises(ValueError) as excinfo:
        run_task(worker.task_ctx, "search_footage", {})
    assert "query" in str(excinfo.value)


def test_get_clip_details_missing_chunk_raises(worker_env):
    worker = _make_worker(worker_env)
    with pytest.raises(ValueError) as excinfo:
        run_task(worker.task_ctx, "get_clip_details", {"chunk_id": "does-not-exist"})
    assert "not found" in str(excinfo.value)


# ---------------------------------------------------------------------------
# End-to-end worker runs
# ---------------------------------------------------------------------------


def test_worker_executes_search_job(worker_env):
    media_id = _seed_media(worker_env)
    job_id = submit_search_footage("ocean waves", top_k=5, database_url=worker_env["database_url"])

    worker = _make_worker(worker_env)
    stats = worker.run(max_jobs=1, idle_exit_sec=5)

    assert stats["claimed"] == 1
    assert stats["processed"] == 1
    assert stats["failed"] == 0

    job = get_job(worker_env["database_url"], job_id)
    assert job["status"] == "done"
    assert job["attempts"] == 1
    assert job["result"]["count"] == 1
    assert job["result"]["results"][0]["media_item_id"] == media_id


def test_worker_marks_job_failed_on_bad_payload(worker_env):
    """A malformed payload fails that job without killing the worker."""
    url = worker_env["database_url"]
    bad = enqueue_job(url, "search_footage", {})  # no query

    worker = _make_worker(worker_env)
    stats = worker.run(max_jobs=1, idle_exit_sec=5)

    assert stats["failed"] == 1
    job = get_job(url, bad)
    assert job["status"] == "failed"
    assert "query" in job["error"]


def test_worker_marks_job_failed_on_unknown_task(worker_env):
    url = worker_env["database_url"]
    job_id = enqueue_job(url, "not_a_real_task", {"x": 1})
    worker = _make_worker(worker_env)
    worker.run(max_jobs=1, idle_exit_sec=5)
    job = get_job(url, job_id)
    assert job["status"] == "failed"
    assert "Unknown task" in job["error"]


def test_worker_continues_after_a_failure(worker_env):
    """One bad job must not stop the worker from draining the rest."""
    media_id = _seed_media(worker_env)
    url = worker_env["database_url"]
    enqueue_job(url, "not_a_real_task", {})
    good = enqueue_job(url, "search_footage", {"query": "ocean", "top_k": 3})

    worker = _make_worker(worker_env)
    stats = worker.run(max_jobs=2, idle_exit_sec=5)

    assert stats["claimed"] == 2
    assert stats["failed"] == 1
    assert stats["processed"] == 1
    assert get_job(url, good)["result"]["results"][0]["media_item_id"] == media_id


def test_worker_idle_exit_with_empty_queue(worker_env):
    worker = _make_worker(worker_env)
    stats = worker.run(idle_exit_sec=1)
    assert stats["claimed"] == 0
    assert stats["processed"] == 0
    assert stats["elapsed_sec"] >= 0


def test_worker_claims_work_enqueued_during_its_idle_window(worker_env):
    """Regression: idle-exit used to be evaluated before the poll that claims work.

    A worker whose idle-exit was <= its poll interval would wake up, decide it was
    idle, and exit while a job was already sitting in the queue.
    """
    _seed_media(worker_env)
    url = worker_env["database_url"]
    holder: dict[str, str] = {}

    def _submit_partway_through_a_poll_cycle():
        time.sleep(0.6)
        holder["id"] = enqueue_job(url, "search_footage", {"query": "ocean", "top_k": 1})

    worker = _make_worker(worker_env, poll_interval_sec=0.5)
    submitter = threading.Thread(target=_submit_partway_through_a_poll_cycle)
    submitter.start()
    stats = worker.run(max_jobs=1, idle_exit_sec=1)
    submitter.join(timeout=10)

    assert holder.get("id"), "the job was never submitted"
    assert stats["claimed"] == 1
    assert stats["processed"] == 1
    assert get_job(url, holder["id"])["status"] == "done"


def test_worker_respects_max_jobs(worker_env):
    _seed_media(worker_env)
    url = worker_env["database_url"]
    for i in range(4):
        enqueue_job(url, "search_footage", {"query": f"ocean {i}", "top_k": 1})

    worker = _make_worker(worker_env)
    stats = worker.run(max_jobs=2, idle_exit_sec=5)
    assert stats["claimed"] == 2
    assert count_jobs(url, status=JobStatus.PENDING) == 2


def test_two_workers_process_each_job_exactly_once(worker_env):
    """The core guarantee: no job is executed twice, even under contention."""
    _seed_media(worker_env)
    url = worker_env["database_url"]
    job_ids = [enqueue_job(url, "search_footage", {"query": f"ocean {i}", "top_k": 1}) for i in range(6)]

    workers = [
        _make_worker(worker_env, worker_id=f"worker-{i}", poll_interval_sec=0.05)
        for i in range(2)
    ]
    results = {}

    def _run(idx):
        results[idx] = workers[idx].run(max_jobs=3, idle_exit_sec=2)

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert results[0]["processed"] + results[1]["processed"] == 6
    assert results[0]["failed"] + results[1]["failed"] == 0

    with get_db_session(url) as session:
        jobs = list(session.execute(select(Job).where(Job.id.in_(job_ids))).scalars().all())

    assert len(jobs) == 6
    assert {j.status for j in jobs} == {JobStatus.DONE}
    # attempts == 1 proves no job was claimed (and therefore run) twice.
    assert {j.attempts for j in jobs} == {1}


def test_worker_marks_itself_stopping_after_run(worker_env):
    url = worker_env["database_url"]
    worker = _make_worker(worker_env, worker_id="stopper")
    worker.run(idle_exit_sec=1)
    with get_db_session(url) as session:
        row = session.get(Worker, "stopper")
        assert row.status == WorkerStatus.STOPPING


# ---------------------------------------------------------------------------
# MCP parity: a queued job must return exactly what the MCP tool returns
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_queued_job_matches_mcp_tool_result(worker_env):
    """The worker mirrors the MCP tool surface byte-for-byte."""
    _seed_media(worker_env)
    settings = worker_env["settings"]
    storage = worker_env["storage"]
    embedder = worker_env["embedder"]
    vector_store = worker_env["vector_store"]

    server = create_mcp_server(
        use_mock=True,
        settings=settings,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        database_url=settings.DATABASE_URL,
    )

    arguments = {"query": "ocean waves", "top_k": 5, "orientation": "landscape"}

    mcp_res = await server.call_tool("search_footage", arguments)
    assert not mcp_res.is_error
    mcp_data = _parse_tool_result(mcp_res)

    job_id = enqueue_job(settings.DATABASE_URL, "search_footage", arguments)
    worker = _make_worker(worker_env)
    worker.run(max_jobs=1, idle_exit_sec=5)
    job = get_job(settings.DATABASE_URL, job_id)

    assert job["status"] == "done"
    job_result = job["result"]

    assert job_result["query"] == mcp_data["query"]
    assert job_result["count"] == mcp_data["count"]
    assert job_result["count"] == 1
    # Compare the full payloads, order-insensitively, to prove parity.
    normalize = lambda payload: sorted(payload["results"], key=lambda r: r["chunk_id"])
    assert normalize(job_result) == normalize(mcp_data)


@pytest.mark.anyio
async def test_queued_clip_details_matches_mcp_tool_result(worker_env):
    media_id = _seed_media(worker_env)
    settings = worker_env["settings"]

    server = create_mcp_server(
        use_mock=True,
        settings=settings,
        storage=worker_env["storage"],
        embedder=worker_env["embedder"],
        vector_store=worker_env["vector_store"],
        database_url=settings.DATABASE_URL,
    )

    with get_db_session(settings.DATABASE_URL) as session:
        chunk_id = session.get(MediaItem, media_id).chunks[0].id

    mcp_res = await server.call_tool("get_clip_details", {"chunk_id": chunk_id})
    mcp_data = _parse_tool_result(mcp_res)

    job_id = enqueue_job(settings.DATABASE_URL, "get_clip_details", {"chunk_id": chunk_id})
    worker = _make_worker(worker_env)
    worker.run(max_jobs=1, idle_exit_sec=5)
    job = get_job(settings.DATABASE_URL, job_id)

    assert job["status"] == "done"
    assert job["result"] == mcp_data


# ---------------------------------------------------------------------------
# Producer convenience helpers
# ---------------------------------------------------------------------------


def test_client_helpers_build_mcp_shaped_payloads(worker_env):
    url = worker_env["database_url"]
    job_id = submit_search_footage(
        "harbour at dawn",
        top_k=3,
        filters={"orientation": "portrait", "media_type": "video", "provider": "pexels"},
        database_url=url,
        backend="qwen",
        idempotency_key="beat-1",
    )
    job = get_job(url, job_id)
    assert job["task"] == "search_footage"
    assert job["backend"] == "qwen"
    assert job["payload"] == {
        "query": "harbour at dawn",
        "top_k": 3,
        "orientation": "portrait",
        "media_type": "video",
        "provider": "pexels",
    }
