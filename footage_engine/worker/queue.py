"""Job queue primitives: enqueue, atomic claim, lease renewal, completion.

Design notes
------------
* Claiming is a compare-and-swap ``UPDATE`` on the job row. Row-level atomicity
  is all the mutual exclusion required, and it behaves identically on SQLite and
  PostgreSQL. ``SELECT ... FOR UPDATE SKIP LOCKED`` is deliberately avoided:
  SQLite silently ignores it, so such a locking mistake would only ever surface
  in production.
* The queue is plain SQL. A caller in any language can enqueue by inserting a
  row into ``jobs``; nothing but the database is shared.
* Every claim carries a lease. If a worker dies (a killed Colab/Kaggle VM, for
  example) its jobs become claimable again once the lease expires.
"""

import logging
import os
import socket
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError

from footage_engine.config import get_settings
from footage_engine.models.db import get_db_session
from footage_engine.models.jobs import Job, JobStatus, Worker, WorkerStatus
from footage_engine.models.media import utc_now

logger = logging.getLogger(__name__)

DEFAULT_LEASE_SEC = 900
DEFAULT_WORKER_STALE_SEC = 10
TERMINAL_STATUSES = (JobStatus.DONE, JobStatus.FAILED)
MAX_ERROR_LEN = 8000


def default_stale_sec() -> int:
    """Liveness window used when no explicit one is given.

    Derived from ``WORKER_HEARTBEAT_SEC`` (two intervals plus slack) because it
    must exceed the heartbeat interval: with a smaller window a perfectly
    healthy worker looks dead for most of every heartbeat cycle.
    """
    try:
        heartbeat = int(get_settings().WORKER_HEARTBEAT_SEC)
    except Exception:  # pragma: no cover - settings should always load
        heartbeat = 30
    return max(DEFAULT_WORKER_STALE_SEC, heartbeat * 2 + 2)


def _iso(value: Optional[datetime]) -> Optional[str]:
    """Render a datetime as an ISO-8601 string for JSON transport."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _status_value(status: Any) -> str:
    return status.value if hasattr(status, "value") else str(status)


def job_to_dict(job: Job) -> dict[str, Any]:
    """Serialize a Job row into a plain JSON-friendly dict."""
    return {
        "id": job.id,
        "task": job.task,
        "payload": job.payload or {},
        "status": _status_value(job.status),
        "result": job.result,
        "error": job.error,
        "attempts": job.attempts,
        "backend": job.backend,
        "idempotency_key": job.idempotency_key,
        "picked_by": job.picked_by,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "completed_at": _iso(job.completed_at),
    }


def worker_to_dict(worker: Worker) -> dict[str, Any]:
    """Serialize a Worker row into a plain JSON-friendly dict."""
    return {
        "id": worker.id,
        "hostname": worker.hostname,
        "backend": worker.backend,
        "device": worker.device,
        "pid": worker.pid,
        "concurrency": worker.concurrency,
        "status": _status_value(worker.status),
        "current_job_id": worker.current_job_id,
        "started_at": _iso(worker.started_at),
        "last_heartbeat": _iso(worker.last_heartbeat),
        "stats": worker.stats or {},
    }


def make_worker_id(backend: str = "xclip", prefix: Optional[str] = None) -> str:
    """Build a unique, human-readable worker id (host:pid:backend:nonce)."""
    host = prefix or socket.gethostname()
    return f"{host}:{os.getpid()}:{backend}:{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Producer side
# ---------------------------------------------------------------------------


def enqueue_job(
    database_url: Optional[str] = None,
    task: str = "",
    payload: Optional[dict[str, Any]] = None,
    *,
    backend: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    job_id: Optional[str] = None,
) -> str:
    """Insert a new pending job and return its id.

    When ``idempotency_key`` is supplied and a job with that key already exists,
    the existing job id is returned instead of creating a duplicate.
    """
    if not task:
        raise ValueError("enqueue_job requires a non-empty 'task' name.")
    if payload is not None and not isinstance(payload, dict):
        raise TypeError("payload must be a dict (JSON object).")

    def _existing(session) -> Optional[str]:
        if not idempotency_key:
            return None
        return session.execute(
            select(Job.id).where(Job.idempotency_key == idempotency_key)
        ).scalar_one_or_none()

    try:
        with get_db_session(database_url) as session:
            found = _existing(session)
            if found:
                return found
            job = Job(
                task=task,
                payload=payload or {},
                backend=backend,
                idempotency_key=idempotency_key,
                status=JobStatus.PENDING,
            )
            if job_id:
                job.id = job_id
            session.add(job)
            session.flush()
            return job.id
    except IntegrityError:
        # Concurrent submit with the same idempotency key won the race.
        with get_db_session(database_url) as session:
            found = _existing(session)
            if found:
                logger.info(f"Idempotent enqueue hit existing job {found}.")
                return found
        raise


# ---------------------------------------------------------------------------
# Consumer side
# ---------------------------------------------------------------------------


def _claimable_clause(now: datetime):
    """Predicate: a job is up for grabs (never claimed, or its lease lapsed)."""
    return or_(
        Job.status == JobStatus.PENDING,
        and_(
            Job.status == JobStatus.PROCESSING,
            or_(Job.lease_expires_at.is_(None), Job.lease_expires_at < now),
        ),
    )


def claim_job(
    database_url: Optional[str] = None,
    worker_id: str = "",
    *,
    backend: Optional[str] = None,
    tasks: Optional[Iterable[str]] = None,
    lease_sec: int = DEFAULT_LEASE_SEC,
    exclude_ids: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    """Atomically claim the oldest eligible job, or return None if there is none.

    ``backend`` restricts claims to jobs targeting that embedding backend (jobs
    with ``backend IS NULL`` are servable by anyone). A row is won only if the
    compare-and-swap update reports a row change, so concurrent workers can never
    own the same job.
    """
    now = utc_now()
    lease_until = now + timedelta(seconds=lease_sec)
    claimed_id: Optional[str] = None

    with get_db_session(database_url) as session:
        claimable = _claimable_clause(now)
        stmt = select(Job.id).where(claimable).order_by(Job.created_at.asc())
        if backend is not None:
            stmt = stmt.where(or_(Job.backend.is_(None), Job.backend == backend))
        if tasks:
            task_list = list(tasks)
            if task_list:
                stmt = stmt.where(Job.task.in_(task_list))
        if exclude_ids:
            excluded = list(exclude_ids)
            if excluded:
                stmt = stmt.where(Job.id.notin_(excluded))

        candidates = list(session.execute(stmt).scalars().all())

        for candidate in candidates:
            result = session.execute(
                update(Job)
                .where(Job.id == candidate, claimable)
                .values(
                    status=JobStatus.PROCESSING,
                    picked_by=worker_id,
                    attempts=Job.attempts + 1,
                    started_at=now,
                    lease_expires_at=lease_until,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount:
                claimed_id = candidate
                break

    if claimed_id is None:
        return None

    # Read back in a fresh session so the committed claim is what we return.
    with get_db_session(database_url) as session:
        job = session.get(Job, claimed_id)
        return job_to_dict(job) if job else None


def heartbeat_worker(
    database_url: Optional[str] = None,
    worker_id: str = "",
    *,
    lease_sec: int = DEFAULT_LEASE_SEC,
    status: Optional[WorkerStatus] = None,
    current_job_id: Optional[str] = None,
    clear_current_job: bool = False,
    stats: Optional[dict[str, Any]] = None,
    renew_jobs: bool = True,
) -> int:
    """Refresh worker liveness and renew the lease on all its in-flight jobs.

    Renewing every job owned by this worker (rather than one id) keeps the
    bookkeeping correct when ``concurrency`` is greater than one. Returns the
    number of job leases renewed.
    """
    now = utc_now()
    renewed = 0
    with get_db_session(database_url) as session:
        worker_values: dict[str, Any] = {"last_heartbeat": now}
        if status is not None:
            worker_values["status"] = status
        if clear_current_job:
            worker_values["current_job_id"] = None
        elif current_job_id is not None:
            worker_values["current_job_id"] = current_job_id
        if stats is not None:
            worker_values["stats"] = dict(stats)

        session.execute(
            update(Worker)
            .where(Worker.id == worker_id)
            .values(**worker_values)
            .execution_options(synchronize_session=False)
        )

        if renew_jobs:
            result = session.execute(
                update(Job)
                .where(Job.picked_by == worker_id, Job.status == JobStatus.PROCESSING)
                .values(lease_expires_at=now + timedelta(seconds=lease_sec))
                .execution_options(synchronize_session=False)
            )
            renewed = result.rowcount or 0
    return renewed


def complete_job(
    database_url: Optional[str] = None,
    job_id: str = "",
    result: Optional[dict[str, Any]] = None,
    *,
    worker_id: Optional[str] = None,
) -> bool:
    """Mark a job DONE and store its JSON result. Returns True if it was updated."""
    with get_db_session(database_url) as session:
        stmt = update(Job).where(Job.id == job_id)
        if worker_id is not None:
            stmt = stmt.where(Job.picked_by == worker_id)
        updated = session.execute(
            stmt.values(
                status=JobStatus.DONE,
                result=result,
                error=None,
                completed_at=utc_now(),
                lease_expires_at=None,
            ).execution_options(synchronize_session=False)
        )
        return bool(updated.rowcount)


def fail_job(
    database_url: Optional[str] = None,
    job_id: str = "",
    error: str = "",
    *,
    worker_id: Optional[str] = None,
    requeue: bool = False,
) -> bool:
    """Mark a job FAILED (or return it to PENDING when ``requeue`` is set)."""
    with get_db_session(database_url) as session:
        stmt = update(Job).where(Job.id == job_id)
        if worker_id is not None:
            stmt = stmt.where(Job.picked_by == worker_id)
        message = (error or "")[:MAX_ERROR_LEN]
        values: dict[str, Any] = {
            "error": message,
            "picked_by": None,
            "lease_expires_at": None,
        }
        if requeue:
            values["status"] = JobStatus.PENDING
        else:
            values["status"] = JobStatus.FAILED
            values["completed_at"] = utc_now()
        updated = session.execute(
            stmt.values(**values).execution_options(synchronize_session=False)
        )
        return bool(updated.rowcount)


def release_jobs(database_url: Optional[str] = None, worker_id: str = "") -> int:
    """Return a worker's in-flight jobs to PENDING (used on graceful shutdown)."""
    with get_db_session(database_url) as session:
        result = session.execute(
            update(Job)
            .where(Job.picked_by == worker_id, Job.status == JobStatus.PROCESSING)
            .values(
                status=JobStatus.PENDING,
                picked_by=None,
                lease_expires_at=None,
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount or 0


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def get_job(database_url: Optional[str] = None, job_id: str = "") -> Optional[dict[str, Any]]:
    """Fetch one job as a dict, or None if it does not exist."""
    with get_db_session(database_url) as session:
        job = session.get(Job, job_id)
        return job_to_dict(job) if job else None


def wait_for_job(
    database_url: Optional[str] = None,
    job_id: str = "",
    *,
    timeout_sec: float = 300.0,
    poll_interval_sec: float = 1.0,
) -> dict[str, Any]:
    """Poll until a job reaches a terminal state. Raises TimeoutError otherwise."""
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        job = get_job(database_url, job_id)
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")
        status = job.get("status")
        if status in {_status_value(s) for s in TERMINAL_STATUSES}:
            return job
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_sec}s waiting for job '{job_id}' (status={status})."
            )
        time.sleep(poll_interval_sec)


def count_jobs(
    database_url: Optional[str] = None,
    *,
    status: Optional[JobStatus] = None,
    task: Optional[str] = None,
    backend: Optional[str] = None,
) -> int:
    """Count jobs matching the given filters."""
    stmt = select(func.count()).select_from(Job)
    if status is not None:
        stmt = stmt.where(Job.status == status)
    if task is not None:
        stmt = stmt.where(Job.task == task)
    if backend is not None:
        stmt = stmt.where(or_(Job.backend.is_(None), Job.backend == backend))
    with get_db_session(database_url) as session:
        return int(session.execute(stmt).scalar_one())


def count_live_workers(
    database_url: Optional[str] = None,
    *,
    within_sec: Optional[int] = None,
    backend: Optional[str] = None,
) -> int:
    """Count workers whose heartbeat is fresh enough to be considered alive.

    Callers use this before enqueueing to find out whether a GPU worker is up.
    """
    window = within_sec if within_sec is not None else default_stale_sec()
    cutoff = utc_now() - timedelta(seconds=window)
    stmt = select(func.count()).select_from(Worker).where(Worker.last_heartbeat >= cutoff)
    if backend is not None:
        stmt = stmt.where(Worker.backend == backend)
    with get_db_session(database_url) as session:
        return int(session.execute(stmt).scalar_one())


def list_workers(
    database_url: Optional[str] = None,
    *,
    within_sec: Optional[int] = None,
    backend: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List registered workers, optionally only those with a fresh heartbeat."""
    stmt = select(Worker).order_by(Worker.last_heartbeat.desc())
    if within_sec is not None:
        stmt = stmt.where(Worker.last_heartbeat >= utc_now() - timedelta(seconds=within_sec))
    if backend is not None:
        stmt = stmt.where(Worker.backend == backend)
    with get_db_session(database_url) as session:
        return [worker_to_dict(w) for w in session.execute(stmt).scalars().all()]


def register_worker(
    database_url: Optional[str] = None,
    worker_id: str = "",
    *,
    hostname: Optional[str] = None,
    backend: str = "xclip",
    device: Optional[str] = None,
    pid: Optional[int] = None,
    concurrency: int = 1,
    status: WorkerStatus = WorkerStatus.IDLE,
    stats: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Insert or refresh this worker's registry row and return it as a dict."""
    now = utc_now()
    with get_db_session(database_url) as session:
        worker = session.get(Worker, worker_id)
        if worker is None:
            worker = Worker(
                id=worker_id,
                hostname=hostname or socket.gethostname(),
                backend=backend,
                device=device,
                pid=pid if pid is not None else os.getpid(),
                concurrency=concurrency,
                status=status,
                started_at=now,
                last_heartbeat=now,
                stats=stats or {},
            )
            session.add(worker)
        else:
            worker.hostname = hostname or socket.gethostname()
            worker.backend = backend
            worker.device = device
            worker.pid = pid if pid is not None else os.getpid()
            worker.concurrency = concurrency
            worker.status = status
            worker.last_heartbeat = now
            worker.stats = stats or worker.stats or {}
        session.flush()
        return worker_to_dict(worker)


def set_worker_status(
    database_url: Optional[str] = None,
    worker_id: str = "",
    status: WorkerStatus = WorkerStatus.IDLE,
) -> bool:
    """Update just the status column of a worker row."""
    with get_db_session(database_url) as session:
        result = session.execute(
            update(Worker)
            .where(Worker.id == worker_id)
            .values(status=status, last_heartbeat=utc_now())
            .execution_options(synchronize_session=False)
        )
        return bool(result.rowcount)


__all__ = [
    "DEFAULT_LEASE_SEC",
    "DEFAULT_WORKER_STALE_SEC",
    "default_stale_sec",
    "TERMINAL_STATUSES",
    "make_worker_id",
    "job_to_dict",
    "worker_to_dict",
    "enqueue_job",
    "claim_job",
    "heartbeat_worker",
    "complete_job",
    "fail_job",
    "release_jobs",
    "get_job",
    "wait_for_job",
    "count_jobs",
    "count_live_workers",
    "list_workers",
    "register_worker",
    "set_worker_status",
]
