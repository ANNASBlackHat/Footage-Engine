"""Job worker runtime: claim jobs, execute them, renew leases, report progress.

A worker holds the embedding model and drains the shared ``jobs`` queue. Any
number of workers can run against the same database (laptop, Colab, Kaggle)
because claiming is atomic; a job is executed exactly once unless its worker
dies, in which case the lease lapses and the job is picked up again.
"""

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterable, Optional

from footage_engine.config import Settings, get_settings
from footage_engine.embeddings import backend_for, get_embedder
from footage_engine.embeddings.base import EmbeddingBackend
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.models.db import init_db
from footage_engine.models.jobs import JobStatus, WorkerStatus
from footage_engine.models.media import utc_now
from footage_engine.retrieval.api import RetrievalAPI
from footage_engine.storage import get_storage_backend
from footage_engine.storage.base import StorageBackend
from footage_engine.vector import get_vector_store
from footage_engine.vector.base import VectorStore
from footage_engine.worker.base import TaskContext
from footage_engine.worker.queue import (
    claim_job,
    complete_job,
    count_jobs,
    count_live_workers,
    fail_job,
    heartbeat_worker,
    make_worker_id,
    register_worker,
    release_jobs,
    set_worker_status,
)
from footage_engine.worker.tasks import available_tasks, run_task

logger = logging.getLogger(__name__)


class JobWorker:
    """Drains queued footage-search jobs with lease-based ownership."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        storage: Optional[StorageBackend] = None,
        embedder: Optional[EmbeddingBackend] = None,
        vector_store: Optional[VectorStore] = None,
        *,
        database_url: Optional[str] = None,
        backend: Optional[str] = None,
        tasks: Optional[Iterable[str]] = None,
        concurrency: Optional[int] = None,
        lease_sec: Optional[int] = None,
        heartbeat_sec: Optional[int] = None,
        poll_interval_sec: Optional[float] = None,
        idle_exit_sec: Optional[int] = None,
        worker_id: Optional[str] = None,
        use_mock: bool = False,
    ):
        self.settings = settings or get_settings()
        if backend is not None:
            self.settings.EMBEDDING_BACKEND = backend  # type: ignore[assignment]

        self.database_url = database_url or self.settings.DATABASE_URL
        init_db(self.database_url)

        self.storage = storage or get_storage_backend(self.settings)
        self.use_mock = (
            use_mock
            or os.environ.get("FOOTAGE_WORKER_USE_MOCK") == "1"
            or os.environ.get("USE_MOCK_EMBEDDER") == "1"
        )
        if embedder is not None:
            self.embedder = embedder
        elif self.use_mock:
            self.embedder = MockEmbedder(dimension=self.settings.EMBEDDING_DIMENSION)
        else:
            self.embedder = get_embedder(self.settings)

        self.backend = backend_for(self.settings, self.embedder)
        self.vector_store = vector_store or get_vector_store(self.settings, backend=self.backend)
        self.retrieval_api = RetrievalAPI(
            settings=self.settings,
            storage=self.storage,
            embedder=self.embedder,
            vector_store=self.vector_store,
            database_url=self.database_url,
        )

        self.tasks = set(tasks) if tasks else None
        self.concurrency = max(
            1,
            int(concurrency if concurrency is not None else self.settings.WORKER_CONCURRENCY),
        )
        self.lease_sec = int(lease_sec if lease_sec is not None else self.settings.WORKER_LEASE_SEC)
        self.heartbeat_sec = max(
            1, int(heartbeat_sec if heartbeat_sec is not None else self.settings.WORKER_HEARTBEAT_SEC)
        )
        self.poll_interval_sec = max(
            0.1,
            float(poll_interval_sec if poll_interval_sec is not None else self.settings.WORKER_POLL_INTERVAL_SEC),
        )
        self.idle_exit_sec = int(
            idle_exit_sec if idle_exit_sec is not None else self.settings.WORKER_IDLE_EXIT_SEC
        )
        self.worker_id = worker_id or make_worker_id(self.backend)

        self.task_ctx = TaskContext(
            settings=self.settings,
            storage=self.storage,
            embedder=self.embedder,
            vector_store=self.vector_store,
            retrieval_api=self.retrieval_api,
            database_url=self.database_url,
            backend=self.backend,
            worker_id=self.worker_id,
        )
        self.stats: dict[str, Any] = {"claimed": 0, "processed": 0, "failed": 0, "elapsed_sec": 0.0}

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._in_flight: set[str] = set()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._started_at = utc_now()

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    def describe_device(self) -> str:
        """Best-effort human-readable description of the compute device in use."""
        device = getattr(self.embedder, "device", None)
        if device is None:
            return "cpu"
        label = str(device)
        try:
            import torch

            if label.startswith("cuda") and torch.cuda.is_available():
                return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:  # torch optional / no GPU
            pass
        return label

    def register(self) -> dict[str, Any]:
        """Upsert this worker's registry row so callers can see it is alive."""
        return register_worker(
            self.database_url,
            self.worker_id,
            hostname=socket.gethostname(),
            backend=self.backend,
            device=self.describe_device(),
            pid=os.getpid(),
            concurrency=self.concurrency,
            status=WorkerStatus.IDLE,
            stats=self._stats_snapshot(),
        )

    def _stats_snapshot(self) -> dict[str, Any]:
        snapshot = {
            "claimed": self.stats["claimed"],
            "processed": self.stats["processed"],
            "failed": self.stats["failed"],
            "in_flight": len(self._in_flight),
        }
        return snapshot

    def _heartbeat_once(self) -> int:
        with self._lock:
            in_flight = list(self._in_flight)
        status = WorkerStatus.BUSY if in_flight else WorkerStatus.IDLE
        current = in_flight[0] if len(in_flight) == 1 else None
        return heartbeat_worker(
            self.database_url,
            self.worker_id,
            lease_sec=self.lease_sec,
            status=status,
            current_job_id=current,
            clear_current_job=current is None,
            stats=self._stats_snapshot(),
        )

    def _heartbeat_loop(self) -> None:
        """Renew our lease (and job leases) until the stop flag is set."""
        while not self._stop.wait(self.heartbeat_sec):
            try:
                self._heartbeat_once()
            except Exception as exc:  # never let the heartbeat kill the worker
                logger.warning(f"Heartbeat failed for worker {self.worker_id}: {exc}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _claim(self) -> Optional[dict[str, Any]]:
        with self._lock:
            exclude = list(self._in_flight)
        try:
            job = claim_job(
                self.database_url,
                self.worker_id,
                backend=self.backend,
                tasks=self.tasks,
                lease_sec=self.lease_sec,
                exclude_ids=exclude,
            )
        except Exception as exc:
            logger.warning(f"Claim attempt failed: {exc}")
            return None
        if job is not None:
            self.stats["claimed"] += 1
            logger.info(
                f"Claimed job {job['id']} task={job['task']} attempt={job['attempts']}."
            )
        return job

    def _execute(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        task = job["task"]
        payload = job.get("payload") or {}
        started = time.monotonic()
        try:
            result = run_task(self.task_ctx, task, payload)
            # Validate before writing: a non-serializable result must not be
            # silently committed as a broken job.
            json.dumps(result)
            if complete_job(self.database_url, job_id, result, worker_id=self.worker_id):
                self.stats["processed"] += 1
                elapsed = time.monotonic() - started
                print(f"  \u2713 [{task}] job {job_id[:8]} done in {elapsed:.1f}s", flush=True)
            else:
                logger.warning(f"Job {job_id} completed but was no longer owned by this worker.")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.stats["failed"] += 1
            logger.exception(f"Job {job_id} ({task}) failed: {message}")
            try:
                fail_job(self.database_url, job_id, message, worker_id=self.worker_id)
            except Exception as write_exc:
                logger.error(f"Could not record failure for job {job_id}: {write_exc}")
            print(f"  \u2717 [{task}] job {job_id[:8]} failed: {message}", flush=True)

    def request_stop(self) -> None:
        """Ask the run loop to stop after finishing in-flight jobs."""
        self._stop.set()

    def shutdown(self) -> None:
        """Stop the heartbeat and return any unfinished jobs to the queue."""
        self._stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        try:
            released = release_jobs(self.database_url, self.worker_id)
            if released:
                logger.info(f"Released {released} unfinished job(s) back to the queue.")
            set_worker_status(self.database_url, self.worker_id, WorkerStatus.STOPPING)
        except Exception as exc:
            logger.warning(f"Shutdown cleanup failed: {exc}")

    def run(
        self,
        max_jobs: Optional[int] = None,
        idle_exit_sec: Optional[int] = None,
    ) -> dict[str, Any]:
        """Claim and execute jobs until stopped, exhausted, or idle too long."""
        idle_exit = self.idle_exit_sec if idle_exit_sec is None else int(idle_exit_sec)
        started = time.monotonic()

        self.register()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="fe-worker-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

        scope = ", ".join(sorted(self.tasks)) if self.tasks else "all search tasks"
        logger.info(
            f"Worker {self.worker_id} started (backend={self.backend}, tasks={scope}, "
            f"concurrency={self.concurrency}, lease={self.lease_sec}s, device={self.describe_device()})."
        )

        last_activity = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="fe-job") as pool:
                pending: dict[Future, str] = {}
                while not self._stop.is_set():
                    for future in [f for f in pending if f.done()]:
                        job_id = pending.pop(future)
                        try:
                            future.result()
                        except Exception as exc:
                            logger.error(f"Unhandled failure for job {job_id}: {exc}")
                        finally:
                            with self._lock:
                                self._in_flight.discard(job_id)
                        last_activity = time.monotonic()

                    self.stats["elapsed_sec"] = round(time.monotonic() - started, 2)

                    if not pending and max_jobs is not None and self.stats["claimed"] >= max_jobs:
                        logger.info(f"Reached --max-jobs={max_jobs}; stopping.")
                        break
                    if len(pending) >= self.concurrency or (
                        max_jobs is not None and self.stats["claimed"] >= max_jobs
                    ):
                        time.sleep(0.05)
                        continue

                    job = self._claim()
                    if job is None:
                        # Decide we are idle only *after* a poll came up empty, so a
                        # short idle-exit can never mask work that is already queued.
                        if not pending and idle_exit and (time.monotonic() - last_activity) >= idle_exit:
                            logger.info(f"Queue idle for {idle_exit}s; stopping.")
                            break
                        time.sleep(self.poll_interval_sec)
                        continue

                    with self._lock:
                        self._in_flight.add(job["id"])
                    last_activity = time.monotonic()
                    pending[pool.submit(self._execute, job)] = job["id"]
        finally:
            self.shutdown()

        self.stats["elapsed_sec"] = round(time.monotonic() - started, 2)
        return dict(self.stats)


# ----------------------------------------------------------------------
# Factory + CLI
# ----------------------------------------------------------------------


def get_worker(**kwargs: Any) -> JobWorker:
    """Construct a JobWorker from keyword arguments (mirrors get_* conventions)."""
    return JobWorker(**kwargs)


def _parse_task_allowlist(args_tasks: Optional[str], settings: Settings) -> Optional[list[str]]:
    raw = args_tasks if args_tasks is not None else settings.WORKER_TASKS
    if not raw:
        return None
    names = [t.strip() for t in raw.split(",") if t.strip()]
    return names or None


def main() -> int:
    """CLI entrypoint for running the async job worker."""
    parser = argparse.ArgumentParser(description="Footage Retrieval Engine async job worker")
    parser.add_argument(
        "--backend",
        choices=["xclip", "qwen"],
        default=None,
        help="Embedding backend (default: EMBEDDING_BACKEND env or 'xclip')",
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task allowlist (default: all)")
    parser.add_argument("--concurrency", type=int, default=None, help="Jobs processed in parallel (default: 1)")
    parser.add_argument("--max-jobs", type=int, default=None, help="Exit after N jobs (default: unlimited)")
    parser.add_argument(
        "--idle-exit",
        type=int,
        default=None,
        help="Exit after N idle seconds; 0 runs forever (default: WORKER_IDLE_EXIT_SEC)",
    )
    parser.add_argument("--lease-sec", type=int, default=None, help="Job lease duration in seconds")
    parser.add_argument("--heartbeat-sec", type=int, default=None, help="Lease renewal interval")
    parser.add_argument("--poll-interval-sec", type=float, default=None, help="Empty-queue poll interval")
    parser.add_argument("--worker-id", default=None, help="Override the generated worker id")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockEmbedder for instant offline testing (no neural network download)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and queue depth, then exit without claiming anything",
    )
    # Tolerate kernel argv (e.g. Colab's -f flag).
    args, _unknown = parser.parse_known_args()

    settings = get_settings()
    backend = args.backend or settings.EMBEDDING_BACKEND
    tasks = _parse_task_allowlist(args.tasks, settings)

    log_level = logging.WARNING if not args.dry_run else logging.INFO
    logging.basicConfig(level=log_level, stream=sys.stderr)

    print("=" * 80, flush=True)
    print("\U0001f9e0 Footage Engine \u2014 Async Search Job Worker", flush=True)
    print("=" * 80, flush=True)
    print(f"\u2022 Database      : {settings.DATABASE_URL}", flush=True)
    print(f"\u2022 Storage       : {settings.STORAGE_BACKEND}", flush=True)
    print(f"\u2022 Vector Store  : {settings.VECTOR_STORE}", flush=True)
    print(f"\u2022 Backend       : {backend}", flush=True)
    print(f"\u2022 Tasks         : {', '.join(tasks) if tasks else 'all search tasks'}", flush=True)
    print("=" * 80, flush=True)

    if args.dry_run:
        init_db(settings.DATABASE_URL)
        pending = count_jobs(settings.DATABASE_URL, status=JobStatus.PENDING)
        processing = count_jobs(settings.DATABASE_URL, status=JobStatus.PROCESSING)
        live = count_live_workers(settings.DATABASE_URL)
        print(f"\u2022 Pending jobs  : {pending}", flush=True)
        print(f"\u2022 In processing : {processing}", flush=True)
        print(f"\u2022 Live workers  : {live}", flush=True)
        print(f"\u2022 Registered    : {', '.join(available_tasks())}", flush=True)
        print("\n(dry-run: nothing claimed or executed)", flush=True)
        return 0

    worker = JobWorker(
        settings=settings,
        backend=args.backend,
        tasks=tasks,
        concurrency=args.concurrency,
        lease_sec=args.lease_sec,
        heartbeat_sec=args.heartbeat_sec,
        poll_interval_sec=args.poll_interval_sec,
        idle_exit_sec=args.idle_exit,
        worker_id=args.worker_id,
        use_mock=args.mock,
    )
    print(f"\u2022 Worker ID     : {worker.worker_id}", flush=True)
    print(f"\u2022 Device        : {worker.describe_device()}", flush=True)
    print(f"\u2022 Concurrency   : {worker.concurrency}", flush=True)
    print("=" * 80, flush=True)

    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"\n\u23f9\ufe0f  Signal {signum} received \u2014 finishing in-flight jobs...", flush=True)
        worker.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):  # not running in the main thread
            pass

    print("\n\u23f3 Waiting for jobs...", flush=True)
    stats = worker.run(max_jobs=args.max_jobs, idle_exit_sec=args.idle_exit)

    print("\n" + "=" * 80, flush=True)
    print(f"\u2713 Worker stopped after {stats['elapsed_sec']}s", flush=True)
    print(f"  \u2022 Jobs claimed   : {stats['claimed']}", flush=True)
    print(f"  \u2022 Succeeded      : {stats['processed']}", flush=True)
    print(f"  \u2022 Failed         : {stats['failed']}", flush=True)
    print("=" * 80, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
