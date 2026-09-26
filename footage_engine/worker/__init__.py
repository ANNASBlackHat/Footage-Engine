"""Asynchronous job worker for the Footage Retrieval Engine.

Jobs are units of footage-search work submitted by a caller (JS or Python, on any
VM) into the ``jobs`` table and executed by a worker holding the embedding model,
optionally on a Colab or Kaggle GPU.
"""

from footage_engine.worker.base import TaskContext, TaskHandler
from footage_engine.worker.client import (
    submit_fine_localize_clip,
    submit_job,
    submit_search_footage,
    submit_search_script_beat,
)
from footage_engine.worker.queue import (
    DEFAULT_LEASE_SEC,
    claim_job,
    complete_job,
    count_jobs,
    count_live_workers,
    enqueue_job,
    fail_job,
    get_job,
    heartbeat_worker,
    job_to_dict,
    list_workers,
    make_worker_id,
    register_worker,
    release_jobs,
    set_worker_status,
    wait_for_job,
    worker_to_dict,
)
from footage_engine.worker.runner import JobWorker, get_worker
from footage_engine.worker.tasks import SEARCH_TASKS, TASK_HANDLERS, available_tasks, run_task

__all__ = [
    # Runner
    "JobWorker",
    "get_worker",
    "TaskContext",
    "TaskHandler",
    # Task registry
    "TASK_HANDLERS",
    "SEARCH_TASKS",
    "available_tasks",
    "run_task",
    # Producer side
    "submit_job",
    "submit_search_footage",
    "submit_search_script_beat",
    "submit_fine_localize_clip",
    # Queue primitives
    "DEFAULT_LEASE_SEC",
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
    "make_worker_id",
    "job_to_dict",
    "worker_to_dict",
]
