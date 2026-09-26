"""SQLAlchemy models for the asynchronous job queue and worker registry.

A ``Job`` is a unit of work submitted by a caller (JS or Python, possibly on a
different VM) and executed by a worker process. Payloads and results are plain
JSON, so any language can enqueue work by inserting a single row.
"""

import enum
from datetime import datetime
from typing import Any, Optional
from sqlalchemy import (
    DateTime,
    Enum,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from footage_engine.models.media import Base, generate_uuid_str, utc_now


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class WorkerStatus(str, enum.Enum):
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"


class Job(Base):
    """A queued unit of work with lease-based ownership."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid_str)
    task: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.PENDING, server_default=text("'PENDING'"), nullable=False
    )
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(
        Integer, default=0, server_default=text("0"), nullable=False
    )
    # Which embedding backend may serve this job ('qwen'/'xclip'); NULL = any.
    backend: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # Optional caller-supplied key so a retried submit cannot enqueue twice.
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    picked_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=func.now(), nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_jobs_idempotency_key"),
        Index("ix_jobs_task_status", "task", "status"),
        Index("ix_jobs_status_created_at", "status", "created_at"),
        Index("ix_jobs_picked_by", "picked_by"),
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id} task={self.task} status={self.status.value}>"


class Worker(Base):
    """A registered worker process, used for liveness checks before enqueueing."""

    __tablename__ = "workers"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    hostname: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    backend: Mapped[str] = mapped_column(String(16), default="qwen", nullable=False)
    device: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    concurrency: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[WorkerStatus] = mapped_column(Enum(WorkerStatus), default=WorkerStatus.IDLE)
    current_job_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    stats: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_workers_last_heartbeat", "last_heartbeat"),
        Index("ix_workers_backend", "backend"),
    )

    def __repr__(self) -> str:
        return f"<Worker id={self.id} backend={self.backend} status={self.status.value}>"
