"""SQLAlchemy domain models for MediaItem and Chunk."""

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid_str() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class MediaStatus(str, enum.Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    DONE = "done"
    FAILED = "failed"


class MediaType(str, enum.Enum):
    VIDEO = "video"
    IMAGE = "image"


class MediaItem(Base):
    __tablename__ = "media_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid_str)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    source_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    license_type: Mapped[str] = mapped_column(String(255), default="unknown")
    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), default=MediaType.VIDEO)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolution: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[MediaStatus] = mapped_column(Enum(MediaStatus), default=MediaStatus.PENDING)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    item_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="media_item", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("provider", "source_id", name="uq_media_provider_source_id"),
        UniqueConstraint("source_url", name="uq_media_source_url"),
        Index("ix_media_items_status", "status"),
        Index("ix_media_items_provider", "provider"),
    )

    def __repr__(self) -> str:
        return f"<MediaItem id={self.id} provider={self.provider} status={self.status.value}>"


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid_str)
    media_item_id: Mapped[str] = mapped_column(String(36), ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False)
    start_ts: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    end_ts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), default=MediaType.VIDEO)
    embedding_model: Mapped[str] = mapped_column(String(100), default="xclip-base-patch32")
    embedding_version: Mapped[str] = mapped_column(String(50), default="1.0")
    vector_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    caption: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    media_item: Mapped["MediaItem"] = relationship("MediaItem", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_media_item_id", "media_item_id"),
        Index("ix_chunks_vector_id", "vector_id"),
    )

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} media_item_id={self.media_item_id} [{self.start_ts}-{self.end_ts}]>"
