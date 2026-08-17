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

    @property
    def width(self) -> Optional[int]:
        if self.resolution and "x" in self.resolution:
            try:
                return int(self.resolution.split("x")[0])
            except ValueError:
                pass
        return None

    @property
    def height(self) -> Optional[int]:
        if self.resolution and "x" in self.resolution:
            try:
                return int(self.resolution.split("x")[1])
            except ValueError:
                pass
        return None

    @property
    def orientation(self) -> str:
        """Returns 'horizontal' (16:9, landscape), 'vertical' (9:16, portrait/shorts), or 'square' (1:1)."""
        w, h = self.width, self.height
        if not w or not h:
            return "unknown"
        if w > h:
            return "horizontal"
        elif h > w:
            return "vertical"
        return "square"

    @property
    def aspect_ratio(self) -> Optional[str]:
        """Returns a simplified aspect ratio string such as '16:9', '9:16', '1:1', '4:3'."""
        w, h = self.width, self.height
        if not w or not h:
            return None
        import math
        ratio = round(w / h, 2)
        if ratio in (1.77, 1.78):
            return "16:9"
        elif ratio in (0.56, 0.57):
            return "9:16"
        elif ratio == 1.0:
            return "1:1"
        elif ratio in (1.33, 1.34):
            return "4:3"
        gcd = math.gcd(w, h)
        return f"{w // gcd}:{h // gcd}"

    def __repr__(self) -> str:
        return f"<MediaItem id={self.id} provider={self.provider} resolution={self.resolution} status={self.status.value}>"


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
    storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
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
