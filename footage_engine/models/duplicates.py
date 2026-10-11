"""SQLAlchemy domain models for Duplicate Groups and Duplicate Members."""

import enum
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from footage_engine.models.media import Base, Chunk, utc_now, generate_uuid_str


class DuplicateTier(str, enum.Enum):
    EXACT = "exact"
    NEAR = "near"
    SIMILAR = "similar"


class DuplicateStatus(str, enum.Enum):
    PENDING = "pending"
    MERGED = "merged"
    KEPT = "kept"
    ARCHIVED = "archived"


class DuplicateGroup(Base):
    __tablename__ = "duplicate_groups"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid_str)
    tier: Mapped[DuplicateTier] = mapped_column(Enum(DuplicateTier), default=DuplicateTier.EXACT, nullable=False)
    status: Mapped[DuplicateStatus] = mapped_column(Enum(DuplicateStatus), default=DuplicateStatus.PENDING, nullable=False)
    suggested_keeper_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    members: Mapped[List["DuplicateMember"]] = relationship("DuplicateMember", back_populates="group", cascade="all, delete-orphan")
    suggested_keeper: Mapped[Optional["Chunk"]] = relationship("Chunk", foreign_keys=[suggested_keeper_id])

    __table_args__ = (
        Index("ix_duplicate_groups_tier", "tier"),
        Index("ix_duplicate_groups_status", "status"),
    )


class DuplicateMember(Base):
    __tablename__ = "duplicate_members"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid_str)
    group_id: Mapped[str] = mapped_column(String(36), ForeignKey("duplicate_groups.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(36), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    similarity: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    is_keeper: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    group: Mapped["DuplicateGroup"] = relationship("DuplicateGroup", back_populates="members")
    chunk: Mapped["Chunk"] = relationship("Chunk", foreign_keys=[chunk_id])

    __table_args__ = (
        Index("ix_duplicate_members_group_id", "group_id"),
        Index("ix_duplicate_members_chunk_id", "chunk_id"),
    )
