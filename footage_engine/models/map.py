"""SQLAlchemy domain models for 2D Embedding Map Points and Clusters."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from footage_engine.models.media import Base, Chunk, utc_now, generate_uuid_str


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid_str)
    model_id: Mapped[str] = mapped_column(String(100), default="xclip-base-patch32", nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    centroid_x: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    centroid_y: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    points: Mapped[List["MapPoint"]] = relationship("MapPoint", back_populates="cluster")

    __table_args__ = (
        Index("ix_clusters_model_id", "model_id"),
    )


class MapPoint(Base):
    __tablename__ = "map_points"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid_str)
    model_id: Mapped[str] = mapped_column(String(100), default="xclip-base-patch32", nullable=False)
    shot_id: Mapped[str] = mapped_column(String(36), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    cluster_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    cluster: Mapped[Optional["Cluster"]] = relationship("Cluster", back_populates="points")
    chunk: Mapped["Chunk"] = relationship("Chunk", foreign_keys=[shot_id])

    __table_args__ = (
        Index("ix_map_points_model_id", "model_id"),
        Index("ix_map_points_shot_id", "shot_id"),
        Index("ix_map_points_cluster_id", "cluster_id"),
    )
