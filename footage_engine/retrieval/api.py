"""Retrieval API and frame-level fine localization."""

import logging
from datetime import datetime, timezone
from typing import Optional
import numpy as np
from PIL import Image
from sqlalchemy import select

from footage_engine.config import Settings, get_settings
from footage_engine.embeddings import get_embedder
from footage_engine.embeddings.base import EmbeddingBackend
from footage_engine.models.db import get_db_session
from footage_engine.models.media import Chunk, MediaItem, MediaType, utc_now
from footage_engine.retrieval.models import ChunkResult, SearchFilters
from footage_engine.storage import get_storage_backend
from footage_engine.storage.base import StorageBackend
from footage_engine.vector import get_vector_store
from footage_engine.vector.base import VectorStore

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


class RetrievalAPI:
    """Core footage retrieval API supporting semantic search, metadata hydration, and frame localization."""

    def __init__(
        self,
        settings: Settings | None = None,
        storage: StorageBackend | None = None,
        embedder: EmbeddingBackend | None = None,
        vector_store: VectorStore | None = None,
        database_url: str | None = None,
    ):
        self.settings = settings or get_settings()
        self.storage = storage or get_storage_backend(self.settings)
        self.embedder = embedder or get_embedder(self.settings)
        self.vector_store = vector_store or get_vector_store(self.settings)
        self.database_url = database_url or self.settings.DATABASE_URL
        self.collection_name = self.settings.ZILLIZ_COLLECTION_NAME

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> list[ChunkResult]:
        """Performs semantic vector search and hydrates results with full metadata and storage URLs."""
        query_vector = self.embedder.embed_text(query)
        filter_expr = filters.to_milvus_expr() if filters else None

        vector_hits = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_expr=filter_expr,
        )

        if not vector_hits:
            return []

        chunk_ids = [hit.chunk_id for hit in vector_hits]
        score_map = {hit.chunk_id: hit.score for hit in vector_hits}

        results: list[ChunkResult] = []
        with get_db_session(self.database_url) as session:
            stmt = select(Chunk).where(Chunk.id.in_(chunk_ids))
            chunks = list(session.execute(stmt).scalars().all())

            # Map chunks by id
            chunk_dict = {c.id: c for c in chunks}

            # Preserve vector score ranking
            for cid in chunk_ids:
                chunk = chunk_dict.get(cid)
                if not chunk:
                    continue

                # Increment usage stats
                chunk.usage_count += 1
                chunk.last_used_at = utc_now()
                media_item = chunk.media_item

                eff_storage_path = chunk.storage_path or media_item.storage_path
                storage_url = self.storage.get_url(eff_storage_path) if eff_storage_path else ""

                res = ChunkResult(
                    chunk_id=chunk.id,
                    media_item_id=media_item.id,
                    score=score_map.get(chunk.id, 0.0),
                    start_ts=chunk.start_ts,
                    end_ts=chunk.end_ts,
                    duration_sec=chunk.end_ts - chunk.start_ts if chunk.end_ts is not None else None,
                    media_type=chunk.media_type.value,
                    provider=media_item.provider,
                    source_url=media_item.source_url,
                    storage_path=eff_storage_path,
                    storage_url=storage_url,
                    resolution=media_item.resolution,
                    license_type=media_item.license_type,
                    caption=chunk.caption,
                    tags=chunk.tags or [],
                    usage_count=chunk.usage_count,
                    last_used_at=chunk.last_used_at,
                    item_metadata=media_item.item_metadata or {},
                )
                results.append(res)

            session.flush()

        return results

    def get_chunk(self, chunk_id: str) -> Optional[ChunkResult]:
        """Retrieves a single chunk by ID with hydrated metadata."""
        with get_db_session(self.database_url) as session:
            chunk = session.get(Chunk, chunk_id)
            if not chunk:
                return None
            media_item = chunk.media_item
            eff_storage_path = chunk.storage_path or media_item.storage_path
            storage_url = self.storage.get_url(eff_storage_path) if eff_storage_path else ""
            return ChunkResult(
                chunk_id=chunk.id,
                media_item_id=media_item.id,
                score=1.0,
                start_ts=chunk.start_ts,
                end_ts=chunk.end_ts,
                duration_sec=chunk.end_ts - chunk.start_ts if chunk.end_ts is not None else None,
                media_type=chunk.media_type.value,
                provider=media_item.provider,
                source_url=media_item.source_url,
                storage_path=eff_storage_path,
                storage_url=storage_url,
                resolution=media_item.resolution,
                license_type=media_item.license_type,
                caption=chunk.caption,
                tags=chunk.tags or [],
                usage_count=chunk.usage_count,
                last_used_at=chunk.last_used_at,
                item_metadata=media_item.item_metadata or {},
            )

    def get_media_item(self, media_item_id: str) -> Optional[MediaItem]:
        """Retrieves raw MediaItem record by ID."""
        with get_db_session(self.database_url) as session:
            item = session.get(MediaItem, media_item_id)
            if item:
                session.expunge(item)
                return item
        return None

    def fine_localize(
        self,
        chunk_id: str,
        query: str,
        fps: float = 1.0,
        threshold_ratio: float = 0.85,
    ) -> tuple[float, float]:
        """Frame-level pass within a single winning chunk (~1fps extraction + per-frame scoring)
        to find the precise contiguous sub-range matching the query."""
        with get_db_session(self.database_url) as session:
            chunk = session.get(Chunk, chunk_id)
            if not chunk:
                raise ValueError(f"Chunk {chunk_id} not found.")

            media_item = chunk.media_item
            start_ts = chunk.start_ts
            end_ts = chunk.end_ts or start_ts + 1.0
            storage_path = media_item.storage_path
            media_type = media_item.media_type

        # If image or zero duration, return bounds directly
        if media_type in (MediaType.IMAGE, "image") or (end_ts - start_ts) <= 1.0:
            return start_ts, end_ts

        if cv2 is None:
            return start_ts, end_ts

        local_path = self.storage.get_local_path(storage_path)
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            return start_ts, end_ts

        try:
            # Query text embedding vector
            q_vec = np.array(self.embedder.embed_text(query), dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm

            # Sample frames sequentially at 1.0 fps (much faster than seeking)
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            frame_interval = max(1, int(video_fps / max(0.2, fps)))
            
            frame_idx = 0
            scores: list[tuple[float, float]] = []  # (ts, score)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                current_sec = frame_idx / video_fps
                if current_sec > end_ts:
                    break
                if current_sec >= start_ts and (frame_idx % frame_interval == 0):
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    frame_vec = np.array(self.embedder.embed_image(pil_img), dtype=np.float32)
                    f_norm = np.linalg.norm(frame_vec)
                    if f_norm > 0:
                        frame_vec = frame_vec / f_norm
                    score = float(np.dot(q_vec, frame_vec))
                    scores.append((current_sec, score))
                frame_idx += 1

            if not scores:
                return start_ts, end_ts

            # Find peak frame score
            max_score = max(s[1] for s in scores)
            cutoff = max_score * threshold_ratio

            # Identify contiguous window around the peak matching cutoff
            step_sec = frame_interval / video_fps
            qualified_times = [ts for ts, s in scores if s >= cutoff]
            if qualified_times:
                refined_start = max(start_ts, min(qualified_times))
                refined_end = min(end_ts, max(qualified_times) + step_sec)
                return round(refined_start, 2), round(refined_end, 2)

            return start_ts, end_ts
        finally:
            cap.release()


# Module-level convenience functions
_default_api: RetrievalAPI | None = None


def get_retrieval_api() -> RetrievalAPI:
    global _default_api
    if _default_api is None:
        _default_api = RetrievalAPI()
    return _default_api


def search(
    query: str,
    top_k: int = 10,
    filters: SearchFilters | None = None,
) -> list[ChunkResult]:
    return get_retrieval_api().search(query=query, top_k=top_k, filters=filters)


def fine_localize(chunk_id: str, query: str) -> tuple[float, float]:
    return get_retrieval_api().fine_localize(chunk_id=chunk_id, query=query)


def get_chunk(chunk_id: str) -> Optional[ChunkResult]:
    return get_retrieval_api().get_chunk(chunk_id=chunk_id)


def get_media_item(media_item_id: str) -> Optional[MediaItem]:
    return get_retrieval_api().get_media_item(media_item_id=media_item_id)
