"""Batch processing pipeline for chunking, embedding, and indexing."""

import logging
import uuid
from typing import Optional
from sqlalchemy import select

from footage_engine.chunking.detector import create_chunks_in_db, preprocess_media
from footage_engine.config import Settings, get_settings
from footage_engine.embeddings import get_embedder
from footage_engine.embeddings.base import EmbeddingBackend
from footage_engine.models.db import get_db_session
from footage_engine.models.media import Chunk, MediaItem, MediaStatus, MediaType
from footage_engine.storage import get_storage_backend
from footage_engine.storage.base import StorageBackend
from footage_engine.vector import get_vector_store
from footage_engine.vector.base import VectorRecord, VectorStore

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes pending/interrupted media items through chunking, embedding, and vector indexing."""

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

    def process_item(self, media_item_id: str) -> bool:
        """Processes a single MediaItem end-to-end. Returns True if succeeded."""
        with get_db_session(self.database_url) as session:
            item = session.get(MediaItem, media_item_id)
            if not item:
                logger.error(f"MediaItem {media_item_id} not found.")
                return False

            if item.status == MediaStatus.DONE:
                logger.info(f"MediaItem {media_item_id} already marked DONE. Skipping.")
                return True

            logger.info(f"Processing MediaItem {item.id} (current status: {item.status.value})...")

            try:
                # Step 1: Chunking
                chunks = item.chunks
                if not chunks:
                    logger.info(f"Generating chunks for {item.id}...")
                    print(f"    → Generating chunks for {item.id}...", flush=True)
                    candidates = preprocess_media(
                        media_item=item,
                        storage=self.storage,
                        chunk_threshold_sec=self.settings.CHUNK_THRESHOLD_SEC,
                        window_sec=self.settings.SLIDING_WINDOW_SEC,
                        overlap_ratio=self.settings.SLIDING_OVERLAP_RATIO,
                    )
                    chunks = create_chunks_in_db(
                        media_item=item,
                        candidates=candidates,
                        session=session,
                        embedding_model=self.embedder.model_name,
                        embedding_version=self.embedder.version,
                    )
                    session.flush()
                    print(f"    ✓ Generated {len(chunks)} chunk(s).", flush=True)

                # Step 2: Embedding
                item.status = MediaStatus.EMBEDDING
                try:
                    local_path = self.storage.get_local_path(item.storage_path)
                except Exception:
                    local_path = self.storage.get_local_path(item.source_url)

                vector_records: list[VectorRecord] = []
                for i, chunk in enumerate(chunks, 1):
                    logger.info(f"Embedding chunk {chunk.id} [{chunk.start_ts}-{chunk.end_ts}]...")
                    s_ts = f"{chunk.start_ts:.1f}s" if chunk.start_ts is not None else "0.0s"
                    e_ts = f"{chunk.end_ts:.1f}s" if chunk.end_ts is not None else "0.0s"
                    print(f"    → Embedding chunk {i}/{len(chunks)} ({s_ts} - {e_ts})...", flush=True)
                    if item.media_type == MediaType.IMAGE:
                        vec = self.embedder.embed_image(local_path)
                    else:
                        vec = self.embedder.embed_video(
                            video_path=local_path,
                            start_ts=chunk.start_ts,
                            end_ts=chunk.end_ts,
                        )

                    vector_id = chunk.id
                    chunk.vector_id = vector_id
                    vec_record = VectorRecord(
                        id=vector_id,
                        vector=vec,
                        chunk_id=chunk.id,
                        media_item_id=item.id,
                        provider=item.provider,
                        media_type=chunk.media_type.value,
                        duration_sec=chunk.end_ts - chunk.start_ts if chunk.end_ts else None,
                        embedding_model=chunk.embedding_model,
                        embedding_version=chunk.embedding_version,
                    )
                    vector_records.append(vec_record)

                # Step 3: Vector Indexing
                if vector_records:
                    logger.info(f"Indexing {len(vector_records)} vectors into collection '{self.collection_name}'...")
                    print(f"    → Indexing {len(vector_records)} vector(s) into vector store...", flush=True)
                    self.vector_store.upsert(self.collection_name, vector_records)
                    for chunk, vrec in zip(chunks, vector_records):
                        chunk.vector_id = vrec.id

                # Step 4: Completion
                item.status = MediaStatus.DONE
                item.error_message = None
                session.flush()
                logger.info(f"Successfully processed MediaItem {item.id} (status: DONE).")
                return True

            except Exception as e:
                logger.exception(f"Failed processing MediaItem {item.id}: {e}")
                item.status = MediaStatus.FAILED
                item.error_message = str(e)
                session.flush()
                return False

    def process_all_pending(self, limit: Optional[int] = None) -> dict[str, int]:
        """Drains pending, chunking, or embedding media items in resumable batches."""
        with get_db_session(self.database_url) as session:
            stmt = (
                select(MediaItem.id)
                .where(
                    MediaItem.status.in_(
                        [
                            MediaStatus.PENDING,
                            MediaStatus.CHUNKING,
                            MediaStatus.EMBEDDING,
                        ]
                    )
                )
                .order_by(MediaItem.ingested_at.asc())
            )
            if limit:
                stmt = stmt.limit(limit)
            item_ids = list(session.execute(stmt).scalars().all())

        logger.info(f"Found {len(item_ids)} media items to process.")
        stats = {"total": len(item_ids), "succeeded": 0, "failed": 0}

        for iid in item_ids:
            ok = self.process_item(iid)
            if ok:
                stats["succeeded"] += 1
            else:
                stats["failed"] += 1

        return stats
