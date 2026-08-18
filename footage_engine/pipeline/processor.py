"""Batch processing pipeline for chunking, embedding, and indexing."""

import logging
import os
import tempfile
import uuid
from typing import Optional
from sqlalchemy import select

from footage_engine.chunking.detector import create_chunks_in_db, extract_video_clip, preprocess_media
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

                # Optional physical chunk slicing and upload to storage
                if self.settings.UPLOAD_CHUNKS_TO_STORAGE and item.media_type != MediaType.IMAGE:
                    for chunk in chunks:
                        if chunk.end_ts and not chunk.storage_path:
                            try:
                                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_chunk:
                                    tmp_chunk_path = tmp_chunk.name
                                extract_video_clip(
                                    video_path=local_path,
                                    start_ts=chunk.start_ts,
                                    end_ts=chunk.end_ts,
                                    output_path=tmp_chunk_path,
                                )
                                if os.path.exists(tmp_chunk_path) and os.path.getsize(tmp_chunk_path) > 100:
                                    with open(tmp_chunk_path, "rb") as cf:
                                        chunk_bytes = cf.read()
                                    chunk_filename = f"chunks/{item.id[:8]}_{chunk.id[:8]}.mp4"
                                    saved_path = self.storage.save_file(chunk_bytes, chunk_filename)
                                    chunk.storage_path = saved_path
                                    logger.info(f"Uploaded physical chunk {chunk.id} to {saved_path}")
                            except Exception as ce:
                                logger.warning(f"Failed to upload physical chunk {chunk.id}: {ce}")
                            finally:
                                if os.path.exists(tmp_chunk_path):
                                    os.remove(tmp_chunk_path)

                # Compute embeddings (Batched forward passes)
                print(f"    → Embedding {len(chunks)} chunk(s) (batch size: {self.settings.EMBEDDING_BATCH_SIZE})...", flush=True)
                if item.media_type == MediaType.IMAGE:
                    vecs = [self.embedder.embed_image(local_path) for _ in chunks]
                elif hasattr(self.embedder, "embed_video_batch"):
                    ranges = [(c.start_ts, c.end_ts) for c in chunks]
                    vecs = self.embedder.embed_video_batch(
                        video_path=local_path,
                        chunk_ranges=ranges,
                        batch_size=self.settings.EMBEDDING_BATCH_SIZE,
                    )
                else:
                    vecs = [
                        self.embedder.embed_video(
                            video_path=local_path,
                            start_ts=c.start_ts,
                            end_ts=c.end_ts,
                        )
                        for c in chunks
                    ]

                vector_records: list[VectorRecord] = []
                for chunk, vec in zip(chunks, vecs):
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

    def process_all_pending(
        self,
        limit: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> dict[str, int]:
        """Drains pending, chunking, or embedding media items with parallel worker support."""
        import concurrent.futures

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

        workers = max_workers if max_workers is not None else self.settings.PROCESS_NUM_WORKERS
        workers = max(1, min(workers, len(item_ids) or 1))

        logger.info(f"Found {len(item_ids)} media items to process with {workers} worker(s).")
        stats = {"total": len(item_ids), "succeeded": 0, "failed": 0}

        if workers <= 1 or len(item_ids) <= 1:
            # Sequential processing
            for iid in item_ids:
                ok = self.process_item(iid)
                if ok:
                    stats["succeeded"] += 1
                else:
                    stats["failed"] += 1
        else:
            # Parallel multi-worker processing
            print(f"🚀 Processing {len(item_ids)} item(s) in parallel with {workers} worker threads...", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_id = {executor.submit(self.process_item, iid): iid for iid in item_ids}
                for future in concurrent.futures.as_completed(future_to_id):
                    iid = future_to_id[future]
                    try:
                        ok = future.result()
                        if ok:
                            stats["succeeded"] += 1
                        else:
                            stats["failed"] += 1
                    except Exception as exc:
                        logger.error(f"Worker generated an exception for {iid}: {exc}")
                        stats["failed"] += 1

        return stats
