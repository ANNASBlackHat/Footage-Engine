"""Zilliz Cloud and Milvus vector store implementation."""

import logging
import os
from typing import Optional
from footage_engine.vector.base import VectorRecord, VectorSearchResult

# Ensure native DNS resolution for gRPC on macOS
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

logger = logging.getLogger(__name__)

try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    MilvusClient = None  # type: ignore
    DataType = None  # type: ignore


class ZillizVectorStore:
    """Production vector indexing backend for Zilliz Cloud (managed Milvus)."""

    def __init__(self, uri: str, token: str):
        if MilvusClient is None:
            raise ImportError(
                "pymilvus is required for ZillizVectorStore. Install with: pip install pymilvus"
            )
        self.uri = uri
        self.token = token
        self.client = MilvusClient(uri=uri, token=token)

    def init_collection(self, collection_name: str, dimension: int = 512) -> None:
        if self.client.has_collection(collection_name):
            logger.info(f"Zilliz collection '{collection_name}' already exists.")
            return

        logger.info(f"Creating Zilliz collection '{collection_name}' (dimension={dimension}, metric=COSINE)...")
        # In MilvusClient v2.3+, simple create_collection automatically builds vector index with metric_type="COSINE"
        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type="COSINE",
            id_type="string",
            max_length=64,
            auto_id=False,
        )

    def upsert(self, collection_name: str, records: list[VectorRecord]) -> list[str]:
        if not records:
            return []

        self.init_collection(collection_name, dimension=len(records[0].vector))
        data = []
        ids = []
        for r in records:
            data.append(
                {
                    "id": r.id,
                    "vector": r.vector,
                    "chunk_id": r.chunk_id,
                    "media_item_id": r.media_item_id,
                    "provider": r.provider,
                    "media_type": r.media_type,
                    "duration_sec": float(r.duration_sec or 0.0),
                    "embedding_model": r.embedding_model,
                    "embedding_version": r.embedding_version,
                }
            )
            ids.append(r.id)

        self.client.upsert(collection_name=collection_name, data=data)
        return ids

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        res = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            filter=filter_expr or "",
            output_fields=["chunk_id", "media_item_id", "provider", "media_type", "duration_sec"],
            search_params=search_params,
        )

        results: list[VectorSearchResult] = []
        if res and len(res) > 0:
            for hit in res[0]:
                entity = hit.get("entity", {})
                results.append(
                    VectorSearchResult(
                        id=str(hit.get("id")),
                        score=float(hit.get("distance", 0.0)),  # Cosine similarity in Milvus
                        chunk_id=entity.get("chunk_id", ""),
                        media_item_id=entity.get("media_item_id", ""),
                        provider=entity.get("provider", ""),
                        media_type=entity.get("media_type", "video"),
                        duration_sec=float(entity.get("duration_sec", 0.0)),
                    )
                )

        return results

    def delete(self, collection_name: str, ids: list[str]) -> bool:
        if not ids:
            return True
        self.client.delete(collection_name=collection_name, ids=ids)
        return True
