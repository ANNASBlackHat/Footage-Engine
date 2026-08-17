"""In-memory cosine similarity vector store for local testing."""

import math
from typing import Optional
from footage_engine.vector.base import VectorRecord, VectorSearchResult


class InMemoryVectorStore:
    """Zero-dependency in-memory vector store implementing exact cosine similarity."""

    def __init__(self):
        # collection_name -> dict of id -> VectorRecord
        self.collections: dict[str, dict[str, VectorRecord]] = {}

    def init_collection(self, collection_name: str, dimension: int = 512) -> None:
        if collection_name not in self.collections:
            self.collections[collection_name] = {}

    def upsert(self, collection_name: str, records: list[VectorRecord]) -> list[str]:
        self.init_collection(collection_name)
        ids = []
        for r in records:
            self.collections[collection_name][r.id] = r
            ids.append(r.id)
        return ids

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        # Assuming vectors are already unit-normalized, dot product == cosine similarity
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        return float(dot)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        self.init_collection(collection_name)
        col = self.collections[collection_name]

        scored: list[VectorSearchResult] = []
        for vid, record in col.items():
            # Basic filtering support for common tags/fields
            if filter_expr:
                if 'provider == "' in filter_expr:
                    wanted_prov = filter_expr.split('provider == "')[1].split('"')[0]
                    if record.provider != wanted_prov:
                        continue
                if 'media_type == "' in filter_expr:
                    wanted_type = filter_expr.split('media_type == "')[1].split('"')[0]
                    if record.media_type != wanted_type:
                        continue

            sim = self._cosine_similarity(query_vector, record.vector)
            scored.append(
                VectorSearchResult(
                    id=record.id,
                    score=sim,
                    chunk_id=record.chunk_id,
                    media_item_id=record.media_item_id,
                    provider=record.provider,
                    media_type=record.media_type,
                    duration_sec=record.duration_sec,
                    metadata=record.metadata,
                )
            )

        # Sort descending by similarity score
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def delete(self, collection_name: str, ids: list[str]) -> bool:
        if collection_name not in self.collections:
            return False
        for vid in ids:
            self.collections[collection_name].pop(vid, None)
        return True
