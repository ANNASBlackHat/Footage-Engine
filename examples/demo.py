"""End-to-end demo script for Footage Retrieval Engine."""

import os
import sys
import numpy as np
from PIL import Image

import footage_engine as fe
from footage_engine.embeddings.mock import MockEmbedder
from footage_engine.vector.in_memory import InMemoryVectorStore


def create_sample_video(path: str, duration_sec: int = 10, fps: int = 24) -> str:
    """Helper to generate a lightweight local test video."""
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Color gradient over time
        val = int(255 * (i / total_frames))
        frame = np.full((240, 320, 3), (val, 150, 255 - val), dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


def main():
    use_mock = "--mock" in sys.argv or os.environ.get("USE_MOCK_EMBEDDER") == "1"

    print("=" * 70)
    print("🎬 Footage Retrieval Engine — End-to-End Demo")
    if use_mock:
        print("   (Running with MockEmbedder for instant offline execution)")
    else:
        print("   (Running with X-CLIP Base Multimodal Model)")
    print("=" * 70)

    embedder = MockEmbedder(dimension=512) if use_mock else None

    # 1. Generate local sample videos for demonstration
    sample_dir = "./data/samples"
    os.makedirs(sample_dir, exist_ok=True)
    video1_path = create_sample_video(f"{sample_dir}/ocean_waves.mp4", duration_sec=15)
    video2_path = create_sample_video(f"{sample_dir}/car_driving.mp4", duration_sec=50)

    print("\n[1/5] Ingesting footage...")
    item1 = fe.ingest(
        source_url=f"file://{os.path.abspath(video1_path)}",
        provider="manual",
        source_id="ocean_clip_01",
        metadata={"title": "Ocean Waves Sunset", "category": "nature"},
        duration_sec=15.0,
    )
    print(f"  ✓ Ingested MediaItem: {item1.id} (provider={item1.provider}, duration={item1.duration_sec}s)")

    item2 = fe.ingest(
        source_url=f"file://{os.path.abspath(video2_path)}",
        provider="manual",
        source_id="car_highway_02",
        metadata={"title": "Car driving on highway", "category": "transport"},
        duration_sec=50.0,
    )
    print(f"  ✓ Ingested MediaItem: {item2.id} (provider={item2.provider}, duration={item2.duration_sec}s)")

    # Test Deduplication
    print("\n[2/5] Testing Pre-Spend Deduplication...")
    duplicate_item = fe.ingest(
        source_url=f"file://{os.path.abspath(video1_path)}",
        provider="manual",
        source_id="ocean_clip_01",
    )
    assert duplicate_item.id == item1.id
    print(f"  ✓ Exact duplicate detected! Returned existing ID: {duplicate_item.id} without re-downloading.")

    # Process batch
    print("\n[3/5] Running Resumable Batch Processor (Chunking + Embedding + Indexing)...")
    processor = fe.BatchProcessor(embedder=embedder) if embedder else fe.BatchProcessor()
    stats = processor.process_all_pending()
    print(f"  ✓ Batch processing complete: {stats}")

    # Search
    print("\n[4/5] Executing Semantic Search...")
    query = "calm ocean waters at sunset"
    print(f"  Query: '{query}'")
    retrieval_api = fe.RetrievalAPI(embedder=embedder) if embedder else fe.get_retrieval_api()
    results = retrieval_api.search(query=query, top_k=5)
    for idx, r in enumerate(results, 1):
        print(
            f"  [{idx}] Chunk ID: {r.chunk_id[:8]}... | Score: {r.score:.3f} | "
            f"Range: [{r.start_ts:.1f}s - {r.end_ts:.1f}s] | Provider: {r.provider} | "
            f"URL: {r.storage_url}"
        )

    # Fine Localization
    if results:
        winning_chunk = results[0]
        print(f"\n[5/5] Running Frame-Level Fine Localization on winning chunk {winning_chunk.chunk_id[:8]}...")
        sub_start, sub_end = retrieval_api.fine_localize(winning_chunk.chunk_id, query)
        print(f"  ✓ Refined sub-range: [{sub_start:.2f}s - {sub_end:.2f}s] (ready for timeline cut)")

    print("\n✨ All operations executed successfully!")


if __name__ == "__main__":
    main()
