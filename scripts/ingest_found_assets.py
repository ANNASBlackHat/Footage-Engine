"""Parser and Ingestion Runner for found_assets.md (Wikimedia Commons, Public Domain & CC assets)."""

import re
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import requests

import footage_engine as fe
from footage_engine.models.db import init_db, get_db_session
from footage_engine.models.media import MediaItem, Chunk, MediaType
from footage_engine.vector import get_vector_store
from footage_engine.config import get_settings
from footage_engine.storage import get_storage_backend

FOUND_ASSETS_PATH = Path("/Users/annasblackhat/Documents/Experiment/footage-search/found_assets.md")


@dataclass
class FoundAsset:
    item_id: str
    title: str
    license: str
    direct_url: str
    source_page: str
    specs: str
    notes: str
    segment: str
    media_type: str  # "video" or "image"


def parse_found_assets(md_path: Path) -> list[FoundAsset]:
    content = md_path.read_text(encoding="utf-8")
    
    # Regex to extract sections and entries
    current_segment = "General"
    assets: list[FoundAsset] = []
    
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Track segment headers e.g. ## Segment 1: Exterior & Scale
        if line.startswith("## Segment") or line.startswith("## ⚠️ OPTIONAL"):
            current_segment = line.lstrip("# ").strip()
            i += 1
            continue
            
        # Track asset headers e.g. ### 1.1 Container ship seen from space
        header_match = re.match(r"^###\s+([\d\.]+)\s+(.+)$", line)
        if header_match:
            item_num = header_match.group(1)
            title = header_match.group(2)
            
            # Read subsequent bullet points
            license_val = ""
            direct_url = ""
            source_page = ""
            specs = ""
            notes = ""
            
            i += 1
            while i < len(lines) and not lines[i].startswith("###") and not lines[i].startswith("##"):
                b_line = lines[i].strip()
                if b_line.startswith("- **License:**"):
                    license_val = b_line.replace("- **License:**", "").strip()
                elif b_line.startswith("- **Direct file:**"):
                    direct_url = b_line.replace("- **Direct file:**", "").strip()
                elif b_line.startswith("- **Source page:**"):
                    source_page = b_line.replace("- **Source page:**", "").strip()
                elif b_line.startswith("- **Specs:**"):
                    specs = b_line.replace("- **Specs:**", "").strip()
                elif b_line.startswith("- **Great for:**") or b_line.startswith("- **Caution:**"):
                    notes += " " + b_line
                i += 1
                
            if direct_url:
                # Determine media type
                url_clean = direct_url.lower().split("?")[0]
                if url_clean.endswith((".webm", ".ogv", ".mp4", ".mov", ".mkv")):
                    m_type = "video"
                else:
                    m_type = "image"
                    
                assets.append(FoundAsset(
                    item_id=item_num,
                    title=title,
                    license=license_val,
                    direct_url=direct_url,
                    source_page=source_page,
                    specs=specs,
                    notes=notes.strip(),
                    segment=current_segment,
                    media_type=m_type
                ))
            continue
            
        # Also parse loose bullet entries like vintage engine room
        loose_match = re.match(r"^-\s+\*\*(.+?)\*\*\s+—\s+(.+?)\s+—\s+(https://\S+)\s+—\s+source:\s+(https://\S+)", line)
        if loose_match:
            title = loose_match.group(1)
            license_val = loose_match.group(2)
            direct_url = loose_match.group(3)
            source_page = loose_match.group(4)
            url_clean = direct_url.lower().split("?")[0]
            m_type = "video" if url_clean.endswith((".webm", ".ogv", ".mp4", ".mov", ".mkv")) else "image"
            assets.append(FoundAsset(
                item_id=f"misc_{len(assets)+1}",
                title=title,
                license=license_val,
                direct_url=direct_url,
                source_page=source_page,
                specs="",
                notes="",
                segment=current_segment,
                media_type=m_type
            ))
            
        i += 1
        
    return assets


def ingest_wikimedia_asset(asset: FoundAsset, orchestrator: fe.Orchestrator) -> MediaItem | None:
    """Ingest a single Wikimedia asset into PostgreSQL and orchestrate processing."""
    settings = get_settings()
    
    with get_db_session(settings.DATABASE_URL) as session:
        # Check deduplication by source_url
        existing = session.query(MediaItem).filter_by(source_url=asset.direct_url).first()
        if existing:
            print(f"   ℹ️  Already in DB: '{asset.title[:40]}' (ID: {existing.id[:8]})")
            return existing

        # Create new MediaItem
        media_item = MediaItem(
            provider="wikimedia",
            source_id=f"wiki_{asset.item_id}",
            media_type=MediaType.VIDEO if asset.media_type == "video" else MediaType.IMAGE,
            source_url=asset.direct_url,
            storage_path=asset.direct_url,  # Direct streaming URL
            license_type=asset.license or "unknown",
            item_metadata={
                "title": asset.title,
                "segment": asset.segment,
                "source_page": asset.source_page,
                "specs": asset.specs,
                "notes": asset.notes,
                "format": "webm" if asset.direct_url.endswith(".webm") else "ogv" if asset.direct_url.endswith(".ogv") else "jpeg",
            }
        )
        session.add(media_item)
        session.commit()
        session.refresh(media_item)
        print(f"   ✓ Registered: [{asset.media_type.upper()}] '{asset.title[:45]}...' ({asset.item_id})")
        return media_item


def main():
    print("=" * 80)
    print("🚢 Wikimedia & Public Domain Asset Ingestion Engine")
    print(f"📖 Source: {FOUND_ASSETS_PATH}")
    print("=" * 80)
    
    if not FOUND_ASSETS_PATH.exists():
        print(f"Error: {FOUND_ASSETS_PATH} not found.")
        sys.exit(1)
        
    assets = parse_found_assets(FOUND_ASSETS_PATH)
    print(f"✓ Parsed {len(assets)} assets ({sum(1 for a in assets if a.media_type == 'video')} videos, {sum(1 for a in assets if a.media_type == 'image')} images)")
    
    settings = get_settings()
    init_db(settings.DATABASE_URL)
    vec_store = get_vector_store()
    orchestrator = fe.Orchestrator()
    processor = fe.BatchProcessor(vector_store=vec_store)
    
    print("\n" + "-" * 80)
    print("📥 Ingesting Public Domain & CC Assets into Database...")
    print("-" * 80)
    
    # Ingest all assets into PostgreSQL
    for asset in assets:
        ingest_wikimedia_asset(asset, orchestrator)
        
    print("\n" + "-" * 80)
    print("🧠 Processing Chunks & Computing X-CLIP Embeddings into Zilliz Cloud...")
    print("-" * 80)
    
    # Reset any failed media items to pending for retry
    with get_db_session(settings.DATABASE_URL) as session:
        from footage_engine.models.media import MediaStatus
        failed_items = session.query(MediaItem).filter_by(status=MediaStatus.FAILED).all()
        for fi in failed_items:
            fi.status = MediaStatus.PENDING
            fi.error_message = None
        session.commit()

    results = processor.process_all_pending()
    print(f"✓ Embeddings complete: {results}")

    print("\n" + "-" * 80)
    print("🔍 Testing Retrieval on Public Domain & WebM Footage:")
    print("-" * 80)
    
    retrieval = fe.RetrievalAPI(vector_store=vec_store)
    test_queries = [
        "satellite view of Ever Given container ship stuck in canal",
        "container ship departing port harbor wide aerial",
        "crane lifting shipping containers on ship deck",
        "violent ocean storm waves crashing heavy weather",
        "ship navigational bridge ECDIS radar console"
    ]
    
    for q in test_queries:
        print(f"\n🔎 Query: \"{q}\"")
        res = retrieval.search(query=q, top_k=2)
        for rank, r in enumerate(res, 1):
            if r.start_ts is not None and r.end_ts is not None:
                ts = f"[{r.start_ts:.1f}s - {r.end_ts:.1f}s]"
            else:
                ts = "[Still]"
            print(f"   Rank #{rank} (Score: {r.score:.4f}) | {r.provider.upper()} ({r.media_type}) | {ts}")
            print(f"      🔗 {r.storage_url or r.source_url}")


if __name__ == "__main__":
    main()
