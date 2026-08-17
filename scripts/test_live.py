"""Short direct test for Pixabay and PostgreSQL."""

import sys
from footage_engine.config import get_settings
from footage_engine.models.db import get_engine, init_db
from footage_engine.sources.pixabay import PixabayAdapter

cfg = get_settings()
print("1. Pixabay key present:", bool(cfg.PIXABAY_API_KEY))
print("2. Connecting to DB:", cfg.DATABASE_URL.split("@")[-1])

adapter = PixabayAdapter(api_key=cfg.PIXABAY_API_KEY)
cands = adapter.search("ships", max_results=3)
print(f"3. Pixabay search returned {len(cands)} candidates:")
for c in cands:
    print(f"   • {c.source_id}: {c.metadata.get('tags')} | URL: {c.source_url[:60]}...")

print("4. Initializing DB tables in PostgreSQL...")
init_db(cfg.DATABASE_URL)
print("✓ DB initialized successfully!")
