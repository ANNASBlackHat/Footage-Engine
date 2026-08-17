# SPEC — Footage Retrieval Engine

## 1. Purpose & Scope

A standalone engine that ingests footage (video/image) from multiple sources, chunks it into
retrievable units, embeds those units, indexes them in a managed vector DB, and exposes a
retrieval API. It is a foundational building block for a future VidRush-style production
pipeline, but it is **fully decoupled** from that pipeline — it can be built, tested, and run
entirely on its own, callable from a notebook today and wrapped in a scheduled service later.

**Explicitly out of scope for this engine** (future, separate systems):
- Script/voiceover generation
- Per-script-beat asset resolution / duration-matching / gap-filling
- The browser-based timeline editor
- Auto-publish / export

## 2. Design Principles

1. **Library-first, service-ready.** Every core operation is a plain Python function, callable
   directly in a Colab cell. Nothing assumes a long-running process, a queue, or a webserver —
   but nothing *blocks* wrapping it in one later either.
2. **Idempotent by construction.** Every ingestion call is safe to re-run. State lives in the DB
   (`status` field per item), never in memory, so a future scheduler/job-queue layer can retry or
   resume without any change to core logic.
3. **Dedup before spend.** Duplicate detection happens *before* any download or storage write —
   not just before embedding — so a repeat ingest costs nothing beyond a DB lookup.
4. **Raw is sacred, derived is disposable.** Raw source video is always stored. Chunks,
   embeddings, and captions are all regenerable from raw — so a future embedding-model swap or
   re-chunking strategy is a recompute, never a data-loss event.
5. **Pluggable everywhere it might change.** Ingestion sources, storage backend, and embedding
   model are each behind a thin interface, not hardcoded — because at least one of these
   (the "external URL DB" source) is explicitly undecided today.

## 3. Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │              Ingestion Sources            │
                    │  (pluggable, common output: candidate)     │
                    ├─────────────┬─────────────┬───────────────┤
                    │  Keyword     │  Direct URL  │  External DB  │
                    │  Search      │  List        │  of URLs      │
                    │ (Pixabay/    │  (manual)    │  (interface   │
                    │  Pexels/     │              │  only for now)│
                    │  Coverr)     │              │               │
                    └──────┬───────┴──────┬───────┴───────┬───────┘
                           │              │                │
                           ▼              ▼                ▼
                    ┌───────────────────────────────────────────┐
                    │            Orchestrator (ingest())          │
                    │  1. Dedup check (exact match) → skip if hit │
                    │  2. Download raw file                       │
                    │  3. Upload raw → managed cloud storage       │
                    │  4. Insert media_item row (status=pending)   │
                    └──────────────────────┬────────────────────┘
                                            ▼
                    ┌───────────────────────────────────────────┐
                    │              Preprocessing                   │
                    │  - Already-short clip (< CHUNK_THRESHOLD)?   │
                    │      → treat whole clip as one chunk          │
                    │  - Longer source?                             │
                    │      → PySceneDetect (AdaptiveDetector)        │
                    │      → per-scene chunk; long uncut scenes      │
                    │        further split via fixed sliding window  │
                    └──────────────────────┬────────────────────┘
                                            ▼
                    ┌───────────────────────────────────────────┐
                    │                Embedding                      │
                    │  X-CLIP Base (default, local/Colab-friendly)  │
                    │  + embedding_model/version tagged per chunk    │
                    │  (optional richer captioning pass — off by     │
                    │   default, pluggable, e.g. Qwen2.5-VL-3B)       │
                    └──────────────────────┬────────────────────┘
                                            ▼
                    ┌───────────────────────────────────────────┐
                    │              Indexing (Zilliz Cloud)          │
                    │  vector + metadata per chunk                   │
                    │  media_item.status → 'done'                    │
                    └───────────────────────────────────────────┘

                    ┌───────────────────────────────────────────┐
                    │              Retrieval API                    │
                    │  search(query, filters) → ranked chunks        │
                    │  get_chunk(id) / get_media_item(id)             │
                    │  fine_localize(chunk_id, query) → refined ts    │
                    └───────────────────────────────────────────┘
```

## 4. Data Model

### `media_items` (one row per raw source video/image)

| Field | Type | Notes |
|---|---|---|
| `id` | UUID (PK) | |
| `provider` | string | e.g. `pixabay`, `pexels`, `coverr`, `manual`, `external_db` |
| `source_id` | string, nullable | provider's native ID, if any |
| `source_url` | string | normalized; **unique constraint on (provider, source_id) OR source_url** |
| `license_type` | string | e.g. `pixabay_free`, `pexels_free`, `unknown` |
| `media_type` | enum | `video` \| `image` |
| `storage_path` | string | location in managed cloud storage (raw file) |
| `duration_sec` | float, nullable | null for images |
| `resolution` | string, nullable | e.g. `1920x1080` |
| `status` | enum | `pending` → `downloading` → `chunking` → `embedding` → `done` \| `failed` |
| `error_message` | string, nullable | populated if `status = failed` |
| `ingested_at` | timestamp | |
| `updated_at` | timestamp | |

### `chunks` (one or more rows per media_item)

| Field | Type | Notes |
|---|---|---|
| `id` | UUID (PK) | |
| `media_item_id` | FK → media_items | |
| `start_ts` | float | seconds, relative to source |
| `end_ts` | float | seconds, relative to source |
| `media_type` | enum | `video` \| `image` (inherited, but explicit for query filtering) |
| `embedding_model` | string | e.g. `xclip-base-patch32` — **critical for future migration safety** |
| `embedding_version` | string | free-form version tag for the same model over time |
| `vector_id` | string | pointer/ID into Zilliz collection |
| `caption` | string, nullable | only populated if richer captioning pass was run |
| `tags` | string[], nullable | |
| `usage_count` | int, default 0 | incremented whenever retrieved & used downstream |
| `last_used_at` | timestamp, nullable | |
| `created_at` | timestamp | |

**Dedup enforcement**: a unique DB constraint on `media_items (provider, source_id)` where
`source_id` is not null, and a separate unique constraint on `source_url` — the orchestrator
checks both before doing any network I/O for the raw file.

## 5. Ingestion Sources — Interface

All three source types funnel into the same core function:

```python
def ingest(source_url: str, provider: str, source_id: str | None = None,
           metadata: dict | None = None) -> MediaItem:
    """Idempotent. Returns existing MediaItem immediately if already ingested."""
```

### 5.1 Keyword Search Source
```python
def search_and_ingest(keyword: str, provider: Literal["pixabay", "pexels", "coverr"],
                       max_results: int = 20) -> list[MediaItem]:
    """Calls provider API, dedup-checks each candidate BEFORE download, ingests new ones."""
```
Each provider is a thin adapter (`PixabayAdapter`, `PexelsAdapter`, `CoverrAdapter`) implementing
a common `search(keyword) -> list[Candidate]` method, where `Candidate` is just
`(source_id, source_url, license_type, duration, resolution)`. Adding a new provider later is
"write one adapter," not a pipeline change.

### 5.2 Direct URL List Source
```python
def ingest_url_list(urls: list[str], provider: str = "manual") -> list[MediaItem]:
```
Straightforward batch wrapper over `ingest()`.

### 5.3 External DB of URLs Source — **deferred, interface-only**
Concrete backend is undecided, so this is built as a protocol, not a hardcoded connector:
```python
class URLProvider(Protocol):
    def fetch_urls(self) -> Iterator[tuple[str, dict]]:
        """Yields (url, metadata) pairs. Backend-agnostic — Postgres, CSV, another
        service's API, whatever it ends up being."""
```
```python
def ingest_from_provider(provider: URLProvider, provider_name: str) -> list[MediaItem]:
```
When the actual source is decided (e.g. reusing the Stock Insight AI Postgres DB), only a single
`URLProvider` implementation needs to be written — no change to `ingest_from_provider` or
anything downstream.

## 6. Preprocessing (Chunking)

```python
CHUNK_THRESHOLD_SEC = 45  # tune empirically

def preprocess(media_item: MediaItem) -> list[ChunkCandidate]:
    if media_item.media_type == "image":
        return [ChunkCandidate(0, None)]  # whole image, one "chunk"
    if media_item.duration_sec <= CHUNK_THRESHOLD_SEC:
        return [ChunkCandidate(0, media_item.duration_sec)]  # already-atomic clip
    scenes = detect_scenes(media_item.storage_path)  # PySceneDetect AdaptiveDetector
    chunks = []
    for scene in scenes:
        if scene.duration <= CHUNK_THRESHOLD_SEC:
            chunks.append(scene)
        else:
            chunks.extend(sliding_window_split(scene, window=10, overlap=0.5))
    return chunks
```

## 7. Embedding

- **Default model: X-CLIP Base** (local, open-source, benchmarked best-in-class among
  free/self-hostable video models for true temporal understanding; ~192ms/clip).
- Every chunk's `embedding_model` + `embedding_version` is stored alongside the vector —
  non-negotiable, since embedding spaces are not cross-compatible between models.
- Optional richer captioning pass (off by default) is a separate, pluggable step — e.g.
  Qwen2.5-VL-3B — that only populates `caption`/`tags`, and never replaces the primary embedding.
- Batch embedding runs are the expected Colab usage pattern: pull all `status='pending'` or
  `status='chunking'` media_items, process in a batch, update status to `done` per item as it
  completes (so a crashed/interrupted Colab session can resume from wherever it left off).

## 8. Storage

| What | Where |
|---|---|
| Raw source video/image | Managed object storage (ImageKit — evaluated as better free-tier fit than Cloudinary for this use case) |
| Chunk metadata + relational data | `media_items` / `chunks` tables (Postgres recommended for consistency with Stock Insight AI, but 
SQLite-compatible for local dev) |
| Embedding vectors | Zilliz Cloud (managed, free tier: 5GB storage / 2.5M vCUs per month / up to 5 collections) |

## 9. Retrieval API

```python
def search(query: str, top_k: int = 10,
           filters: SearchFilters | None = None) -> list[ChunkResult]:
    """filters: media_type, license_type, min/max duration, provider, etc."""

def fine_localize(chunk_id: str, query: str) -> tuple[float, float]:
    """Frame-level pass within a single winning chunk (~1fps extraction + per-frame
    scoring) to find the precise contiguous sub-range matching the query."""

def get_chunk(chunk_id: str) -> ChunkResult
def get_media_item(media_item_id: str) -> MediaItem
```
This is the entire surface a future video-generation layer or editor needs — both the
"pre-computed shortlist" generation step and any later live "search again" editor action call
into this same `search()` function; nothing downstream needs its own retrieval logic.

## 10. Deployment Phasing

**Phase 1 (now):** plain Python package (suggested name: `footage_engine`), imported and called
cell-by-cell in Colab. Config (API keys, DB connection, storage credentials) via `.env` /
config file, not hardcoded.

**Phase 2 (later):** the same functions get wrapped by a job queue / scheduler — e.g. a cron
trigger calling `search_and_ingest(keyword, provider)` daily for a watchlist of keywords, or a
lightweight worker process draining a queue of `ingest_url_list` jobs. No change to core
functions required, because of the idempotency + status-field design in §2 and §4 — this is the
entire reason those constraints exist.

## 11. Deferred / Explicitly Open Items

- **External URL-DB source**: interface defined (§5.3), concrete backend not yet chosen.
- **Perceptual/content-hash dedup**: out of scope for v1 (exact-match only, per decision), but
  schema should not block adding a `content_hash` column to `media_items` later without a
  migration headache.
- **Rich captioning tier**: optional module, off by default, no build priority yet.
- **Multi-provider cloud storage**: single provider (ImageKit) for v1; storage calls sit behind
  a thin interface so a second provider is a config addition, not a rewrite.

## 12. Success Criteria for v1

1. Can ingest from all three source types (keyword search, direct URL list, and the
   interface-ready external-DB stub) into one consistent `media_items`/`chunks` schema.
2. Re-running the same ingestion (same URLs/keywords) produces zero duplicate rows and zero
   redundant downloads.
3. `search("person typing on laptop")` returns ranked chunks with correct, precise start/end
   timestamps, cuttable directly.
4. A raw video with multiple internal cuts is correctly split into multiple distinct,
   independently-retrievable chunks.
5. Runs end-to-end inside a free-tier Colab session for a batch of ~20-50 source videos.
