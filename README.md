# Footage Retrieval Engine

A modular, standalone Python engine for ingesting video and image footage across multiple stock providers, associating footage with canonical entities (ships, animals, people, locations), chunking clips via adaptive scene detection, computing multimodal embeddings with X-CLIP/Qwen-VL, and executing entity-filtered semantic search with frame-level fine localization.

---

## Tech Stack

Verified from `pyproject.toml` and `uv.lock`:

- **Language & Runtime:** Python `>=3.10`
- **Configuration & Validation:** Pydantic v2 (`pydantic`, `pydantic-settings`), `python-dotenv`
- **Relational Database:** SQLAlchemy 2.0 (SQLite by default, PostgreSQL supported)
- **Computer Vision & Video Processing:** OpenCV (`opencv-python-headless`), PySceneDetect (`scenedetect[opencv]`), Pillow (`PIL`)
- **Multimodal AI & Embeddings:** PyTorch (`torch`), HuggingFace Transformers (Microsoft X-CLIP `microsoft/xclip-base-patch32`), NumPy
- **Vector Database:** Zilliz Cloud / Milvus (`pymilvus`) with fast In-Memory Vector Store fallback
- **Storage Backends:** Local Filesystem, ImageKit.io (`imagekitio`)
- **Stock & Web Media Providers:** REST API clients for Pexels, Pixabay, Coverr, YouTube (via `yt-dlp`), and direct URLs
- **Testing:** `pytest`, `pytest-mock`
- **Package & Dependency Management:** `uv` / `setuptools`

---

## Project Structure

```
footage-engine/
├── data/                  # Local storage directory for media assets and samples
├── examples/              # End-to-end demo scripts and Google Colab notebook
│   ├── demo.py            # Local end-to-end pipeline demo with synthetic clips
│   ├── demo_colab.ipynb   # Interactive Google Colab notebook walkthrough
│   └── run_real.py        # Live provider ingestion and retrieval runner
├── footage_engine/        # Core library package
│   ├── chunking/          # Scene detection and sliding-window video chunking
│   ├── embeddings/        # Multimodal embedding extractors (X-CLIP, Mock)
│   ├── models/            # SQLAlchemy database entities and Pydantic schemas
│   ├── pipeline/          # Resumable batch processor for pending media items
│   ├── retrieval/         # Vector search, hybrid filtering, and fine localization
│   ├── sources/           # Stock media providers (Pexels, Pixabay, Coverr, Direct)
│   ├── storage/           # Storage backends (Local filesystem, ImageKit)
│   ├── vector/            # Vector store clients (Zilliz Cloud, In-Memory)
│   ├── worker/            # Async job queue + worker (queue-driven footage search)
│   ├── config.py          # Environment settings loaded via Pydantic
│   └── orchestrator.py    # Ingestion orchestrator with pre-spend deduplication
├── scripts/               # Narrative workflow and dataset ingestion utilities
│   ├── demo_narrative_search.py  # Multi-segment story search demo
│   ├── find_ships.py             # Provider discovery script
│   ├── ingest_found_assets.py    # Batch asset ingestion script
│   ├── ingest_from_urls.py       # Ingest media from a file of URLs
│   ├── ingest_story_footage.py   # Multi-provider narrative ingestion script
│   ├── submit_job.py             # Enqueue a footage-search job and optionally wait
│   ├── run_worker.py             # Run the async job worker locally
│   ├── kaggle_worker.py          # Run the job worker on a Kaggle GPU
│   ├── colab_worker.py           # Run the job worker on a Colab GPU VM
│   └── test_live.py              # Quick provider API connectivity check
├── tests/                 # Automated unit and integration test suite
├── .env.example           # Template for environment configuration
├── WORKER_API_DOC.md     # Integration guide for services consuming the job queue
├── pyproject.toml         # Project dependencies, build metadata, and pytest settings
└── SPEC.md                # System specification and architecture design document
```

---

## Prerequisites

- **Python:** `>= 3.10`
- **Package Manager:** `uv` (recommended) or `pip` (with `venv`)
- **Hardware Acceleration (Optional):** NVIDIA GPU (`cuda`), Apple Silicon (`mps`), or CPU for embedding generation (`EMBEDDING_DEVICE=auto`)

---

## Setup / Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd footage-engine
```

### 2. Install dependencies

Using **`uv`** (recommended):
```bash
# Install core and all optional dependencies (dev and video)
uv sync --all-extras
```

Using standard **`pip`**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,video]"
```

### 3. Configure environment variables
```bash
cp .env.example .env
```

---

## Environment Variables

Configure your `.env` file according to the options defined in `.env.example` / `footage_engine/config.py`:

| Variable | Type / Default | Description |
|---|---|---|
| **Stock & Web Providers** | | |
| `PEXELS_API_KEY` | string | API key for [Pexels API](https://www.pexels.com/api/) |
| `PIXABAY_API_KEY` | string | API key for [Pixabay API](https://pixabay.com/api/docs/) |
| `COVERR_API_KEY` | string | API key for [Coverr API](https://coverr.co/) |
| `YOUTUBE_COOKIES` | string | Optional: Local file path (`/path/to/cookies.txt`), URL, or raw Netscape/base64 string |
| `YOUTUBE_COOKIES_FROM_BROWSER` | string | Optional: Browser to extract session cookies from (`chrome`, `firefox`, `brave`, `safari`, `edge`) |
| **Relational Database** | | |
| `DATABASE_URL` | `sqlite:///./footage_engine.db` | Database connection string (SQLite or PostgreSQL) |
| **Storage Backend** | | |
| `UPLOAD_RAW_TO_STORAGE` | `false` | If `true`, uploads raw media to storage; if `false`, streams from source |
| `STORAGE_BACKEND` | `local` / `imagekit` | Storage provider backend (`local` or `imagekit`) |
| `LOCAL_STORAGE_DIR` | `./data/storage` | Directory path when using local file storage |
| `IMAGEKIT_PUBLIC_KEY` | string | ImageKit public key |
| `IMAGEKIT_PRIVATE_KEY` | string | ImageKit private key |
| `IMAGEKIT_URL_ENDPOINT` | string | ImageKit URL endpoint |
| **Vector Store** | | |
| `VECTOR_STORE` | `in_memory` / `zilliz` | Vector store backend (defaults to fast in-memory store if empty) |
| `ZILLIZ_URI` | string | Zilliz Cloud / Milvus cluster URI |
| `ZILLIZ_TOKEN` | string | Zilliz Cloud API token |
| `ZILLIZ_COLLECTION_NAME`| string | Name of Milvus collection for chunk embeddings |
| `QWEN_ZILLIZ_URI` | string | Dedicated Qwen cluster URI (falls back to `ZILLIZ_URI`) |
| `QWEN_ZILLIZ_TOKEN` | string | Dedicated Qwen cluster token (falls back to `ZILLIZ_TOKEN`) |
| `QWEN_ZILLIZ_COLLECTION_NAME` | string | Qwen collection (falls back to `ZILLIZ_COLLECTION_NAME`) |
| **Embedding Model** | | |
| `EMBEDDING_BACKEND` | `xclip` / `qwen` | Active embedding backend (default `xclip`) |
| `DEFAULT_EMBEDDING_MODEL` | `microsoft/xclip-base-patch32` | HuggingFace multimodal model name |
| `DEFAULT_EMBEDDING_VERSION` | `1.0` | Model version tag stored with indexed vectors |
| `EMBEDDING_DIMENSION` | `512` | Vector embedding dimension size |
| `EMBEDDING_DEVICE` | `auto` | Device for inference: `auto`, `cuda:0`, `mps`, or `cpu` |
| **Chunking Parameters** | | |
| `CHUNK_THRESHOLD_SEC` | `45.0` | Clips shorter than this threshold become a single chunk |
| `SLIDING_WINDOW_SEC` | `10.0` | Window length (seconds) for sub-chunking long scenes |
| `SLIDING_OVERLAP_RATIO`| `0.5` | Overlap ratio (0.0 to 1.0) between sliding windows |
| **LLM & Multi-Query Expansion** | | |
| `GEMINI_API_KEY` / `LLM_API_KEY` | string | API key for Gemini or any OpenAI-compatible provider (Groq, OpenAI, Ollama, etc.) |
| `LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | Base URL for OpenAI-compatible endpoint |
| `LLM_MODEL` | `gemini-2.5-flash` | Model used for script beat decomposition and LLM Judge reranking |

---

## Running the Project

### 1. Run the End-to-End Demo

The demo generates synthetic videos, tests pre-spend deduplication, executes scene chunking and batch processing, runs semantic search, and performs frame-level fine localization.

```bash
# Fast offline execution using MockEmbedder (no model download required)
uv run python examples/demo.py --mock

# Full execution using Microsoft X-CLIP multimodal model
uv run python examples/demo.py
```

### 2. Multi-Provider Ingestion & Entity Scripts

```bash
# Ingest media from a file of URLs tagged with a canonical entity
uv run python scripts/ingest_from_urls.py urls.txt --entity "USS Cyclops" --entity-type ship

# Ingest media from URLs into general B-roll pool (unassigned)
uv run python scripts/ingest_from_urls.py urls.txt

# Ingest a YouTube documentary and associate with an entity
uv run python scripts/ingest_youtube_video.py "https://www.youtube.com/watch?v=..." --entity "Aye-aye" --entity-type animal

# Ingest keywords from a text file tagged with an entity
uv run python scripts/ingest_from_keywords.py --file keywords.txt --entity "Bermuda Triangle" --entity-type location

# Ingest curated narrative assets from Pexels, Pixabay, and Coverr
uv run python scripts/ingest_story_footage.py

# Preview extracted URLs without ingesting
uv run python scripts/ingest_from_urls.py urls.txt --dry-run

# Limit to first N URLs and label the provider
uv run python scripts/ingest_from_urls.py urls.txt --max 20 --provider wikimedia

# Skip embedding processing (ingest only)
SKIP_PROCESSING=1 uv run python scripts/ingest_from_urls.py urls.txt

# Query footage via CLI with entity scoping
uv run python scripts/search_cli.py "ship sinking in heavy storm" --entity "USS Cyclops"

# Multi-Query search for narrative voiceover beats (auto-expands into concrete visual queries + entity detection)
uv run python scripts/search_cli.py "As dusk settled, the great warship slipped quietly past the harbor fortresses" --beat

# Multi-Query search with optional LLM Judge reranker and confidence threshold
uv run python scripts/search_cli.py "A violent midnight storm battered the cargo hull" --beat --rerank --confidence-floor 0.65

# Query general pool via CLI (unrestricted)
uv run python scripts/search_cli.py "calm ocean sunset"

# Query narrative story segments with ranked retrieval and fine localization
uv run python scripts/demo_narrative_search.py
```

### 3. Programmatic Usage

```python
import footage_engine as fe
from footage_engine.retrieval.models import SearchFilters

# 1. Ingest media tagged with a canonical entity (auto-deduplicated)
item = fe.ingest(
    source_url="https://example.com/cyclops_storm.mp4",
    provider="pexels",
    source_id="123456",
    media_type="video",
    entity_name="USS Cyclops",
    entity_type="ship",
)

# 2. Process pending items (scene detection, chunking, X-CLIP embedding, vector indexing)
processor = fe.BatchProcessor()
processor.process_all_pending()

# 3. Perform entity-filtered semantic search (standard fast path ~20-50ms)
retrieval = fe.get_retrieval_api()
results = retrieval.search(
    query="cargo vessel navigating open ocean storm",
    top_k=5,
    filters=SearchFilters(entity_name="USS Cyclops"),
)

for res in results:
    print(f"Match: {res.chunk_id} | Entity: {res.entity_name} | Score: {res.score:.4f} | [{res.start_ts}s - {res.end_ts}s]")

# 4. Multi-Query expansion & optional LLM Judge for complex voiceover beats
beat_results = retrieval.search_beat(
    beat_text="On March 4th, the USS Cyclops vanished into the calm waters of the Atlantic.",
    top_k=5,
    rerank=True,             # Optional LLM Judge reorders candidates against original beat prose
    confidence_floor=0.6,    # Optional score filter
)

# 5. Search general pool (untagged + tagged footage)
general_results = retrieval.search(query="calm ocean waves", top_k=5)

# 6. Fine localize the exact cut within a winning chunk
if results:
    start_cut, end_cut = retrieval.fine_localize(
        chunk_id=results[0].chunk_id,
        query="cargo vessel navigating open ocean storm",
        fps=1.0,
    )
    print(f"Refined cut timestamps: {start_cut:.2f}s - {end_cut:.2f}s")
```

---

## Model Context Protocol (MCP) Server

Footage Retrieval Engine exposes its core capabilities through a built-in **Model Context Protocol (MCP)** server, enabling AI coding assistants and video production agents (Claude Desktop, Cursor, Antigravity, cline) to discover, search, localize, and ingest footage directly.

### 1. Installation

Install dependencies with the `mcp` extra:

```bash
uv pip install -e ".[dev,video,mcp]"
# or using pip
pip install -e ".[dev,video,mcp]"
```

### 2. Starting the Server

```bash
# Run over stdio (default, recommended for Claude Desktop / Cursor)
uv run footage-engine-mcp

# Run with MockEmbedder for instant offline testing (no neural network download)
uv run footage-engine-mcp --mock

# Run as an SSE HTTP service for remote agents
uv run footage-engine-mcp --transport sse --host 0.0.0.0 --port 8000
```

### 3. Client Configuration

#### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "footage-engine": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/footage-engine",
        "run",
        "footage-engine-mcp"
      ]
    }
  }
}
```

#### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "footage-engine": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/footage-engine",
        "run",
        "footage-engine-mcp"
      ]
    }
  }
}
```

### 4. Exposed MCP Tools, Resources & Prompts

| Type | Name | Description |
|---|---|---|
| **Tool** | `search_footage` | Natural language semantic search with filters for canonical entity (`entity_name` or `entity_id`), `media_type` ('video'/'image'), `orientation` ('landscape'/'horizontal' or 'vertical'/'portrait'), `min_duration`, `max_duration`, and `provider`. Returns ranked chunks with cut timestamps, entity associations, aspect ratios, and storage URLs. |
| **Tool** | `search_script_beat` | Multi-Query expansion search for complex voiceover/script beats. Decomposes abstract narration prose into concrete visual queries, merges candidates by vector similarity, and optionally applies an LLM Judge reranker (`rerank=True`). |
| **Tool** | `fine_localize_clip` | 1fps frame-level scoring inside a winning chunk to refine exact start/end cut points. |
| **Tool** | `get_clip_details` | Full metadata, resolution, parent media info, and storage URL for a chunk. |
| **Tool** | `get_media_item_details` | Full details for a raw media item and all its partitioned chunk segments. |
| **Tool** | `ingest_url` | Direct URL / YouTube ingestion with pre-spend deduplication and optional canonical `entity_name` / `entity_id`. |
| **Tool** | `ingest_keywords` | Search stock providers (Pexels, Pixabay, Coverr) and ingest candidates. |
| **Tool** | `list_entities` | List registered canonical entities with their IDs, aliases, and entity types. |
| **Tool** | `resolve_or_create_entity` | Resolve an entity by name/alias or register a new canonical entity. |
| **Tool** | `process_pending_queue` | Batch process pending items through chunking and vector indexing. |
| **Tool** | `get_library_stats` | Global stats on indexed assets, providers, and vector store backend. |
| **Resource** | `footage://chunks/{chunk_id}` | JSON payload of chunk metadata and stream URL. |
| **Resource** | `footage://stats` | Live library statistics. |
| **Prompt** | `broll-match-beat` | Directorial prompt translating narration text into optimal visual search queries. |

---

## Async Job Worker (footage search as a queue)

The MCP server is synchronous: a caller holds a session open and waits. The job
worker is its asynchronous counterpart - a caller submits a **footage-search
request** as a row in the `jobs` table and collects the result later. Task names
and payload keys are identical to the MCP tools, and both sides build their
responses with the same serializers (`footage_engine/retrieval/serialize.py`),
so a queued job returns exactly what the equivalent MCP tool returns.

This is what lets a JS frontend (or any other language) on a different VM ask for
B-roll while a Colab/Kaggle GPU does the embedding work.

### Why a queue instead of just the MCP

* `search_footage` and `search_script_beat` embed the query with the model.
* `fine_localize_clip` embeds the query once and then **every sampled frame** of
  the chunk - the heaviest GPU operation in the engine.
* A queued job survives its worker dying: if an ephemeral Colab/Kaggle VM is
  killed mid-search, the job is still there and becomes claimable again once its
  lease expires.

### 1. Submit jobs

```bash
# Queue a semantic search
uv run python scripts/submit_job.py --query "container ship aerial" --wait

# Queue a script beat (Multi-Query expansion + optional LLM judge)
uv run python scripts/submit_job.py --task search_script_beat \
  --beat "On March 4th, the USS Cyclops vanished." --rerank --wait

# Refine the exact cut inside a winning chunk
uv run python scripts/submit_job.py --task fine_localize_clip \
  --chunk-id <chunk-id> --query "cargo vessel in storm" --wait

# Inspect queue depth and live workers
uv run python scripts/submit_job.py --stats
```

From Python:

```python
from footage_engine.worker import submit_search_footage, wait_for_job

job_id = submit_search_footage("harbour at dawn", top_k=5, backend="qwen")
job = wait_for_job(job_id=job_id, timeout_sec=300)
print(job["result"]["results"])
```

From any other language (JS, Go, ...), enqueue with plain SQL - nothing but the
database is shared:

```sql
-- status, attempts and created_at have server defaults: only these three are required
INSERT INTO jobs (id, task, payload)
VALUES ('<uuid>', 'search_footage', '{"query": "harbour at dawn", "top_k": 5}');

-- optional: pin to a worker backend and make the submit idempotent
INSERT INTO jobs (id, task, payload, backend, idempotency_key)
VALUES ('<uuid>', 'search_footage', '{"query": "harbour at dawn"}', 'qwen', 'beat-42');

-- poll
SELECT status, result, error FROM jobs WHERE id = '<uuid>';
```

Stored `status` values are the enum **names** - `PENDING`, `PROCESSING`, `DONE`,
`FAILED` - even though the Python API and CLI report them lowercase. On Postgres
`status` is a native `jobstatus` enum, so an explicit insert needs a cast
(`'PENDING'::jobstatus`). `idempotency_key` is unique, so a retried submit cannot
enqueue the same request twice. `backend` pins a job to the `qwen` or `xclip`
worker (they own separate vector collections).

Full integration reference for another service: **[WORKER_API_DOC.md](WORKER_API_DOC.md)**.

### 2. Run a worker

```bash
# Local worker (X-CLIP on CPU/MPS)
uv run python scripts/run_worker.py --backend xclip --idle-exit 60

# Offline check with no model download
uv run python scripts/run_worker.py --mock --dry-run

# GPU worker on Kaggle / Colab (Qwen)
python scripts/kaggle_worker.py --backend qwen --idle-exit 900
python scripts/colab_worker.py  --backend qwen --idle-exit 900
```

| Flag | Meaning |
|---|---|
| `--backend {xclip,qwen}` | Which embedding model/collection this worker serves |
| `--tasks A,B` | Allowlist, e.g. `search_footage,fine_localize_clip` |
| `--concurrency N` | Jobs processed in parallel (default: 1) |
| `--max-jobs N` | Exit after N jobs |
| `--idle-exit SEC` | Exit after SEC idle seconds; `0` = run forever |
| `--mock` | MockEmbedder - no neural network download |
| `--dry-run` | Print resolved config and queue depth, then exit |

### 3. How exclusivity works

Claiming is a compare-and-swap `UPDATE` on the job row:

```sql
UPDATE jobs
   SET status='processing', picked_by=:worker,
       attempts=attempts+1, lease_expires_at=:exp
 WHERE id = :candidate AND status='pending';  -- or an expired lease
```

`SELECT ... FOR UPDATE SKIP LOCKED` is deliberately **not** used: SQLite silently
ignores it, so such a locking mistake would only ever surface in production. Every
claim carries a lease which a heartbeat thread renews, so a worker that dies (or a
Colab VM that gets killed) has its jobs reclaimed automatically. When nothing is
double-executed, `jobs.attempts` stays at `1` for every job - which is exactly
what `tests/test_worker.py` asserts under contention.

### 4. Running on a Colab / Kaggle GPU

The worker is meant to live where the GPU is. Anything that can reach the shared
Postgres can be a caller, so the GPU box never needs an inbound port.

**Prerequisites**

| Requirement | Why |
|---|---|
| Hosted Postgres (`DATABASE_URL=postgresql://...`) | Colab and Kaggle cannot reach `localhost`; the laptop and the VM must share one queue |
| `VECTOR_STORE=zilliz` | **Required for cross-VM search.** With the default `in_memory` each process holds a private index and a remote worker finds nothing |
| GPU accelerator enabled | For `--backend qwen`; `EMBEDDING_DEVICE=auto` selects `cuda` when present |
| `STORAGE_BACKEND=imagekit` (or a remote URL path) | `fine_localize_clip` must download the video. If the VM cannot resolve the file it returns the original cut bounds unchanged - a silent no-op |
| `psycopg2-binary` | The Postgres driver is **not** declared in `pyproject.toml`; install it explicitly |
| `colab` CLI: `pip install google-colab-cli` | Option A only |

#### Option A - drive a Colab GPU VM from your laptop

`scripts/colab_worker.py` mirrors `colab_backfill.py`: it bundles the repo code
(never `.env`), serialises your local `Settings` into an env file, installs deps on
the VM, runs the worker, then stops the VM.

```bash
pip install google-colab-cli          # once
uv run python scripts/colab_worker.py --backend qwen --idle-exit 900
```

| Flag | Use |
|---|---|
| `--bundle-only` | Build and inspect the bundle without touching Colab |
| `--dry-run` | Config + queue check on the VM, without loading the model |
| `--keep` | Leave the VM running afterwards |
| `--skip-setup` | Reuse an existing session and skip pip installs |
| `--exec-timeout 7200` | Extend the `colab exec` window |

`--idle-exit` is what stops the VM burning quota: `900` means "quit after 15
minutes with nothing to do". `colab exec` is bounded by `--exec-timeout`, so for a
worker that runs indefinitely use Option B instead.

#### Option B - run it in a notebook cell (long-lived worker)

Set the secrets **before** importing `footage_engine`, because `get_settings()` is
cached:

```python
import os
os.environ["DATABASE_URL"]          = "postgresql://user:pass@host:5432/footage_engine"
os.environ["VECTOR_STORE"]          = "zilliz"
os.environ["ZILLIZ_URI"]            = "..."
os.environ["ZILLIZ_TOKEN"]          = "..."
os.environ["QWEN_ZILLIZ_URI"]       = "..."
os.environ["QWEN_ZILLIZ_TOKEN"]     = "..."
os.environ["STORAGE_BACKEND"]       = "imagekit"
os.environ["IMAGEKIT_PUBLIC_KEY"]   = "..."
os.environ["IMAGEKIT_PRIVATE_KEY"]  = "..."
os.environ["IMAGEKIT_URL_ENDPOINT"] = "..."
os.environ["GEMINI_API_KEY"]        = "..."  # optional: Multi-Query + LLM judge
```

```python
!git clone --depth 1 https://github.com/ANNASBlackHat/Footage-Engine.git
%cd Footage-Engine
!pip -q install -e ".[qwen,video]" psycopg2-binary
!python scripts/run_worker.py --backend qwen --dry-run      # verify cheaply first
!python scripts/run_worker.py --backend qwen --idle-exit 0  # then run
```

The first real run downloads torch and `Qwen/Qwen3-VL-Embedding-2B` (a few GB).
`Ctrl-C` finishes in-flight jobs, returns unfinished ones to `pending`, and marks
the worker `stopping`.

#### Option C - Kaggle

```python
!git clone https://github.com/ANNASBlackHat/Footage-Engine.git
%cd Footage-Engine
!python scripts/kaggle_worker.py --backend qwen --idle-exit 900
```

Enable **Internet** and a **GPU accelerator** in the right-hand panel, then attach
secrets under *Add-ons -> Secrets* (Kaggle exposes them as environment variables).
`kaggle_worker.py` checks for `DATABASE_URL`, `VECTOR_STORE` and the `ZILLIZ_*` /
`QWEN_ZILLIZ_*` names before starting, and installs only what Kaggle does not
already ship.

#### Confirming the worker is wired up

```bash
uv run python scripts/submit_job.py --stats
```

A healthy GPU worker appears in the output:

```json
{
  "live_workers": 1,
  "workers": [
    { "id": "colab-vm:1234:qwen:9f2c1a4e", "hostname": "...", "backend": "qwen",
      "device": "cuda (Tesla T4)", "status": "idle", "concurrency": 1 }
  ]
}
```

A worker counts as live while its heartbeat is newer than `default_stale_sec()`,
which is derived from `WORKER_HEARTBEAT_SEC` so it always exceeds the heartbeat
interval - a killed VM therefore drops off the list on its own. Jobs it was
holding become claimable again after `WORKER_LEASE_SEC`; a job showing
`attempts > 1` was retried after its worker died.

Pin `--tasks` on a GPU worker so DB-only jobs do not consume GPU time:

```bash
uv run python scripts/colab_worker.py --backend qwen --tasks search_footage,search_script_beat,fine_localize_clip
```

> **Note:** cross-VM search requires `VECTOR_STORE=zilliz`. With the default
> `in_memory` store each process holds a private index and a remote worker will
> find nothing.

---

## Running Tests

Run the test suite using pytest:

```bash
# Run all unit tests with uv
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test module
uv run pytest tests/test_chunking.py
```

---

## Deployment

Footage Retrieval Engine is architected **library-first** and **service-ready**:
- **Library / Interactive Execution:** Designed to be imported directly into Python workflows or run inside Google Colab environments (`examples/demo_colab.ipynb`).
- **Batch / Service Mode:** State is fully persisted in the relational database (`MediaStatus` lifecycle: `pending` -> `downloaded` -> `chunked` -> `embedded` -> `done`), making batch processing tasks safe to trigger via scheduled cron jobs or worker queues.

<!-- TODO: Configure CI/CD pipeline (.github/workflows) for automated testing and package publishing -->
