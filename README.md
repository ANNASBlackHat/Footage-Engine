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
│   ├── config.py          # Environment settings loaded via Pydantic
│   └── orchestrator.py    # Ingestion orchestrator with pre-spend deduplication
├── scripts/               # Narrative workflow and dataset ingestion utilities
│   ├── demo_narrative_search.py  # Multi-segment story search demo
│   ├── find_ships.py             # Provider discovery script
│   ├── ingest_found_assets.py    # Batch asset ingestion script
│   ├── ingest_from_urls.py       # Ingest media from a file of URLs
│   ├── ingest_story_footage.py   # Multi-provider narrative ingestion script
│   └── test_live.py              # Quick provider API connectivity check
├── tests/                 # Automated unit and integration test suite
├── .env.example           # Template for environment configuration
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
