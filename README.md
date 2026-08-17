# Footage Retrieval Engine

A modular, standalone Python engine for ingesting video and image footage across multiple stock providers, chunking clips via adaptive scene detection, computing multimodal embeddings with X-CLIP, and executing semantic search with frame-level fine localization.

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
| **Embedding Model** | | |
| `DEFAULT_EMBEDDING_MODEL` | `microsoft/xclip-base-patch32` | HuggingFace multimodal model name |
| `DEFAULT_EMBEDDING_VERSION` | `1.0` | Model version tag stored with indexed vectors |
| `EMBEDDING_DIMENSION` | `512` | Vector embedding dimension size |
| `EMBEDDING_DEVICE` | `auto` | Device for inference: `auto`, `cuda:0`, `mps`, or `cpu` |
| **Chunking Parameters** | | |
| `CHUNK_THRESHOLD_SEC` | `45.0` | Clips shorter than this threshold become a single chunk |
| `SLIDING_WINDOW_SEC` | `10.0` | Window length (seconds) for sub-chunking long scenes |
| `SLIDING_OVERLAP_RATIO`| `0.5` | Overlap ratio (0.0 to 1.0) between sliding windows |

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

### 2. Multi-Provider Ingestion & Narrative Search Scripts

```bash
# Ingest curated narrative assets from Pexels, Pixabay, and Coverr
uv run python scripts/ingest_story_footage.py

# Query narrative story segments with ranked retrieval and fine localization
uv run python scripts/demo_narrative_search.py
```

### 3. Programmatic Usage

```python
import footage_engine as fe

# 1. Ingest media from stock provider or URL (with automatic deduplication)
item = fe.ingest(
    source_url="https://example.com/video.mp4",
    provider="pexels",
    source_id="123456",
    media_type="video",
)

# 2. Process pending items (scene detection, chunking, X-CLIP embedding, vector indexing)
processor = fe.BatchProcessor()
processor.process_all_pending()

# 3. Perform semantic search across indexed chunks
retrieval = fe.get_retrieval_api()
results = retrieval.search(query="cargo vessel navigating open ocean storm", top_k=5)

for res in results:
    print(f"Match: {res.chunk_id} | Score: {res.score:.4f} | [{res.start_ts}s - {res.end_ts}s]")

# 4. Fine localize the exact cut within a winning chunk
if results:
    start_cut, end_cut = retrieval.fine_localize(
        chunk_id=results[0].chunk_id,
        query="cargo vessel navigating open ocean storm",
        fps=1.0,
    )
    print(f"Refined cut timestamps: {start_cut:.2f}s - {end_cut:.2f}s")
```

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
